import numpy as np
import torch
import logging
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from .stl_utils import (
    batch_stl_robustness, 
    binary_classification, 
    majority_vote
)
from .certification import compute_certified_radius

logger = logging.getLogger(__name__)


class STLRandomizedSmoother:
    def __init__(
        self,
        model,
        stl_formulas: Dict,
        sigma: float = 0.1,
        n_samples: int = 100,
        device: str = 'cpu',
        feature_names: Optional[List[str]] = None
    ):
        self.model = model
        self.model.eval()
        
        self.sigma = sigma
        self.n_samples = n_samples
        self.device = torch.device(device)
        
        if feature_names is None:
            feature_names = ['total_flow', 'avg_occupancy', 'avg_speed',
                           'aq_NO2', 'aq_NOx', 'aq_O3', 'aq_PM25']
        self.feature_names = feature_names
        self.stl_formulas = stl_formulas
    
    def predict(self, x: torch.Tensor, logic_name: Optional[str] = None) -> torch.Tensor:
        with torch.no_grad():
            if logic_name is not None:
                y_pred = self.model(x, logic_name=logic_name, add_noise=False)
            else:
                y_pred = self.model(x, add_noise=False)
        
        return y_pred
    
    def _add_noise(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x) * self.sigma
        return x + noise
    
    def predict_smooth(
        self,
        x: torch.Tensor,
        logic_name: Optional[str] = None,
        x_condition: Optional[torch.Tensor] = None,
        return_all: bool = False
    ) -> Dict:
        all_predictions = []
        formula_robustness = {name: [] for name in self.stl_formulas.keys()}
        formula_classifications = {name: [] for name in self.stl_formulas.keys()}
        
        for i in range(self.n_samples):
            x_noisy = self._add_noise(x)
            
            with torch.no_grad():
                if x_condition is not None:
                    y_pred = self.model(x_noisy, x_condition=x_condition, add_noise=False)
                elif logic_name is not None:
                    y_pred = self.model(x_noisy, logic_name=logic_name, add_noise=False)
                else:
                    y_pred = self.model(x_noisy, add_noise=False)
            
            all_predictions.append(y_pred)
            
            for formula_name, formula in self.stl_formulas.items():
                rho_scores = batch_stl_robustness(y_pred, formula, self.feature_names)
                formula_robustness[formula_name].append(rho_scores)
                
                satisfied = binary_classification(rho_scores)
                formula_classifications[formula_name].append(satisfied)
        
        all_predictions = torch.stack(all_predictions)
        y_smooth = all_predictions.mean(dim=0)
        
        smoothed_classifications = {}
        proportion_satisfied = {}
        
        for formula_name in self.stl_formulas.keys():
            classifications = np.stack(formula_classifications[formula_name])
            if classifications.ndim == 1:
                classifications = classifications[:, None]
            
            batch_size = classifications.shape[1]
            batch_predictions = []
            batch_p_A = []
            
            for b in range(batch_size):
                pred, p_A, p_B = majority_vote(classifications[:, b])
                batch_predictions.append(pred)
                batch_p_A.append(p_A)
            
            smoothed_classifications[formula_name] = np.array(batch_predictions)
            proportion_satisfied[formula_name] = np.array(batch_p_A)
        
        results = {
            'y_smooth': y_smooth,
            'smoothed_classifications': smoothed_classifications,
            'proportion_satisfied': proportion_satisfied,
            'formula_robustness': formula_robustness,
        }
        
        if return_all:
            results['all_predictions'] = all_predictions
            results['all_classifications'] = formula_classifications
        
        return results
    
    def certify(
        self,
        x: torch.Tensor,
        alpha: float = 0.001,
        logic_name: Optional[str] = None,
        x_condition: Optional[torch.Tensor] = None,
        n_samples: Optional[int] = None,
        verbose: bool = False
    ) -> Dict:
        if n_samples is None:
            n_samples = self.n_samples
        
        min_samples = max(10, int(np.ceil(-np.log(alpha))))
        if n_samples < min_samples:
            logger.warning(f"WARNING: n_samples={n_samples} may be too small for alpha={alpha}")
            logger.warning(f"  Recommended: n_samples >= {min_samples}")
            logger.warning(f"  Confidence bounds may be unreliable!")
        
        temp_n_samples = self.n_samples
        self.n_samples = n_samples
        
        results = self.predict_smooth(x, logic_name=logic_name, x_condition=x_condition, return_all=True)
        
        self.n_samples = temp_n_samples
        
        batch_size = x.shape[0]
        certification_results = {}
        
        all_diagnostics = {name: [] for name in self.stl_formulas.keys()}
        
        for formula_name in self.stl_formulas.keys():
            classifications = np.stack(results['all_classifications'][formula_name])
            
            batch_certifications = []
            
            for b in range(batch_size):
                count_satisfied = int(np.sum(classifications[:, b]))
                count_violated = n_samples - count_satisfied
                
                p_A = count_satisfied / n_samples
                p_B = count_violated / n_samples
                
                if p_A <= 0.5:
                    radius = 0.0
                    prediction_label = 'ABSTAIN'
                    diag = {
                        'p_hat': p_A,
                        'p_lower': 0.0,
                        'count': count_satisfied,
                        'n_samples': n_samples,
                        'failed_reason': 'no_majority_sat',
                        'alpha': alpha,
                        'certified': False
                    }
                else:
                    radius, prediction, diag = compute_certified_radius(
                        p_A, p_B, self.sigma, n_samples, alpha, return_diagnostics=True
                    )
                    prediction_label = 'SATISFIED' if radius > 0 else 'ABSTAIN'
                
                batch_certifications.append({
                    'prediction': prediction_label,
                    'certified_radius': radius,
                    'p_satisfied': p_A,
                    'p_violated': p_B,
                    'diagnostics': diag
                })
                
                all_diagnostics[formula_name].append(diag)
            
            certification_results[formula_name] = batch_certifications
        
        if verbose and batch_size == 1:
            self._print_certification_diagnostics(all_diagnostics, formula_name=None)
        
        return certification_results
    
    def _print_certification_diagnostics(self, diagnostics_dict, formula_name=None):
        print("\n" + "="*80)
        print("CERTIFICATION DIAGNOSTICS")
        print("="*80)
        
        formulas_to_show = [formula_name] if formula_name else diagnostics_dict.keys()
        
        for fname in formulas_to_show:
            if fname not in diagnostics_dict:
                continue
            diag = diagnostics_dict[fname][0]
            
            print(f"\nFormula: {fname}")
            print(f"  Empirical p_hat:    {diag['p_hat']:.4f}")
            print(f"  Lower bound p_lower: {diag['p_lower']:.4f}")
            print(f"  Count/N:            {diag['count']}/{diag['n_samples']}")
            print(f"  Alpha:              {diag['alpha']}")
            
            if 'failed_reason' in diag:
                print(f"  FAILED: {diag['failed_reason']}")
            elif diag.get('certified', False):
                print(f"  Status:             CERTIFIED")
            else:
                print(f"  Status:             NOT CERTIFIED")
        
        print("="*80)
    
    def certify_batch(
        self,
        X: torch.Tensor,
        alpha: float = 0.001,
        logic_name: Optional[str] = None,
        n_samples: Optional[int] = None,
        batch_size: int = 32
    ) -> Dict:
        if n_samples is None:
            n_samples = self.n_samples
        
        logger.info(f"Certifying {len(X)} samples in batches of {batch_size}")
        
        all_results = {formula_name: [] for formula_name in self.stl_formulas.keys()}
        
        n_batches = (len(X) + batch_size - 1) // batch_size
        
        for i in tqdm(range(n_batches), desc="Certifying batches"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(X))
            
            x_batch = X[start_idx:end_idx].to(self.device)
            
            batch_results = self.certify(x_batch, alpha, logic_name, n_samples)
            
            for formula_name, certifications in batch_results.items():
                all_results[formula_name].extend(certifications)
        
        aggregate_results = self._compute_aggregate_stats(all_results)
        
        return aggregate_results
    
    def _compute_aggregate_stats(self, all_results: Dict) -> Dict:
        aggregate = {}
        
        for formula_name, certifications in all_results.items():
            radii = [c['certified_radius'] for c in certifications]
            predictions = [c['prediction'] for c in certifications]
            p_satisfied = [c['p_satisfied'] for c in certifications]
            p_hats = [c['diagnostics']['p_hat'] for c in certifications]
            p_lowers = [c['diagnostics']['p_lower'] for c in certifications]
            
            certified_count = sum(1 for r in radii if r > 0)
            satisfied_count = sum(1 for p in predictions if p == 'SATISFIED')
            
            failed_reasons = {}
            for c in certifications:
                if 'failed_reason' in c['diagnostics']:
                    reason = c['diagnostics']['failed_reason']
                    failed_reasons[reason] = failed_reasons.get(reason, 0) + 1
            
            certified_radii = [r for r in radii if r > 0]
            aggregate[formula_name] = {
                'certified_rate': certified_count / len(certifications),
                'satisfaction_rate': satisfied_count / len(certifications),
                'avg_certified_radius': np.mean(certified_radii) if certified_count > 0 else 0.0,
                'median_certified_radius': np.median(certified_radii) if certified_count > 0 else 0.0,
                'p75_certified_radius': np.percentile(certified_radii, 75) if certified_count > 0 else 0.0,
                'min_certified_radius': np.min(certified_radii) if certified_count > 0 else 0.0,
                'max_certified_radius': np.max(certified_radii) if certified_count > 0 else 0.0,
                'std_radius': np.std(radii),
                'avg_p_satisfied': np.mean(p_satisfied),
                'avg_p_hat': np.mean(p_hats),
                'avg_p_lower': np.mean(p_lowers),
                'p_hat_std': np.std(p_hats),
                'radius_std': np.std(radii),
                'failed_certifications': failed_reasons,
                'all_certifications': certifications
            }
        
        return aggregate
