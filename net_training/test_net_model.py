import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import logging
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from net.net import LogicGuidedDiffusionForecast
from logics.logic_corrector import SimpleLogicCorrector
from logics.stl_attack_repair import AttackGenerator, STLGuidedRepairer, TeLExLearner
from smoothing.stl_utils import compute_stl_robustness, batch_stl_robustness, binary_classification
from analysis.stl_formulas import Always, Eventually, Atomic, STLAnd

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


class NetModelTester:
    def __init__(self, model_path, device='cpu', learn_thresholds=True):
        self.device = torch.device(device)
        self.model = None
        self.corrector = SimpleLogicCorrector()
        self.logic_names = self.corrector.get_logic_names()
        self.load_model(model_path)
        
        if learn_thresholds:
            self._learn_logic_thresholds()
    
    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        
        in_channels = checkpoint.get('in_channels', 7)
        forecast_horizon = checkpoint.get('forecast_horizon', 6)
        
        model_dir = os.path.dirname(path)
        config_path = os.path.join(model_dir, 'config.json')
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            model_config = config['model']
            diffusion_hidden = model_config.get('diffusion_hidden', 32)
            diffusion_levels = model_config.get('diffusion_levels', 2)
            seq_model_type = model_config.get('seq_model_type', 'seq2seq')
            seq_d_model = model_config.get('seq_d_model', 64)
            seq_n_heads = model_config.get('seq_n_heads', 2)
            seq_n_layers = model_config.get('seq_n_layers', 2)
            seq_dim_feedforward = model_config.get('seq_dim_feedforward', 512)
            num_diffusion_steps = model_config.get('num_diffusion_steps', 10)
            use_conditioned = model_config.get('use_conditioned_diffusion', True)
            logger.info(f"Loaded architecture from config.json: seq_type={seq_model_type}, hidden={diffusion_hidden}, d_model={seq_d_model}, layers={seq_n_layers}")
        else:
            diffusion_hidden = 32
            diffusion_levels = 2
            seq_model_type = 'seq2seq'
            seq_d_model = 64
            seq_n_heads = 2
            seq_n_layers = 2
            seq_dim_feedforward = 512
            num_diffusion_steps = 10
            use_conditioned = True
            logger.info("Config.json not found, using default reduced capacity architecture")
        
        self.model = LogicGuidedDiffusionForecast(
            in_channels=in_channels,
            diffusion_hidden=diffusion_hidden,
            diffusion_levels=diffusion_levels,
            seq_d_model=seq_d_model,
            seq_n_heads=seq_n_heads,
            seq_n_layers=seq_n_layers,
            forecast_horizon=forecast_horizon,
            num_diffusion_steps=num_diffusion_steps,
            use_conditioned_diffusion=use_conditioned,
            seq_model_type=seq_model_type,
            seq_dim_feedforward=seq_dim_feedforward
        )
        
        state_dict = checkpoint['model_state_dict']
        try:
            self.model.load_state_dict(state_dict)
        except RuntimeError as e:
            msg = str(e)
            if "condition_weight" in msg:
                # Older checkpoints without condition_weight; load non-strict
                self.model.load_state_dict(state_dict, strict=False)
                logger.info("Loaded checkpoint without condition_weight (non-strict).")
            else:
                raise
        self.model.to(self.device)
        self.model.eval()
        
        # Store whether model is conditioned (for test workflow decisions)
        self.is_conditioned = use_conditioned
        
        logger.info(f"Model loaded from {path}")
        logger.info(f"Model type: {'CONDITIONED' if use_conditioned else 'UNCONDITIONED'}")
    
    def _learn_logic_thresholds(self):
        # Load a subset of training data to learn thresholds
        ds_dir = 'pytorch_datasets'
        try:
            # Try loading training data first
            Y_train = np.load(os.path.join(ds_dir, 'Y_train.npy'))
            # Use first 1000 samples to learn thresholds
            Y_subset = Y_train[:1000]
        except FileNotFoundError:
            # Fallback to test data if training not available
            Y_test = np.load(os.path.join(ds_dir, 'Y_test.npy'))
            Y_subset = Y_test[:1000]
        
        # Transpose to (N, features, timesteps)
        Y_subset_t = np.transpose(Y_subset, (0, 2, 1))
        
        # Learn thresholds (silently)
        self.corrector.learn_thresholds_from_data(Y_subset_t, verbose=False)
    
    def evaluate(self, X_test, Y_test, logic_name=None, use_attacked=False, attack_type='gaussian', epsilon=1.5):
        X_test_np = X_test.cpu().numpy() if isinstance(X_test, torch.Tensor) else X_test
        feature_names = ['total_flow', 'avg_occupancy', 'avg_speed', 
                        'aq_NO2', 'aq_NOx', 'aq_O3', 'aq_PM25']
        
        X_input_np = None
        X_condition_np = None
        
        # If not already attacked and use_attacked=False, use clean data
        if not use_attacked:
            X_input_np = X_test_np
            # For conditioned models with clean data, use clean data itself as condition
            if self.is_conditioned:
                X_condition_np = X_test_np
        else:
            # Attack the data
            from logics.stl_attack_repair import AttackGenerator
            attacker = AttackGenerator(attack_type=attack_type, epsilon=epsilon, seed=None)
            X_input_np = attacker.attack(X_test_np)
            
            if self.is_conditioned:
                # For conditioned models: learn STL from clean data and repair attacked data
                learner = TeLExLearner(feature_names, timesteps=X_test_np.shape[2])
                learned_bounds = learner.learn_bounds_simple(X_test_np, feature_names, verbose=False)
                
                repairer = STLGuidedRepairer(feature_names, learned_bounds=learned_bounds)
                X_condition_np = repairer.repair(X_input_np, method='comprehensive', verbose=False)
        
        X_test_tensor = torch.FloatTensor(X_input_np).to(self.device)
        Y_test_tensor = torch.FloatTensor(Y_test).to(self.device)
        
        with torch.no_grad():
            if self.is_conditioned and X_condition_np is not None:
                # Conditioned model: use condition
                X_condition_tensor = torch.FloatTensor(X_condition_np).to(self.device)
                y_pred = self.model(X_test_tensor, x_condition=X_condition_tensor, add_noise=False)
            else:
                # Unconditioned model or no condition available: baseline
                y_pred = self.model(X_test_tensor, x_condition=None, add_noise=False)
        
        # Compute per-sample errors for std calculation
        mse_per_sample = torch.mean((y_pred - Y_test_tensor) ** 2, dim=(1, 2))  # (batch_size,)
        mae_per_sample = torch.mean(torch.abs(y_pred - Y_test_tensor), dim=(1, 2))  # (batch_size,)
        
        mse = torch.mean(mse_per_sample).item()
        mae = torch.mean(mae_per_sample).item()
        mse_std = torch.std(mse_per_sample).item()
        mae_std = torch.std(mae_per_sample).item()
        
        return {
            'mse': mse,
            'mae': mae,
            'mse_std': mse_std,
            'mae_std': mae_std,
            'predictions': y_pred.cpu().numpy(),
            'targets': Y_test_tensor.cpu().numpy()
        }
    
    def test_all_logics(self, X_test, Y_test, n_samples=100, attack_type='gaussian', epsilon=1.5):
        if not self.is_conditioned:
            logger.warning("test_all_logics skipped: Model is UNCONDITIONED (no logic conditioning)")
            return {}
        
        logger.info(f"Testing with {len(self.logic_names)} logic properties...")
        
        results = {}
        n_samples = min(n_samples, len(X_test))
        X_subset = X_test[:n_samples]
        Y_subset = Y_test[:n_samples]
        
        # Convert to numpy
        X_subset_np = X_subset.cpu().numpy() if isinstance(X_subset, torch.Tensor) else X_subset
        feature_names = ['total_flow', 'avg_occupancy', 'avg_speed', 
                        'aq_NO2', 'aq_NOx', 'aq_O3', 'aq_PM25']
        
        # Step 1: Learn STL bounds from clean data
        learner = TeLExLearner(feature_names, timesteps=X_subset_np.shape[2])
        learned_bounds = learner.learn_bounds_simple(X_subset_np, feature_names, verbose=False)
        
        # Step 2: Attack test data (same attack for all logics)
        attacker = AttackGenerator(attack_type=attack_type, epsilon=epsilon, seed=None)
        X_attacked_np = attacker.attack(X_subset_np)
        X_attacked_tensor = torch.FloatTensor(X_attacked_np).to(self.device)
        Y_subset_tensor = torch.FloatTensor(Y_subset).to(self.device)
        
        # Step 3: For each logic, repair with that specific logic and test
        for logic_name in self.logic_names:
            try:
                # Repair attacked data using THIS SPECIFIC LOGIC
                x_repaired_np = self.corrector.correct(X_attacked_np, logic_name=logic_name)
                X_repaired_tensor = torch.FloatTensor(x_repaired_np).to(self.device)
                
                # Test with [attacked | repaired_with_this_logic]
                with torch.no_grad():
                    y_pred = self.model(X_attacked_tensor, x_condition=X_repaired_tensor, add_noise=False)
                
                # Compute per-sample errors for std calculation
                mse_per_sample = torch.mean((y_pred - Y_subset_tensor) ** 2, dim=(1, 2))
                mae_per_sample = torch.mean(torch.abs(y_pred - Y_subset_tensor), dim=(1, 2))
                
                mse = torch.mean(mse_per_sample).item()
                mae = torch.mean(mae_per_sample).item()
                mse_std = torch.std(mse_per_sample).item()
                mae_std = torch.std(mae_per_sample).item()
                
                results[logic_name] = {
                    'mse': mse,
                    'mae': mae,
                    'mse_std': mse_std,
                    'mae_std': mae_std,
                    'predictions': y_pred.cpu().numpy(),
                    'targets': Y_subset_tensor.cpu().numpy()
                }
                logger.info(f"  {logic_name}: MSE={mse:.6f}±{mse_std:.6f}, MAE={mae:.6f}±{mae_std:.6f}")
            except Exception as e:
                logger.warning(f"  {logic_name}: Error - {e}")
                results[logic_name] = {'error': str(e)}
        
        return results
    
    def test_adversarial(self, X_test, Y_test, n_samples=100, attack_type='gaussian', epsilon=1.5):
        logger.info(f"Testing with adversarial attacks: attack_type={attack_type}, epsilon={epsilon}")
        
        X_subset = X_test[:n_samples]
        Y_subset = Y_test[:n_samples]
        
        # Convert to numpy for attack/repair
        X_subset_np = X_subset.cpu().numpy() if isinstance(X_subset, torch.Tensor) else X_subset
        
        feature_names = ['total_flow', 'avg_occupancy', 'avg_speed', 
                        'aq_NO2', 'aq_NOx', 'aq_O3', 'aq_PM25']
        
        # Step 1: Attack the test data
        attacker = AttackGenerator(attack_type=attack_type, epsilon=epsilon, seed=None)
        X_attacked_np = attacker.attack(X_subset_np)
        X_attacked_tensor = torch.FloatTensor(X_attacked_np).to(self.device)
        Y_subset_tensor = torch.FloatTensor(Y_subset).to(self.device)
        
        if self.is_conditioned:
            # CONDITIONED MODEL: Learn STL bounds → Repair → Test with [attacked | repaired]
            learner = TeLExLearner(feature_names, timesteps=X_subset_np.shape[2])
            learned_bounds = learner.learn_bounds_simple(X_subset_np, feature_names, verbose=False)
            
            repairer = STLGuidedRepairer(feature_names, learned_bounds=learned_bounds)
            X_repaired_np = repairer.repair(X_attacked_np, method='comprehensive', verbose=False)
            X_repaired_tensor = torch.FloatTensor(X_repaired_np).to(self.device)
            
            with torch.no_grad():
                y_pred = self.model(X_attacked_tensor, x_condition=X_repaired_tensor, add_noise=False)
            
            # Compute per-sample errors for std calculation
            mse_per_sample = torch.mean((y_pred - Y_subset_tensor) ** 2, dim=(1, 2))
            mae_per_sample = torch.mean(torch.abs(y_pred - Y_subset_tensor), dim=(1, 2))
            
            mse = torch.mean(mse_per_sample).item()
            mae = torch.mean(mae_per_sample).item()
            mse_std = torch.std(mse_per_sample).item()
            mae_std = torch.std(mae_per_sample).item()
            
            return {
                'mse': mse,
                'mae': mae,
                'mse_std': mse_std,
                'mae_std': mae_std,
                'predictions': y_pred.cpu().numpy(),
                'targets': Y_subset_tensor.cpu().numpy(),
                'learned_bounds': learned_bounds
            }
        else:
            # UNCONDITIONED MODEL: Just test on attacked data
            with torch.no_grad():
                y_pred = self.model(X_attacked_tensor, x_condition=None, add_noise=False)
            
            # Compute per-sample errors for std calculation
            mse_per_sample = torch.mean((y_pred - Y_subset_tensor) ** 2, dim=(1, 2))
            mae_per_sample = torch.mean(torch.abs(y_pred - Y_subset_tensor), dim=(1, 2))
            
            mse = torch.mean(mse_per_sample).item()
            mae = torch.mean(mae_per_sample).item()
            mse_std = torch.std(mse_per_sample).item()
            mae_std = torch.std(mae_per_sample).item()
        
        return {
                'mse': mse,
                'mae': mae,
                'mse_std': mse_std,
                'mae_std': mae_std,
                'predictions': y_pred.cpu().numpy(),
                'targets': Y_subset_tensor.cpu().numpy()
        }
    
    def visualize_predictions(self, X_test, Y_test, logic_name=None, n_samples=10, save_path='plots/net_model_predictions.png',
                             use_attacked=False, attack_type='gaussian', epsilon=1.5):
        X_subset = X_test[:n_samples]
        Y_subset = Y_test[:n_samples]
        
        result = self.evaluate(X_subset, Y_subset, logic_name=logic_name, 
                              use_attacked=use_attacked, attack_type=attack_type, epsilon=epsilon)
        predictions = result['predictions']  # (n_samples, n_features, horizon)
        targets = result['targets']
        
        # Concatenate samples along time axis to create long trajectory
        # Shape: (n_samples, n_features, horizon) -> (n_features, n_samples * horizon)
        predictions_concat = predictions.transpose(1, 0, 2).reshape(predictions.shape[1], -1)
        targets_concat = targets.transpose(1, 0, 2).reshape(targets.shape[1], -1)
        
        total_timesteps = predictions_concat.shape[1]
        time_steps = np.arange(total_timesteps)
        
        # Feature names for better labels
        feature_names = ['total_flow', 'avg_occupancy', 'avg_speed', 
                        'aq_NO2', 'aq_NOx', 'aq_O3', 'aq_PM25']
        
        # Create subplots: one per feature (first 4 features for clarity)
        n_features_to_plot = min(4, predictions.shape[1])
        fig, axes = plt.subplots(n_features_to_plot, 1, figsize=(16, 3 * n_features_to_plot))
        if n_features_to_plot == 1:
            axes = [axes]
        
        for feat_idx in range(n_features_to_plot):
            ax = axes[feat_idx]
            
            # Plot ground truth and predictions
            ax.plot(time_steps, targets_concat[feat_idx, :], 
                   label='Ground Truth', linestyle='--', linewidth=2, alpha=0.8, color='blue')
            ax.plot(time_steps, predictions_concat[feat_idx, :], 
                   label='Prediction', linewidth=2, alpha=0.8, color='orange')
            
            # Add vertical lines to show sample boundaries
            horizon = predictions.shape[2]
            for i in range(1, n_samples):
                ax.axvline(x=i * horizon, color='gray', linestyle=':', alpha=0.5, linewidth=1)
            
            # Labels and title
            feature_name = feature_names[feat_idx] if feat_idx < len(feature_names) else f'Feature {feat_idx}'
            title = f'{feature_name}'
            if logic_name and feat_idx == 0:
                title += f' (Logic: {logic_name})'
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_ylabel('Normalized Value', fontsize=10)
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)
            
            # Only show x-label on bottom plot
            if feat_idx == n_features_to_plot - 1:
                ax.set_xlabel(f'Time Steps (concatenated {n_samples} samples, {horizon} steps each)', fontsize=10)
        
        # Overall title
        title_text = f'Prediction Visualization ({n_samples} samples concatenated)'
        if use_attacked:
            title_text += f' - Attacked Input ({attack_type}, ε={epsilon})'
        if logic_name:
            title_text += f' - With Logic: {logic_name}'
        fig.suptitle(title_text, fontsize=14, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Predictions visualization saved to {save_path}")
    
    def test_stl_satisfaction(self, X_test, Y_test, n_samples=100, attack_type='gaussian', epsilon=1.5, use_attacked=False, stl_formulas=None, directly_enforced=None, other_formulas=None):
        
        X_subset = X_test[:n_samples]
        Y_subset = Y_test[:n_samples]
        
        feature_names = ['total_flow', 'avg_occupancy', 'avg_speed', 
                        'aq_NO2', 'aq_NOx', 'aq_O3', 'aq_PM25']
        
        # Convert to numpy for attack if needed
        X_subset_np = X_subset.cpu().numpy() if isinstance(X_subset, torch.Tensor) else X_subset
        Y_subset_np = Y_subset.cpu().numpy() if isinstance(Y_subset, torch.Tensor) else Y_subset
        
        X_attacked_np = None
        X_repaired_np = None
        
        # Attack data if requested
        if use_attacked:
            # Use fixed seed for reproducibility (42 for adversarial testing)
            attacker = AttackGenerator(attack_type=attack_type, epsilon=epsilon, seed=42)
            X_attacked_np = attacker.attack(X_subset_np.copy())
            
            if self.is_conditioned:
                # For conditioned models: learn STL from clean data and repair attacked data
                learner = TeLExLearner(feature_names, timesteps=X_subset_np.shape[2])
                learned_bounds = learner.learn_bounds_simple(X_subset_np, feature_names, verbose=False)
                
                repairer = STLGuidedRepairer(feature_names, learned_bounds=learned_bounds)
                X_repaired_np = repairer.repair(X_attacked_np, method='comprehensive', verbose=False)
        
        # Prepare input tensors
        if use_attacked:
            X_input_np = X_attacked_np
        else:
            X_input_np = X_subset_np
        
        X_subset_tensor = torch.FloatTensor(X_input_np).to(self.device)
        Y_subset_tensor = torch.FloatTensor(Y_subset).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            if self.is_conditioned:
                if use_attacked and X_repaired_np is not None:
                    # Conditioned model with attacked data: Use attacked data as input, repaired data as condition
                    X_repaired_tensor = torch.FloatTensor(X_repaired_np).to(self.device)
                    y_pred = self.model(X_subset_tensor, x_condition=X_repaired_tensor, add_noise=False)
                else:
                    # Conditioned model with clean data: Use clean data itself as condition
                    y_pred = self.model(X_subset_tensor, x_condition=X_subset_tensor, add_noise=False)
            else:
                # Unconditioned model: No conditioning
                y_pred = self.model(X_subset_tensor, x_condition=None, add_noise=False)
        
        # Convert predictions to numpy: (batch, features, horizon)
        y_pred_np = y_pred.cpu().numpy()
        y_gt_np = Y_subset_tensor.cpu().numpy()
        horizon = y_pred_np.shape[2]
        
        # Debug: Compare prediction statistics
        if use_attacked:
            # Store prediction stats for comparison
            pred_mean = np.mean(y_pred_np)
            pred_std = np.std(y_pred_np)
            pred_min = np.min(y_pred_np)
            pred_max = np.max(y_pred_np)
        else:
            pred_mean = np.mean(y_pred_np)
            pred_std = np.std(y_pred_np)
            pred_min = np.min(y_pred_np)
            pred_max = np.max(y_pred_np)
        
        # PER-SAMPLE STL: Generate formulas individually for each sample from its own ground truth
        # Then compute STL satisfaction on that sample's predictions
        from smoothing.test_smoothing import create_meaningful_stl_formulas
        from collections import defaultdict
        
        # Collect results across all samples for each formula
        formula_results_all = defaultdict(lambda: {'robustness': [], 'satisfaction': []})
        directly_enforced = None  # Will be set from first sample
        other_formulas_set = None
        
        logger.info(f"Generating per-sample STL formulas for {n_samples} samples...")
        
        import sys
        import os
        
        for i in range(n_samples):
            if (i + 1) % 100 == 0:
                logger.info(f"  Processed {i + 1}/{n_samples} samples...")
            
            # Generate STL formulas from THIS sample's ground truth only
            y_gt_sample = y_gt_np[i:i+1]  # (1, features, horizon)
            
            # Suppress TeLEx verbose output
            old_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            try:
                sample_stl_formulas, sample_enforced, sample_other = create_meaningful_stl_formulas(
                    y_gt_sample, feature_names, horizon=horizon, verbose=False
                )
            finally:
                sys.stdout.close()
                sys.stdout = old_stdout
            
            # Set enforced/other lists from first sample (same structure for all samples)
            if directly_enforced is None:
                directly_enforced = sample_enforced
                other_formulas_set = sample_other
            
            # Compute robustness for THIS sample using ITS formulas
            y_pred_sample = y_pred_np[i].T  # (horizon, features)
            
            for formula_name, formula in sample_stl_formulas.items():
                rho = compute_stl_robustness(y_pred_sample, formula, feature_names)
                formula_results_all[formula_name]['robustness'].append(rho)
                formula_results_all[formula_name]['satisfaction'].append(rho >= 0.0)
        
        logger.info(f"  Completed all {n_samples} samples!")
        
        # Aggregate results across all samples
        results = {}
        results_enforced = {}
        results_other = {}
        
        for formula_name, data in formula_results_all.items():
            robustness_scores = np.array(data['robustness'])
            satisfaction_flags = np.array(data['satisfaction'])
            
            satisfaction_rate = np.mean(satisfaction_flags)
            avg_robustness = np.mean(robustness_scores)
            min_robustness = np.min(robustness_scores)
            worst_10pct_robustness = np.percentile(robustness_scores, 10)
            
            result = {
                'satisfaction_rate': satisfaction_rate,
                'avg_robustness': avg_robustness,
                'min_robustness': min_robustness,
                'worst_10pct_robustness': worst_10pct_robustness,
                'robustness_scores': robustness_scores
            }
            results[formula_name] = result
            
            # Separate into enforced vs other
            if formula_name in directly_enforced:
                results_enforced[formula_name] = result
            else:
                results_other[formula_name] = result
        
        # Print main table: Directly enforced properties
        logger.info("\n" + "=" * 80)
        if use_attacked:
            logger.info("STL SATISFACTION/ROBUSTNESS RESULTS (Directly Enforced by Condition) - Adversarial Data")
        else:
            logger.info("STL SATISFACTION/ROBUSTNESS RESULTS (Directly Enforced by Condition) - Clean Data")
        logger.info("=" * 80)
        logger.info("These properties are directly enforced by the STL repair process:")
        logger.info("  - Reasonable bounds (Type 3): Learned min/max bounds for all features")
        logger.info("  - Flow-speed correlation (Type 2): flow_speed_correlation logic")
        logger.info("-" * 80)
        logger.info(f"{'Formula':<35} {'Sat%':>8} {'Avg ρ':>10} {'10th% ρ':>10} {'Min ρ':>10}")
        logger.info("-" * 80)
        
        for formula_name, result in results_enforced.items():
            logger.info(
                f"{formula_name:<35} "
                f"{result['satisfaction_rate']:>7.1%} "
                f"{result['avg_robustness']:>10.4f} "
                f"{result['worst_10pct_robustness']:>10.4f} "
                f"{result['min_robustness']:>10.4f}"
            )
        
        logger.info("-" * 80)
        if results_enforced:
            overall_satisfaction = np.mean([r['satisfaction_rate'] for r in results_enforced.values()])
            overall_avg_robustness = np.mean([r['avg_robustness'] for r in results_enforced.values()])
            overall_worst_10pct = np.mean([r['worst_10pct_robustness'] for r in results_enforced.values()])
            logger.info(f"{'OVERALL (Enforced)':<35} {overall_satisfaction:>7.1%} {overall_avg_robustness:>10.4f} {overall_worst_10pct:>10.4f}")
        logger.info("=" * 80)
        
        # Print separate table: Other properties
        if results_other:
            logger.info("\n" + "=" * 80)
            if use_attacked:
                logger.info("STL SATISFACTION/ROBUSTNESS RESULTS (Adversarial Data)")
            else:
                logger.info("STL SATISFACTION/ROBUSTNESS RESULTS (Clean Data)")
            logger.info("=" * 80)
            logger.info("These properties are NOT directly enforced but tested for generalization:")
            logger.info("  - Cross-signal consistency (Type 2): high_speed_low_occ")
            logger.info("  - Complex temporal logic (Type 4): Recovery, Stability, Persistence")
            logger.info("-" * 80)
            logger.info(f"{'Formula':<35} {'Sat%':>8} {'Avg ρ':>10} {'10th% ρ':>10} {'Min ρ':>10}")
            logger.info("-" * 80)
            
            for formula_name, result in results_other.items():
                logger.info(
                    f"{formula_name:<35} "
                    f"{result['satisfaction_rate']:>7.1%} "
                    f"{result['avg_robustness']:>10.4f} "
                    f"{result['worst_10pct_robustness']:>10.4f} "
                    f"{result['min_robustness']:>10.4f}"
                )
            
            logger.info("-" * 80)
            overall_satisfaction_other = np.mean([r['satisfaction_rate'] for r in results_other.values()])
            overall_avg_robustness_other = np.mean([r['avg_robustness'] for r in results_other.values()])
            overall_worst_10pct_other = np.mean([r['worst_10pct_robustness'] for r in results_other.values()])
            logger.info(f"{'OVERALL (Other)':<35} {overall_satisfaction_other:>7.1%} {overall_avg_robustness_other:>10.4f} {overall_worst_10pct_other:>10.4f}")
            logger.info("=" * 80)
        
        return results
    
    def test_stl_satisfaction_combined(self, X_test, Y_test, n_samples=100, attack_type='gaussian', epsilon=1.5):
        n_samples = min(n_samples, len(X_test))
        X_subset = X_test[:n_samples]
        Y_subset = Y_test[:n_samples]
        feature_names = ['total_flow', 'avg_occupancy', 'avg_speed', 
                        'aq_NO2', 'aq_NOx', 'aq_O3', 'aq_PM25']
        
        # Convert to numpy
        X_subset_np = X_subset.cpu().numpy() if isinstance(X_subset, torch.Tensor) else X_subset
        Y_subset_np = Y_subset.cpu().numpy() if isinstance(Y_subset, torch.Tensor) else Y_subset
        
        # Attack data
        from logics.stl_attack_repair import AttackGenerator, TeLExLearner, STLGuidedRepairer
        attacker = AttackGenerator(attack_type=attack_type, epsilon=epsilon, seed=42)
        X_attacked_np = attacker.attack(X_subset_np.copy())
        
        # For conditioned models, prepare repairs
        X_repaired_np = None
        if self.is_conditioned:
            learner = TeLExLearner(feature_names, timesteps=X_subset_np.shape[2])
            learned_bounds = learner.learn_bounds_simple(X_subset_np, feature_names, verbose=False)
            repairer = STLGuidedRepairer(feature_names, learned_bounds=learned_bounds)
            X_repaired_np = repairer.repair(X_attacked_np, method='comprehensive', verbose=False)
        
        # Get clean predictions
        X_clean_tensor = torch.FloatTensor(X_subset_np).to(self.device)
        with torch.no_grad():
            if self.is_conditioned:
                y_pred_clean = self.model(X_clean_tensor, x_condition=X_clean_tensor, add_noise=False)
            else:
                y_pred_clean = self.model(X_clean_tensor, x_condition=None, add_noise=False)
        y_pred_clean_np = y_pred_clean.cpu().numpy()
        
        # Get adversarial predictions
        X_attacked_tensor = torch.FloatTensor(X_attacked_np).to(self.device)
        with torch.no_grad():
            if self.is_conditioned:
                X_repaired_tensor = torch.FloatTensor(X_repaired_np).to(self.device)
                y_pred_attacked = self.model(X_attacked_tensor, x_condition=X_repaired_tensor, add_noise=False)
            else:
                y_pred_attacked = self.model(X_attacked_tensor, x_condition=None, add_noise=False)
        y_pred_attacked_np = y_pred_attacked.cpu().numpy()
        
        # Ground truth
        Y_subset_tensor = torch.FloatTensor(Y_subset_np).to(self.device)
        y_gt_np = Y_subset_tensor.cpu().numpy()
        horizon = y_gt_np.shape[2]
        
        # PER-SAMPLE STL: Generate formulas ONCE per sample, test BOTH clean and adversarial
        from smoothing.test_smoothing import create_meaningful_stl_formulas
        from collections import defaultdict
        
        formula_results_clean = defaultdict(lambda: {'robustness': [], 'satisfaction': []})
        formula_results_attacked = defaultdict(lambda: {'robustness': [], 'satisfaction': []})
        directly_enforced_final = None
        other_formulas_set = None
        
        logger.info(f"Generating per-sample STL formulas for {n_samples} samples...")
        logger.info("(Formulas generated ONCE per sample, then tested on both clean and adversarial predictions)")
        
        import sys
        import os
        
        for i in range(n_samples):
            if (i + 1) % 100 == 0:
                logger.info(f"  Processed {i + 1}/{n_samples} samples...")
            
            # Generate STL formulas from THIS sample's ground truth ONCE
            y_gt_sample = y_gt_np[i:i+1]  # (1, features, horizon)
            x_sample = X_subset_np[i:i+1]  # Could be (1, features, seq_len) or (1, seq_len, features)
            
            # Ensure x_sample has shape (1, features, seq_len) to match y_gt_sample (1, features, horizon)
            if x_sample.shape[1] != len(feature_names):
                # Shape is (1, seq_len, features) - transpose to (1, features, seq_len)
                x_sample = x_sample.transpose(0, 2, 1)
            
            # CONCATENATE input (24 timesteps) + output (6 timesteps) = 30 timesteps combined
            # This combined sequence is used for threshold mining (percentiles, bounds)
            # But formulas are still evaluated on the 6-timestep output predictions
            combined_sample = np.concatenate([x_sample, y_gt_sample], axis=2)  # (1, features, 30)
            
            # Suppress TeLEx verbose output
            old_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            try:
                sample_stl_formulas, sample_enforced, sample_other = create_meaningful_stl_formulas(
                    y_gt_sample, feature_names, horizon=horizon, verbose=False, X_data=combined_sample
                )
            finally:
                sys.stdout.close()
                sys.stdout = old_stdout
            
            # Optional: print formulas for first sample (disabled for compact output)
            
            if directly_enforced_final is None:
                directly_enforced_final = sample_enforced
                other_formulas_set = sample_other
            
            # Test BOTH clean and adversarial predictions with the SAME formulas
            y_pred_clean_sample = y_pred_clean_np[i].T  # (horizon, features)
            y_pred_attacked_sample = y_pred_attacked_np[i].T  # (horizon, features)
            
            for formula_name, formula in sample_stl_formulas.items():
                # Clean
                rho_clean = compute_stl_robustness(y_pred_clean_sample, formula, feature_names)
                formula_results_clean[formula_name]['robustness'].append(rho_clean)
                formula_results_clean[formula_name]['satisfaction'].append(rho_clean >= 0.0)
                
                # Adversarial
                rho_attacked = compute_stl_robustness(y_pred_attacked_sample, formula, feature_names)
                formula_results_attacked[formula_name]['robustness'].append(rho_attacked)
                formula_results_attacked[formula_name]['satisfaction'].append(rho_attacked >= 0.0)
        
        logger.info(f"  Completed all {n_samples} samples!")
        
        # Aggregate results for both
        def aggregate_results(formula_results, directly_enforced_set):
            results = {}
            results_enforced = {}
            results_other = {}
            
            for formula_name, data in formula_results.items():
                robustness_scores = np.array(data['robustness'])
                satisfaction_flags = np.array(data['satisfaction'])
                
                result = {
                    'satisfaction_rate': np.mean(satisfaction_flags),
                    'avg_robustness': np.mean(robustness_scores),
                    'min_robustness': np.min(robustness_scores),
                    'worst_10pct_robustness': np.percentile(robustness_scores, 10),
                    'robustness_scores': robustness_scores
                }
                results[formula_name] = result
                
                if formula_name in directly_enforced_set:
                    results_enforced[formula_name] = result
                else:
                    results_other[formula_name] = result
            
            return {'results_enforced': results_enforced, 'results_other': results_other}
        
        results_clean = aggregate_results(formula_results_clean, directly_enforced_final)
        results_attacked = aggregate_results(formula_results_attacked, directly_enforced_final)
        
        # Log results
        summary_clean = self._log_stl_results(results_clean, use_attacked=False)
        summary_attacked = self._log_stl_results(results_attacked, use_attacked=True)
        
        return results_clean, results_attacked, summary_clean, summary_attacked
    
    def _format_stl_readable(self, formula, feature_names):
        """Convert STL formula to human-readable string"""
        from analysis.stl_formulas import Always, Eventually, Atomic, STLAnd, STLOr, STLNot
        
        if isinstance(formula, Always):
            child_str = self._format_stl_readable(formula.child, feature_names)
            return f"G[{formula.t_start},{formula.t_end}]({child_str})"
        
        elif isinstance(formula, Eventually):
            child_str = self._format_stl_readable(formula.child, feature_names)
            return f"F[{formula.t_start},{formula.t_end}]({child_str})"
        
        elif isinstance(formula, STLNot):
            child_str = self._format_stl_readable(formula.child, feature_names)
            return f"¬({child_str})"
        
        elif isinstance(formula, STLAnd):
            left_str = self._format_stl_readable(formula.left, feature_names)
            right_str = self._format_stl_readable(formula.right, feature_names)
            return f"({left_str} ∧ {right_str})"
        
        elif isinstance(formula, STLOr):
            left_str = self._format_stl_readable(formula.left, feature_names)
            right_str = self._format_stl_readable(formula.right, feature_names)
            return f"({left_str} ∨ {right_str})"
        
        elif isinstance(formula, Atomic):
            # Determine which feature based on weight vector
            if formula.feature_idx is not None:
                feat_name = feature_names[formula.feature_idx]
                # Check if weight is negative (indicates upper bound: -x >= -c means x <= c)
                weight = formula.w[formula.feature_idx]
            else:
                # Multi-feature (e.g., difference between features)
                nonzero = [i for i, w in enumerate(formula.w) if w != 0]
                if len(nonzero) == 2:
                    feat_name = f"{feature_names[nonzero[0]]}-{feature_names[nonzero[1]]}"
                    weight = formula.w[nonzero[0]]
                else:
                    feat_name = "expr"
                    weight = 1.0
            
            # Format threshold and operator
            # Handle sign flips when weight is negative:
            # -x >= -c  →  x <= c
            # -x <= c   →  x >= -c
            if weight < 0:
                if formula.relop == '>=':
                    op = '<='
                    threshold = -formula.c
                elif formula.relop == '<=':
                    op = '>='
                    threshold = -formula.c
                else:
                    op = formula.relop if formula.relop else '>='
                    threshold = formula.c
            else:
                op = formula.relop if formula.relop else '>='
                threshold = formula.c
            
            return f"{feat_name} {op} {threshold:.3f}"
        
        else:
            return str(formula)
    
    def _log_stl_results(self, results, use_attacked=False):
        results_enforced = results['results_enforced']
        results_other_all = results['results_other']
        # Split out windowed range and trend formulas for separate tables
        results_window = {k: v for k, v in results_other_all.items() if "window_" in k}
        results_trend = {k: v for k, v in results_other_all.items() if "trend_" in k}
        results_other = {
            k: v for k, v in results_other_all.items()
            if "window_" not in k and "trend_" not in k
        }

        def _overall_summary(groups):
            all_values = []
            for g in groups:
                all_values.extend(g.values())
            if not all_values:
                return None
            return {
                'satisfaction': float(np.mean([r['satisfaction_rate'] for r in all_values])),
                'avg_rho': float(np.mean([r['avg_robustness'] for r in all_values])),
                'worst_10pct': float(np.mean([r['worst_10pct_robustness'] for r in all_values])),
                'min_rho': float(np.mean([r['min_robustness'] for r in all_values])),
            }
        
        if results_enforced:
            logger.info("\n" + "=" * 80)
            if use_attacked:
                logger.info("STL SATISFACTION/ROBUSTNESS RESULTS (Adversarial Data)")
            else:
                logger.info("STL SATISFACTION/ROBUSTNESS RESULTS (Clean Data)")
            logger.info("=" * 80)
            logger.info("-" * 80)
            logger.info(f"{'Formula':<35} {'Sat%':>8} {'Avg ρ':>10} {'10th% ρ':>10} {'Min ρ':>10}")
            logger.info("-" * 80)
            for formula_name, result in results_enforced.items():
                logger.info(
                    f"{formula_name:<35} "
                    f"{result['satisfaction_rate']:>7.1%} "
                    f"{result['avg_robustness']:>10.4f} "
                    f"{result['worst_10pct_robustness']:>10.4f} "
                    f"{result['min_robustness']:>10.4f}"
                )
            logger.info("-" * 80)
            overall_satisfaction = np.mean([r['satisfaction_rate'] for r in results_enforced.values()])
            overall_avg_robustness = np.mean([r['avg_robustness'] for r in results_enforced.values()])
            overall_worst_10pct = np.mean([r['worst_10pct_robustness'] for r in results_enforced.values()])
            logger.info(f"{'OVERALL (Enforced)':<35} {overall_satisfaction:>7.1%} {overall_avg_robustness:>10.4f} {overall_worst_10pct:>10.4f}")
            logger.info("=" * 80)

        if results_other:
            logger.info("\n" + "=" * 80)
            if use_attacked:
                logger.info("STL SATISFACTION/ROBUSTNESS RESULTS (Adversarial Data)")
            else:
                logger.info("STL SATISFACTION/ROBUSTNESS RESULTS (Clean Data)")
            logger.info("=" * 80)
            logger.info("-" * 80)
            logger.info(f"{'Formula':<35} {'Sat%':>8} {'Avg ρ':>10} {'10th% ρ':>10} {'Min ρ':>10}")
            logger.info("-" * 80)
            for formula_name, result in results_other.items():
                logger.info(
                    f"{formula_name:<35} "
                    f"{result['satisfaction_rate']:>7.1%} "
                    f"{result['avg_robustness']:>10.4f} "
                    f"{result['worst_10pct_robustness']:>10.4f} "
                    f"{result['min_robustness']:>10.4f}"
                )
            logger.info("-" * 80)
            overall_satisfaction = np.mean([r['satisfaction_rate'] for r in results_other.values()])
            overall_avg_robustness = np.mean([r['avg_robustness'] for r in results_other.values()])
            overall_worst_10pct = np.mean([r['worst_10pct_robustness'] for r in results_other.values()])
            logger.info(f"{'OVERALL (Other)':<35} {overall_satisfaction:>7.1%} {overall_avg_robustness:>10.4f} {overall_worst_10pct:>10.4f}")
            logger.info("=" * 80)

        if results_trend:
            logger.info("\n" + "=" * 80)
            logger.info("TREND DIRECTION")
            logger.info("=" * 80)
            logger.info(f"{'Formula':<35} {'Sat%':>8} {'Avg ρ':>10} {'10th% ρ':>10} {'Min ρ':>10}")
            logger.info("-" * 80)
            for formula_name, result in results_trend.items():
                logger.info(
                    f"{formula_name:<35} "
                    f"{result['satisfaction_rate']:>7.1%} "
                    f"{result['avg_robustness']:>10.4f} "
                    f"{result['worst_10pct_robustness']:>10.4f} "
                    f"{result['min_robustness']:>10.4f}"
                )
            logger.info("-" * 80)
            overall_satisfaction = np.mean([r['satisfaction_rate'] for r in results_trend.values()])
            overall_avg_robustness = np.mean([r['avg_robustness'] for r in results_trend.values()])
            overall_worst_10pct = np.mean([r['worst_10pct_robustness'] for r in results_trend.values()])
            logger.info(f"{'OVERALL (Trend)':<35} {overall_satisfaction:>7.1%} {overall_avg_robustness:>10.4f} {overall_worst_10pct:>10.4f}")
            logger.info("=" * 80)

        if results_window:
            logger.info("\n" + "=" * 80)
            logger.info("WINDOWED BOUNDS (3-step)")
            logger.info("=" * 80)
            logger.info(f"{'Formula':<35} {'Sat%':>8} {'Avg ρ':>10} {'10th% ρ':>10} {'Min ρ':>10}")
            logger.info("-" * 80)
            for formula_name, result in results_window.items():
                logger.info(
                    f"{formula_name:<35} "
                    f"{result['satisfaction_rate']:>7.1%} "
                    f"{result['avg_robustness']:>10.4f} "
                    f"{result['worst_10pct_robustness']:>10.4f} "
                    f"{result['min_robustness']:>10.4f}"
                )
            logger.info("-" * 80)
            overall_satisfaction = np.mean([r['satisfaction_rate'] for r in results_window.values()])
            overall_avg_robustness = np.mean([r['avg_robustness'] for r in results_window.values()])
            overall_worst_10pct = np.mean([r['worst_10pct_robustness'] for r in results_window.values()])
            logger.info(f"{'OVERALL (Window)':<35} {overall_satisfaction:>7.1%} {overall_avg_robustness:>10.4f} {overall_worst_10pct:>10.4f}")
            logger.info("=" * 80)

        # Aggregated over all formulas
        all_values = []
        for group in (results_enforced, results_other, results_trend, results_window):
            all_values.extend(group.values())
        if all_values:
            overall_satisfaction = np.mean([r['satisfaction_rate'] for r in all_values])
            overall_avg_robustness = np.mean([r['avg_robustness'] for r in all_values])
            overall_worst_10pct = np.mean([r['worst_10pct_robustness'] for r in all_values])
            logger.info("\n" + "=" * 80)
            logger.info("OVERALL (All Formulas)")
            logger.info("=" * 80)
            logger.info(f"{'Sat%':>8} {'Avg ρ':>10} {'10th% ρ':>10}")
            logger.info("-" * 80)
            logger.info(f"{overall_satisfaction:>7.1%} {overall_avg_robustness:>10.4f} {overall_worst_10pct:>10.4f}")
            logger.info("=" * 80)

        # Return overall summary for degradation calculations
        return _overall_summary([results_enforced, results_other, results_trend, results_window])


def main():
    parser = argparse.ArgumentParser(description='Test Net Model')
    parser.add_argument('--model', type=str, default='training_runs/run_20251228_201409/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                       help='Device to use for testing')
    parser.add_argument('--n_samples', type=int, default=1000,
                       help='Number of samples for logic testing')
    parser.add_argument('--n_vis', type=int, default=10,
                       help='Number of samples to concatenate for visualization')
    parser.add_argument('--attack-type', type=str, default='gaussian',
                       choices=['gaussian', 'uniform', 'false_high', 'false_low', 'temporal', 'targeted_feature',
                               'spike', 'dropout', 'spoofing', 'clipping', 'jitter', 'mixed'],
                       help='Attack type for adversarial testing (default: gaussian)')
    parser.add_argument('--epsilon', type=float, default=1.5,
                       help='Attack epsilon/magnitude (default: 1.5)')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    model_path = args.model
    
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}. Please provide a valid model path.")
        return
    
    # Allow switching between datasets (e.g., pytorch_datasets vs pytorch_datasets_cali)
    ds_dir = os.environ.get('DATASET_DIR', 'pytorch_datasets')
    X_test = np.load(os.path.join(ds_dir, 'X_clean_test.npy'))
    Y_test = np.load(os.path.join(ds_dir, 'Y_test.npy'))
    
    # Limit to first 5000 samples (don't load all 2.65M into memory!)
    max_test_samples = 5000
    X_test = X_test[:max_test_samples]
    Y_test = Y_test[:max_test_samples]
    
    with open(os.path.join(ds_dir, 'timeseries_meta.json')) as f:
        meta = json.load(f)
    
    X_test_t = np.transpose(X_test, (0, 2, 1))
    Y_test_t = np.transpose(Y_test, (0, 2, 1))
    
    tester = NetModelTester(model_path, device=device)
    
    logger.info("\n" + "=" * 60)
    if tester.is_conditioned:
        logger.info("CLEAN DATA EVALUATION")
        logger.info("=" * 60)
    else:
        logger.info("CLEAN DATA EVALUATION (Clean Data, Full 5000 samples, NO Conditioning)")
        logger.info("=" * 60)
    baseline = tester.evaluate(X_test_t, Y_test_t, use_attacked=False)
    logger.info(f"MSE: {baseline['mse']:.6f}±{baseline['mse_std']:.6f}")
    logger.info(f"MAE: {baseline['mae']:.6f}±{baseline['mae_std']:.6f}")
    
    logger.info("\n" + "=" * 60)
    logger.info("ADVERSARIAL TESTING")
    logger.info("=" * 60)
    
    adversarial_result = tester.test_adversarial(
        X_test_t, Y_test_t, 
        n_samples=args.n_samples,
        attack_type=args.attack_type,
        epsilon=args.epsilon
    )
    logger.info(f"MSE: {adversarial_result['mse']:.6f}±{adversarial_result['mse_std']:.6f}")
    logger.info(f"MAE: {adversarial_result['mae']:.6f}±{adversarial_result['mae_std']:.6f}")
    
    logger.info("\n" + "=" * 60)
    logger.info("TESTING WITH DIFFERENT LOGIC PROPERTIES")
    logger.info("=" * 60)
    if tester.is_conditioned:
        logic_results = tester.test_all_logics(
            X_test_t, Y_test_t, 
            n_samples=args.n_samples,
            attack_type=args.attack_type,
            epsilon=args.epsilon
        )
    else:
        logger.info("SKIPPED: Model is unconditioned (no logic conditioning available)")
        logic_results = {}
    
    # PER-SAMPLE STL: Generate formulas ONCE per sample from GT, test BOTH clean and adversarial  
    # This ensures consistent, fair comparison using the same formulas
    logger.info("\n" + "=" * 60)
    logger.info("STL SATISFACTION/ROBUSTNESS TESTING")
    logger.info("=" * 60)
    stl_results_clean, stl_results_attacked, stl_summary_clean, stl_summary_attacked = tester.test_stl_satisfaction_combined(
        X_test_t, Y_test_t,
        n_samples=args.n_samples,
        attack_type=args.attack_type,
        epsilon=args.epsilon
    )

    # Degradation summary
    logger.info("\n" + "=" * 60)
    logger.info("PERFORMANCE DEGRADATION (attacked - clean)")
    logger.info("=" * 60)
    logger.info(f"MSE: {adversarial_result['mse'] - baseline['mse']:+.6f}")
    logger.info(f"MAE: {adversarial_result['mae'] - baseline['mae']:+.6f}")
    if stl_summary_clean and stl_summary_attacked:
        sat_delta = stl_summary_attacked['satisfaction'] - stl_summary_clean['satisfaction']
        logger.info(f"STL Sat%: {sat_delta:+.2%}")
        logger.info(f"STL Avg ρ: {stl_summary_attacked['avg_rho'] - stl_summary_clean['avg_rho']:+.4f}")
        logger.info(f"STL 10th% ρ: {stl_summary_attacked['worst_10pct'] - stl_summary_clean['worst_10pct']:+.4f}")
        logger.info(f"STL Min ρ: {stl_summary_attacked['min_rho'] - stl_summary_clean['min_rho']:+.4f}")
    
    logger.info("\n" + "=" * 60)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("=" * 60)
    
    # Extract model name for save paths
    model_name = os.path.basename(os.path.dirname(model_path))
    
    # Generate visualization based on model type (using attacked data)
    if tester.is_conditioned:
        # Conditioned model: only generate conditioned visualization
        tester.visualize_predictions(X_test_t, Y_test_t, logic_name='traffic_flow_stable', n_samples=args.n_vis,
                                     save_path=f'plots/{model_name}_conditioned_concatenated.png',
                                     use_attacked=True, attack_type=args.attack_type, epsilon=args.epsilon)
    else:
        # Unconditioned model: only generate baseline visualization
        tester.visualize_predictions(X_test_t, Y_test_t, logic_name=None, n_samples=args.n_vis, 
                                     save_path=f'plots/{model_name}_baseline_concatenated.png',
                                     use_attacked=True, attack_type=args.attack_type, epsilon=args.epsilon)
    
    logger.info("\nTesting complete!")


if __name__ == "__main__":
    main()

