import numpy as np
import sys
import os
from typing import Dict, List, Tuple, Optional, Union
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from telex import synth
    from telex import stl as telex_stl
    from telex import parametrizer
    TELEX_AVAILABLE = True
except ImportError:
    TELEX_AVAILABLE = False
    print("Warning: TeLEx not available. STL learning will be limited.")

from logics.logic_corrector import SimpleLogicCorrector

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


class STLTemplateLibrary:
    def __init__(self, timesteps=6):
        self.timesteps = timesteps
        
    def get_always_lower_template(self, feature_name, min_val=-500, max_val=500):
        return f"G[0,{self.timesteps-1}]({feature_name} >= a? {min_val};{max_val})"
    
    def get_always_upper_template(self, feature_name, min_val=-500, max_val=500):
        return f"G[0,{self.timesteps-1}]({feature_name} <= a? {min_val};{max_val})"
    
    def get_always_bounded_templates(self, feature_name, min_val=-500, max_val=500):
        lower = self.get_always_lower_template(feature_name, min_val, max_val)
        upper = self.get_always_upper_template(feature_name, min_val, max_val)
        return {'lower': lower, 'upper': upper}
    
    def get_eventually_lower_template(self, feature_name, min_val=-10, max_val=10):
        return f"F[0,{self.timesteps-1}]({feature_name} >= a? {min_val};{max_val})"
    
    def get_eventually_upper_template(self, feature_name, min_val=-10, max_val=10):
        return f"F[0,{self.timesteps-1}]({feature_name} <= a? {min_val};{max_val})"


class TeLExLearner:
    def __init__(self, feature_names, timesteps=6, method='gradient'):
        self.feature_names = feature_names
        self.timesteps = timesteps
        self.method = method
        self.template_lib = STLTemplateLibrary(timesteps)
        self.learned_params = {}
        
        if not TELEX_AVAILABLE:
            logger.warning("TeLEx not available. Will use simple min/max bounds.")
    
    def _convert_to_telex_format(self, data, feature_idx):
        n_samples = data.shape[0]
        traces = []
        
        for i in range(n_samples):
            trace = {}
            for fname in self.feature_names:
                if fname in feature_idx:
                    idx = feature_idx[fname]
                    trace[fname] = data[i, idx, :].tolist()
            traces.append(trace)
        
        return traces
    
    def learn_bounds_simple(self, data, feature_names, verbose=True):
        bounds = {}
        
        if verbose:
            print("\n" + "="*80)
            print("LEARNING STL BOUNDS (Simple Min/Max Method)")
            print("="*80)
        
        for idx, feature_name in enumerate(feature_names):
            if idx >= data.shape[1]:
                continue
            
            feature_data = data[:, idx, :].flatten()
            lower_bound = np.min(feature_data)
            upper_bound = np.max(feature_data)
            
            bounds[feature_name] = {
                'lower': lower_bound,
                'upper': upper_bound
            }
            
            if verbose:
                print(f"{feature_name:>15s}: [{lower_bound:>8.4f}, {upper_bound:>8.4f}]")
        
        if verbose:
            print("="*80 + "\n")
        self.learned_params = bounds
        return bounds
    
    def learn_bounds_telex(self, data, feature_names, templates='always'):
        if not TELEX_AVAILABLE:
            logger.warning("TeLEx not available, falling back to simple min/max")
            return self.learn_bounds_simple(data, feature_names)
        
        feature_idx = {name: idx for idx, name in enumerate(feature_names)}
        traces = self._convert_to_telex_format(data, feature_idx)
        
        bounds = {}
        
        print("\n" + "="*80)
        print("LEARNING STL BOUNDS (TeLEx Synthesis)")
        print("="*80)
        print(f"Method: {self.method}")
        print(f"Samples: {len(traces)}")
        print(f"Template type: {templates}")
        print("="*80 + "\n")
        
        for feature_name in feature_names:
            print(f"Learning bounds for {feature_name}...")
            
            try:
                if templates in ['always', 'both']:
                    lower_template = self.template_lib.get_always_lower_template(feature_name)
                    upper_template = self.template_lib.get_always_upper_template(feature_name)
                    
                    parsed_lower = telex_stl.parse(lower_template)
                    parsed_upper = telex_stl.parse(upper_template)
                    
                    learned_lower = synth.synthSTLParam(parsed_lower, traces, method=self.method)
                    learned_upper = synth.synthSTLParam(parsed_upper, traces, method=self.method)
                    
                    lower_params = parametrizer.getParams(learned_lower)
                    upper_params = parametrizer.getParams(learned_upper)
                    
                    bounds[feature_name] = {
                        'lower': lower_params.get('a', np.min(data[:, feature_idx[feature_name], :])),
                        'upper': upper_params.get('a', np.max(data[:, feature_idx[feature_name], :]))
                    }
                    
                    print(f"  Learned: [{bounds[feature_name]['lower']:.4f}, {bounds[feature_name]['upper']:.4f}]")
                
            except Exception as e:
                logger.warning(f"TeLEx failed for {feature_name}: {e}")
                feature_data = data[:, feature_idx[feature_name], :].flatten()
                bounds[feature_name] = {
                    'lower': np.min(feature_data),
                    'upper': np.max(feature_data)
                }
                print(f"  Fallback: [{bounds[feature_name]['lower']:.4f}, {bounds[feature_name]['upper']:.4f}]")
        
        print("\n" + "="*80 + "\n")
        self.learned_params = bounds
        return bounds


class AttackGenerator:
    def __init__(self, attack_type='gaussian', epsilon=0.5, seed=42):
        self.attack_type = attack_type
        self.epsilon = epsilon
        self.seed = seed
        np.random.seed(seed)
        
        self.attack_methods = {
            'gaussian': self._gaussian_attack,
            'false_high': self._false_high_attack,
            'false_low': self._false_low_attack,
            'targeted_feature': self._targeted_feature_attack,
            'temporal': self._temporal_attack,
            'uniform': self._uniform_attack,
            'spike': self._spike_attack,
            'dropout': self._dropout_attack,
            'spoofing': self._spoofing_attack,
            'clipping': self._clipping_attack,
            'jitter': self._jitter_attack,
            'mixed': self._mixed_attack,
        }
    
    def attack(self, x, target_features=None, target_timesteps=None):
        if self.attack_type not in self.attack_methods:
            logger.error(f"Unknown attack type: {self.attack_type}")
            return x.copy()
        
        return self.attack_methods[self.attack_type](
            x, 
            target_features=target_features, 
            target_timesteps=target_timesteps
        )
    
    def _mixed_attack(self, x, target_features=None, target_timesteps=None):
        """Randomly choose one of several structured attacks per call.

        Uses the same epsilon but picks among dropout, false_high,
        temporal, and spoofing.
        """
        choices = ['dropout', 'false_high', 'temporal', 'spoofing']
        chosen = np.random.choice(choices)
        logger.info(f"[AttackGenerator] mixed attack chose: {chosen}")
        return self.attack_methods[chosen](
            x,
            target_features=target_features,
            target_timesteps=target_timesteps,
        )
    
    def _gaussian_attack(self, x, target_features=None, target_timesteps=None):
        x_tilde = x.copy()
        noise = np.random.normal(0, self.epsilon, x.shape)
        
        if target_features is not None:
            mask = np.zeros_like(x, dtype=bool)
            for feat_idx in target_features:
                mask[:, feat_idx, :] = True
            noise = noise * mask
        
        if target_timesteps is not None:
            mask = np.zeros_like(x, dtype=bool)
            for t in target_timesteps:
                mask[:, :, t] = True
            noise = noise * mask
        
        x_tilde = x_tilde + noise
        return x_tilde
    
    def _uniform_attack(self, x, target_features=None, target_timesteps=None):
        x_tilde = x.copy()
        noise = np.random.uniform(-self.epsilon, self.epsilon, x.shape)
        
        if target_features is not None:
            mask = np.zeros_like(x, dtype=bool)
            for feat_idx in target_features:
                mask[:, feat_idx, :] = True
            noise = noise * mask
        
        if target_timesteps is not None:
            mask = np.zeros_like(x, dtype=bool)
            for t in target_timesteps:
                mask[:, :, t] = True
            noise = noise * mask
        
        x_tilde = x_tilde + noise
        return x_tilde
    
    def _false_high_attack(self, x, target_features=None, target_timesteps=None):
        x_tilde = x.copy()
        factor = np.random.uniform(1.5, 1.5 + self.epsilon * 2, x.shape)
        
        attack_prob = min(1.0, self.epsilon / 5.0)
        attack_mask = np.random.random(x.shape) < attack_prob
        
        if target_features is None:
            target_features = range(x.shape[1])
        
        for feat_idx in target_features:
            if target_timesteps is None:
                x_tilde[:, feat_idx, :] = np.where(
                    attack_mask[:, feat_idx, :],
                    x_tilde[:, feat_idx, :] * factor[:, feat_idx, :],
                    x_tilde[:, feat_idx, :]
                )
            else:
                for t in target_timesteps:
                    if attack_mask[:, feat_idx, t].any():
                        x_tilde[:, feat_idx, t] = np.where(
                            attack_mask[:, feat_idx, t],
                            x_tilde[:, feat_idx, t] * factor[:, feat_idx, t],
                            x_tilde[:, feat_idx, t]
                        )
        
        return x_tilde
    
    def _false_low_attack(self, x, target_features=None, target_timesteps=None):
        x_tilde = x.copy()
        factor = np.random.uniform(0.5 - self.epsilon * 0.4, 0.5, x.shape)
        
        attack_prob = min(1.0, self.epsilon / 5.0)
        attack_mask = np.random.random(x.shape) < attack_prob
        
        if target_features is None:
            target_features = range(x.shape[1])
        
        for feat_idx in target_features:
            if target_timesteps is None:
                x_tilde[:, feat_idx, :] = np.where(
                    attack_mask[:, feat_idx, :],
                    x_tilde[:, feat_idx, :] * factor[:, feat_idx, :],
                    x_tilde[:, feat_idx, :]
                )
            else:
                for t in target_timesteps:
                    if attack_mask[:, feat_idx, t].any():
                        x_tilde[:, feat_idx, t] = np.where(
                            attack_mask[:, feat_idx, t],
                            x_tilde[:, feat_idx, t] * factor[:, feat_idx, t],
                            x_tilde[:, feat_idx, t]
                        )
        
        return x_tilde
    
    def _targeted_feature_attack(self, x, target_features=None, target_timesteps=None):
        if target_features is None:
            target_features = [0]
        
        return self._gaussian_attack(x, target_features=target_features, target_timesteps=target_timesteps)
    
    def _temporal_attack(self, x, target_features=None, target_timesteps=None):
        x_tilde = x.copy()
        n_timesteps = x.shape[2]
        
        if target_timesteps is None:
            start = n_timesteps // 3
            end = 2 * n_timesteps // 3
            target_timesteps = list(range(start, end))
        else:
            start = min(target_timesteps)
            end = max(target_timesteps) + 1
        
        window_size = end - start
        if window_size < 2:
            return x_tilde

        min_segment_size = max(2, window_size // 3)  # At least 1/3 of window
        segment_size = max(min_segment_size, int(window_size * self.epsilon * 1.5))
        if segment_size >= window_size:
            segment_size = window_size // 2
        
        shift_target = n_timesteps - segment_size
        
        if shift_target <= start:
            shift_target = start + segment_size
            if shift_target + segment_size > n_timesteps:
                segment_size = min(segment_size, (n_timesteps - start) // 2)
                shift_target = n_timesteps - segment_size
        
        if target_features is None:
            target_features = range(x.shape[1])
        
        for batch_idx in range(x.shape[0]):
            for feat_idx in target_features:
                segment = x_tilde[batch_idx, feat_idx, start:start+segment_size].copy()
                fill_value = segment[0]
                x_tilde[batch_idx, feat_idx, shift_target:shift_target+segment_size] = segment
                x_tilde[batch_idx, feat_idx, start:shift_target] = fill_value
        
        return x_tilde
    
    def _spike_attack(self, x, target_features=None, target_timesteps=None):
        x_tilde = x.copy()
        n_samples, n_features, n_timesteps = x.shape
        
        if target_timesteps is None:
            target_timesteps = list(range(n_timesteps))
        
        if target_features is None:
            target_features = range(n_features)

        for batch_idx in range(n_samples):
            for feat_idx in target_features:
                feature_std = np.std(x[batch_idx, feat_idx, :])
                spike_magnitude = self.epsilon * feature_std
                
                spike_prob = min(0.3, self.epsilon / 10.0)  # Cap at 30% of timesteps
                spike_mask = np.random.random(n_timesteps) < spike_prob
                
                for t in target_timesteps:
                    if spike_mask[t]:
                        direction = np.random.choice([-1, 1])
                        x_tilde[batch_idx, feat_idx, t] += direction * spike_magnitude * np.random.uniform(1.0, 2.0)
        
        return x_tilde
    
    def _dropout_attack(self, x, target_features=None, target_timesteps=None):
        x_tilde = x.copy()
        n_samples, n_features, n_timesteps = x.shape
        
        if target_timesteps is None:
            target_timesteps = list(range(n_timesteps))
        
        if target_features is None:
            target_features = range(n_features)
        
        dropout_prob = min(0.5, self.epsilon / 5.0)
        segment_len = max(1, int(round(self.epsilon)))
        
        timesteps = list(target_timesteps)
        for batch_idx in range(n_samples):
            for feat_idx in target_features:
                t_idx = 0
                while t_idx < len(timesteps):
                    if np.random.random() < dropout_prob:
                        for k in range(segment_len):
                            if t_idx + k >= len(timesteps):
                                break
                            t = timesteps[t_idx + k]
                            x_tilde[batch_idx, feat_idx, t] = 0.0
                        t_idx += segment_len
                    else:
                        t_idx += 1
        
        return x_tilde
    
    def _spoofing_attack(self, x, target_features=None, target_timesteps=None):
        x_tilde = x.copy()
        n_samples, n_features, n_timesteps = x.shape
        
        if target_timesteps is None:
            target_timesteps = list(range(n_timesteps))
        
        if target_features is None:
            target_features = range(n_features)
        
        spoof_prob = min(0.4, self.epsilon / 5.0)
        segment_len = max(2, int(round(self.epsilon * 2)))
        
        timesteps = list(target_timesteps)
        for batch_idx in range(n_samples):
            for feat_idx in target_features:
                t_idx = 0
                while t_idx < len(timesteps):
                    if np.random.random() < spoof_prob:
                        for k in range(segment_len):
                            if t_idx + k >= len(timesteps):
                                break
                            t = timesteps[t_idx + k]
                            if t > 0:
                                x_tilde[batch_idx, feat_idx, t] = x_tilde[batch_idx, feat_idx, t-1]
                        t_idx += segment_len
                    else:
                        t_idx += 1
        
        return x_tilde
    
    def _clipping_attack(self, x, target_features=None, target_timesteps=None):
        x_tilde = x.copy()
        n_samples, n_features, n_timesteps = x.shape
        
        if target_timesteps is None:
            target_timesteps = list(range(n_timesteps))
        
        if target_features is None:
            target_features = range(n_features)
        
        clip_prob = min(0.3, self.epsilon / 5.0)
        
        for batch_idx in range(n_samples):
            for feat_idx in target_features:
                feature_min = np.min(x[:, feat_idx, :])
                feature_max = np.max(x[:, feat_idx, :])
                feature_range = feature_max - feature_min
                
                for t in target_timesteps:
                    if np.random.random() < clip_prob:
                        clip_type = np.random.choice(['cap', 'floor'])
                        
                        if clip_type == 'cap':
                            cap_value = feature_min + np.random.uniform(0.5, 1.5) * feature_range
                            x_tilde[batch_idx, feat_idx, t] = np.minimum(x_tilde[batch_idx, feat_idx, t], cap_value)
                        
                        elif clip_type == 'floor':
                            floor_value = feature_min + np.random.uniform(-1.5, -0.5) * feature_range
                            x_tilde[batch_idx, feat_idx, t] = np.maximum(x_tilde[batch_idx, feat_idx, t], floor_value)
        
        return x_tilde
    
    def _jitter_attack(self, x, target_features=None, target_timesteps=None):
        x_tilde = x.copy()
        n_samples, n_features, n_timesteps = x.shape
        
        if n_timesteps < 3:
            return x_tilde
        
        if target_timesteps is None:
            start = n_timesteps // 4
            end = 3 * n_timesteps // 4
            target_timesteps = list(range(start, end))
        
        if target_features is None:
            target_features = range(n_features)
        
        jitter_prob = min(0.5, self.epsilon / 3.0)
        
        for batch_idx in range(n_samples):
            for feat_idx in target_features:
                if np.random.random() < jitter_prob:
                    jitter_type = np.random.choice(['shift', 'shuffle', 'swap'])
                    
                    if jitter_type == 'shift':
                        shift_amount = np.random.choice([-2, -1, 1, 2])
                        shifted = np.roll(x_tilde[batch_idx, feat_idx, :], shift_amount)
                        for t in target_timesteps:
                            x_tilde[batch_idx, feat_idx, t] = shifted[t]
                    
                    elif jitter_type == 'shuffle':
                        if len(target_timesteps) >= 3:
                            window_size = min(3, len(target_timesteps))
                            start_idx = np.random.randint(0, len(target_timesteps) - window_size + 1)
                            window = target_timesteps[start_idx:start_idx + window_size]
                            values = x_tilde[batch_idx, feat_idx, window].copy()
                            np.random.shuffle(values)
                            x_tilde[batch_idx, feat_idx, window] = values
                    
                    elif jitter_type == 'swap':
                        if len(target_timesteps) >= 4:
                            mid = len(target_timesteps) // 2
                            seg1_start = target_timesteps[0]
                            seg1_end = target_timesteps[mid]
                            seg2_start = target_timesteps[mid]
                            seg2_end = target_timesteps[-1] + 1
                            
                            seg1 = x_tilde[batch_idx, feat_idx, seg1_start:seg1_end].copy()
                            seg2 = x_tilde[batch_idx, feat_idx, seg2_start:seg2_end].copy()
                            
                            if len(seg1) == len(seg2):
                                x_tilde[batch_idx, feat_idx, seg1_start:seg1_end] = seg2
                                x_tilde[batch_idx, feat_idx, seg2_start:seg2_end] = seg1
        
        return x_tilde


class STLGuidedRepairer:
    def __init__(self, feature_names, learned_bounds=None, margin=0.0):
        self.feature_names = feature_names
        self.learned_bounds = learned_bounds or {}
        self.corrector = SimpleLogicCorrector(feature_names=feature_names, margin=margin)
        self.feature_idx = {name: idx for idx, name in enumerate(feature_names)}
    
    def set_learned_bounds(self, learned_bounds):
        self.learned_bounds = learned_bounds
    
    def repair(self, x_tilde, method='comprehensive', verbose=False):
        if method == 'bounds_only':
            return self._repair_with_bounds(x_tilde)
        elif method == 'logic_only':
            return self._repair_with_logic(x_tilde)
        elif method == 'comprehensive':
            return self._repair_comprehensive(x_tilde, verbose=verbose)
        else:
            logger.error(f"Unknown repair method: {method}")
            return x_tilde.copy()
    
    def _repair_comprehensive(self, x_tilde, verbose=False):
        x_repaired = x_tilde.copy()
        
        if verbose:
            print("\nApplying comprehensive STL-guided repair:")
            print("-" * 80)
            
        if self.learned_bounds:
            if verbose:
                print("Applying learned bounds for all features...")
            bounds_dict = {}
            for feature_name, bounds in self.learned_bounds.items():
                if isinstance(bounds, dict):
                    lower = bounds.get('lower')
                    upper = bounds.get('upper')
                    bounds_dict[feature_name] = (lower, upper)
                    if verbose:
                        print(f"  {feature_name}: [{lower:.4f}, {upper:.4f}]")
                elif isinstance(bounds, tuple):
                    bounds_dict[feature_name] = bounds
                    if verbose:
                        print(f"  {feature_name}: [{bounds[0]:.4f}, {bounds[1]:.4f}]")
            
            x_repaired = self.corrector.enforce_multi_feature_bounds(x_repaired, bounds_dict)
            if verbose:
                change = np.mean(np.abs(x_repaired - x_tilde))
                print(f"  Change from bounds enforcement: {change:.6f}")
        
        basic_logics = []
        if 'total_flow' in self.feature_idx:
            basic_logics.append('traffic_flow_stable')
        if 'avg_speed' in self.feature_idx:
            basic_logics.append('speed_positive')
        if 'aq_NO2' in self.feature_idx:
            basic_logics.append('aq_no2_positive')
        if 'aq_O3' in self.feature_idx:
            basic_logics.append('aq_o3_positive')
        
        if basic_logics:
            if verbose:
                for logic_name in basic_logics:
                    print(f"  Applying {logic_name}...")
            x_before = x_repaired.copy()
            x_repaired = self.corrector.correct_multiple(x_repaired, basic_logics)
            if verbose:
                change = np.mean(np.abs(x_repaired - x_before))
                print(f"  Change from basic constraints: {change:.6f}")
        
        correlation_logics = []
        if 'total_flow' in self.feature_idx and 'avg_speed' in self.feature_idx:
            correlation_logics.append('flow_speed_correlation')
        if 'aq_NO2' in self.feature_idx and 'aq_NOx' in self.feature_idx:
            correlation_logics.append('aq_correlation')
        
        if correlation_logics:
            if verbose:
                for logic_name in correlation_logics:
                    print(f"  Applying {logic_name}...")
            x_before = x_repaired.copy()
            x_repaired = self.corrector.correct_multiple(x_repaired, correlation_logics)
            if verbose:
                change = np.mean(np.abs(x_repaired - x_before))
                print(f"  Change from correlation constraints: {change:.6f}")
        
        if verbose:
            total_change = np.mean(np.abs(x_repaired - x_tilde))
            print(f"\nTotal repair change: {total_change:.6f}")
            print("-" * 80)
        
        return x_repaired
    
    def _repair_with_bounds(self, x_tilde):
        if not self.learned_bounds:
            logger.warning("No learned bounds available for repair")
            return x_tilde.copy()
        
        bounds_dict = {}
        for feature_name, bounds in self.learned_bounds.items():
            if isinstance(bounds, dict):
                bounds_dict[feature_name] = (bounds.get('lower'), bounds.get('upper'))
            elif isinstance(bounds, tuple):
                bounds_dict[feature_name] = bounds
        
        return self.corrector.enforce_multi_feature_bounds(x_tilde, bounds_dict)
    
    def _repair_with_logic(self, x_tilde):
        all_logic_names = []
        
        if 'total_flow' in self.feature_idx:
            all_logic_names.append('traffic_flow_stable')
        if 'avg_speed' in self.feature_idx:
            all_logic_names.append('speed_positive')
        if 'aq_NO2' in self.feature_idx:
            all_logic_names.append('aq_no2_positive')
        if 'aq_O3' in self.feature_idx:
            all_logic_names.append('aq_o3_positive')

        if 'total_flow' in self.feature_idx and 'avg_speed' in self.feature_idx:
            all_logic_names.append('flow_speed_correlation')
        if 'aq_NO2' in self.feature_idx and 'aq_NOx' in self.feature_idx:
            all_logic_names.append('aq_correlation')
        
        if all_logic_names:
            x_repaired = self.corrector.correct_multiple(x_tilde, all_logic_names)
            return x_repaired
        else:
            return x_tilde.copy()


class STLAttackRepairEvaluator:
    def __init__(self, feature_names, stl_formulas=None):
        self.feature_names = feature_names
        self.stl_formulas = stl_formulas or {}
    
    def evaluate(self, x_clean, x_attacked, x_repaired):
        metrics = {}
        
        attack_l2 = np.linalg.norm(x_attacked - x_clean) / np.linalg.norm(x_clean)
        attack_linf = np.max(np.abs(x_attacked - x_clean))
        attack_mean = np.mean(np.abs(x_attacked - x_clean))
        
        metrics['attack'] = {
            'l2_relative': attack_l2,
            'linf': attack_linf,
            'mean_absolute': attack_mean
        }
        
        repair_l2_from_clean = np.linalg.norm(x_repaired - x_clean) / np.linalg.norm(x_clean)
        repair_l2_from_attacked = np.linalg.norm(x_repaired - x_attacked) / np.linalg.norm(x_attacked)
        repair_linf = np.max(np.abs(x_repaired - x_clean))
        repair_mean = np.mean(np.abs(x_repaired - x_clean))
        correction_magnitude = np.mean(np.abs(x_repaired - x_attacked))
        
        metrics['repair'] = {
            'l2_from_clean': repair_l2_from_clean,
            'l2_from_attacked': repair_l2_from_attacked,
            'linf': repair_linf,
            'mean_absolute': repair_mean,
            'correction_magnitude': correction_magnitude
        }
        
        feature_metrics = {}
        for idx, feature_name in enumerate(self.feature_names):
            if idx >= x_clean.shape[1]:
                continue
            
            feat_clean = x_clean[:, idx, :]
            feat_attacked = x_attacked[:, idx, :]
            feat_repaired = x_repaired[:, idx, :]
            
            feature_metrics[feature_name] = {
                'attack_mean': np.mean(np.abs(feat_attacked - feat_clean)),
                'repair_mean': np.mean(np.abs(feat_repaired - feat_clean)),
                'correction': np.mean(np.abs(feat_repaired - feat_attacked))
            }
        
        metrics['features'] = feature_metrics
        
        return metrics
    
    def print_metrics(self, metrics):
        print("\n" + "="*80)
        print("ATTACK-REPAIR EVALUATION METRICS")
        print("="*80)
        
        print("\nATTACK METRICS:")
        print("-" * 80)
        print(f"{'Metric':<30} {'Value':>15}")
        print("-" * 80)
        for key, value in metrics['attack'].items():
            print(f"{key:<30} {value:>15.6f}")
        
        print("\nREPAIR METRICS:")
        print("-" * 80)
        print(f"{'Metric':<30} {'Value':>15}")
        print("-" * 80)
        for key, value in metrics['repair'].items():
            print(f"{key:<30} {value:>15.6f}")
        
        print("\nFEATURE-WISE METRICS:")
        print("-" * 80)
        print(f"{'Feature':<15} {'Attack':>15} {'Repair':>15} {'Correction':>15}")
        print("-" * 80)
        for feature, fmetrics in metrics['features'].items():
            print(f"{feature:<15} "
                  f"{fmetrics['attack_mean']:>15.6f} "
                  f"{fmetrics['repair_mean']:>15.6f} "
                  f"{fmetrics['correction']:>15.6f}")
        
        print("="*80 + "\n")
