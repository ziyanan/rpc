import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

class SimpleLogicCorrector:
    def __init__(self, feature_names=None, margin=0.0):
        if feature_names is None:
            feature_names = ['total_flow', 'avg_occupancy', 'avg_speed', 
                           'aq_NO2', 'aq_NOx', 'aq_O3', 'aq_PM25']
        self.feature_names = feature_names
        self.feature_idx = {name: idx for idx, name in enumerate(feature_names)}
        self.margin = max(0.0, margin)
        self.learned_params = None
        
        self.corrections = {
            'traffic_flow_stable': self._clip_positive_flow,
            'speed_positive': self._clip_positive_speed,
            'aq_no2_positive': self._clip_positive_aq_no2,
            'aq_o3_positive': self._clip_positive_aq_o3,
            'flow_speed_correlation': self._enforce_flow_speed_correlation,
            'aq_correlation': self._enforce_aq_correlation,
            'eventually_high_flow': self._ensure_high_flow_peak,
            'eventually_low_speed': self._ensure_low_speed_dip,
            'eventually_high_aq': self._ensure_high_aq_peak,
            'correlation_flow_speed': self._maintain_flow_speed_diff,
            'correlation_aq': self._maintain_aq_diff,
            'until_high_flow': self._until_high_flow,
            'until_congestion': self._until_congestion,
            'until_no2_spike': self._until_no2_spike,
            'until_coupling_break': self._until_coupling_break,
            'until_nox_inversion': self._until_nox_inversion,
            'until_speed_outage': self._until_speed_outage,
            'until_maint_window': self._until_maint_window,
            'until_o3_surge': self._until_o3_surge,
        }
        self.logic_requirements = {
            'traffic_flow_stable': ['total_flow'],
            'speed_positive': ['avg_speed'],
            'aq_no2_positive': ['aq_NO2'],
            'aq_o3_positive': ['aq_O3'],
            'flow_speed_correlation': ['total_flow', 'avg_speed'],
            'aq_correlation': ['aq_NO2', 'aq_NOx'],
            'eventually_high_flow': ['total_flow'],
            'eventually_low_speed': ['avg_speed'],
            'eventually_high_aq': ['aq_NO2'],
            'correlation_flow_speed': ['total_flow', 'avg_speed'],
            'correlation_aq': ['aq_NO2', 'aq_NOx'],
            'until_high_flow': ['total_flow'],
            'until_congestion': ['avg_occupancy'],
            'until_no2_spike': ['aq_NO2'],
            'until_coupling_break': ['aq_NO2', 'aq_NOx'],
            'until_nox_inversion': ['aq_NOx', 'aq_NO2'],
            'until_speed_outage': ['avg_speed'],
            'until_maint_window': ['total_flow'],
            'until_o3_surge': ['aq_O3', 'aq_NO2', 'aq_NOx'],
        }
    
        self.learned_params = None
    
    def learn_thresholds_from_data(self, data, verbose=True):
        if verbose:
            logger.info("="*80)
            logger.info("LEARNING LOGIC THRESHOLDS FROM DATA")
            logger.info("="*80)
            logger.info(f"Data shape: {data.shape}")
            logger.info(f"Computing percentile-based thresholds...\n")
        
        learned = {}
        
        for idx, feature_name in enumerate(self.feature_names):
            if idx >= data.shape[1]:
                continue
            
            feature_data = data[:, idx, :].flatten()
            
            learned[feature_name] = {
                'min': float(np.min(feature_data)),
                'max': float(np.max(feature_data)),
                'p5': float(np.percentile(feature_data, 5)),
                'p10': float(np.percentile(feature_data, 10)),
                'p25': float(np.percentile(feature_data, 25)),
                'p50': float(np.percentile(feature_data, 50)),  # median
                'p75': float(np.percentile(feature_data, 75)),
                'p90': float(np.percentile(feature_data, 90)),
                'p95': float(np.percentile(feature_data, 95)),
                'mean': float(np.mean(feature_data)),
                'std': float(np.std(feature_data))
            }
            
            if verbose:
                logger.info(f"{feature_name:>15s}: "
                          f"[{learned[feature_name]['p5']:.3f}, {learned[feature_name]['p95']:.3f}] "
                          f"(5th-95th percentile)")
        
        if verbose:
            logger.info("\nTemporal derivative thresholds (95th percentile):")
        
        for idx, feature_name in enumerate(self.feature_names):
            if idx >= data.shape[1]:
                continue
            
            feature_data = data[:, idx, :]
            diffs = np.abs(np.diff(feature_data, axis=1))  # (N, timesteps-1)
            delta_threshold = float(np.percentile(diffs, 95))
            
            learned[feature_name]['temporal_diff_p95'] = delta_threshold
            
            if verbose:
                logger.info(f"{feature_name:>15s}: delta <= {delta_threshold:.3f}")
        
        if 'total_flow' in self.feature_idx and 'avg_speed' in self.feature_idx:
            flow_idx = self.feature_idx['total_flow']
            speed_idx = self.feature_idx['avg_speed']
            
            flow_data = data[:, flow_idx, :].flatten()
            speed_data = data[:, speed_idx, :].flatten()
            
            learned['flow_speed_correlation'] = {
                'high_flow_threshold': float(np.percentile(flow_data, 70)),
                'low_speed_threshold': float(np.percentile(speed_data, 30))
            }
        
        if 'aq_NO2' in self.feature_idx and 'aq_NOx' in self.feature_idx:
            no2_idx = self.feature_idx['aq_NO2']
            nox_idx = self.feature_idx['aq_NOx']
            
            no2_data = data[:, no2_idx, :].flatten()
            nox_data = data[:, nox_idx, :].flatten()
            
            learned['aq_correlation'] = {
                'high_no2_threshold': float(np.percentile(no2_data, 70)),
                'low_nox_threshold': float(np.percentile(nox_data, 30))
            }
        
        for feature_name in ['total_flow', 'aq_NO2', 'aq_NOx', 'aq_O3']:
            if feature_name in learned:
                learned[f'eventually_high_{feature_name}'] = {
                    'threshold': learned[feature_name]['p90']
                }
        
        if 'avg_speed' in learned:
            learned['eventually_low_speed'] = {
                'threshold': learned['avg_speed']['p10']
            }
        
        learned['until_high_flow'] = {
            'phi_threshold': learned.get('total_flow', {}).get('p10', 0),
            'psi_threshold': learned.get('total_flow', {}).get('p90', 1)
        } if 'total_flow' in learned else {}
        
        learned['until_congestion'] = {
            'phi_threshold': learned.get('avg_occupancy', {}).get('p75', 0.5),
            'psi_threshold': learned.get('avg_occupancy', {}).get('p75', 0.5)
        } if 'avg_occupancy' in learned else {}
        
        learned['until_no2_spike'] = {
            'phi_threshold': learned.get('aq_NO2', {}).get('p75', 0.5),
            'psi_threshold': learned.get('aq_NO2', {}).get('p75', 0.5)
        } if 'aq_NO2' in learned else {}
        
        learned['until_speed_outage'] = {
            'phi_threshold': learned.get('avg_speed', {}).get('p10', -1),
            'psi_threshold': learned.get('avg_speed', {}).get('p10', -1)
        } if 'avg_speed' in learned else {}
        
        if verbose:
            logger.info("\n" + "="*80)
            logger.info(f"Learned thresholds for {len(self.feature_names)} features")
            logger.info("="*80 + "\n")
        
        self.learned_params = learned
        return learned
    
    def _get_learned_params_for_logic(self, logic_name):
        if self.learned_params is None:
            return {}
        
        if logic_name in self.learned_params:
            return self.learned_params[logic_name]
        
        if logic_name == 'flow_speed_correlation':
            return self.learned_params.get('flow_speed_correlation', {})
        elif logic_name == 'aq_correlation':
            return self.learned_params.get('aq_correlation', {})
        elif logic_name.startswith('eventually_'):
            return self.learned_params.get(logic_name, {})
        elif logic_name.startswith('until_'):
            return self.learned_params.get(logic_name, {})
        
        return {}
    
    def _has_required_features(self, logic_name):
        required = self.logic_requirements.get(logic_name, [])
        return all(name in self.feature_idx for name in required)

    def correct(self, x, logic_name, params=None):
        if not self._has_required_features(logic_name):
            logger.warning(f"Skipping logic '{logic_name}' due to missing features")
            return x.copy()
        if logic_name not in self.corrections:
            logger.warning(f"Unknown logic: {logic_name}, returning original data")
            return x.copy()
        
        if params is None and self.learned_params is not None:
            learned_params = self._get_learned_params_for_logic(logic_name)
            merged_params = self._get_merged_params(logic_name, learned_params)
        else:
            merged_params = self._get_merged_params(logic_name, params)
        
        return self.corrections[logic_name](x, **merged_params)
    
    def correct_multiple(self, x, logic_names, params_list=None):
        x_corrected = x.copy()
        if params_list is None:
            params_list = [None] * len(logic_names)
        
        for logic_name, params in zip(logic_names, params_list):
            x_corrected = self.correct(x_corrected, logic_name, params=params)
        return x_corrected
    
    def _get_merged_params(self, logic_name, params):
        merged = {}
        
        if self.learned_params is not None:
            learned = self._get_learned_params_for_logic(logic_name)
            if learned:
                merged = learned.copy() if isinstance(learned, dict) else {}

        if params is not None:
            merged.update(params)
        
        return merged
    
    def _clip_positive_flow(self, x, **kwargs):
        x_corrected = x.copy()
        flow_idx = self.feature_idx['total_flow']
        x_corrected[:, flow_idx, :] = np.maximum(x_corrected[:, flow_idx, :], self.margin)
        return x_corrected
    
    def _clip_positive_speed(self, x, **kwargs):
        x_corrected = x.copy()
        speed_idx = self.feature_idx['avg_speed']
        x_corrected[:, speed_idx, :] = np.maximum(x_corrected[:, speed_idx, :], self.margin)
        return x_corrected
    
    def _clip_positive_aq_no2(self, x, **kwargs):
        x_corrected = x.copy()
        no2_idx = self.feature_idx['aq_NO2']
        x_corrected[:, no2_idx, :] = np.maximum(x_corrected[:, no2_idx, :], -2 + self.margin)
        return x_corrected
    
    def _clip_positive_aq_o3(self, x, **kwargs):
        x_corrected = x.copy()
        o3_idx = self.feature_idx['aq_O3']
        x_corrected[:, o3_idx, :] = np.maximum(x_corrected[:, o3_idx, :], -2 + self.margin)
        return x_corrected
    
    def _enforce_flow_speed_correlation(self, x, high_flow_threshold=0.5, low_speed_threshold=0.2):
        x_corrected = x.copy()
        flow_idx = self.feature_idx['total_flow']
        speed_idx = self.feature_idx['avg_speed']
        
        high_flow_mask = x_corrected[:, flow_idx, :-1] > high_flow_threshold
        low_speed_mask = x_corrected[:, speed_idx, :-1] <= low_speed_threshold + self.margin
        violation_mask = high_flow_mask & low_speed_mask
        
        x_corrected[:, speed_idx, :-1][violation_mask] = low_speed_threshold + self.margin + 0.01
        
        return x_corrected
    
    def _enforce_aq_correlation(self, x, high_no2_threshold=0.5, low_nox_threshold=0.3):
        x_corrected = x.copy()
        no2_idx = self.feature_idx['aq_NO2']
        nox_idx = self.feature_idx['aq_NOx']
        
        high_no2_mask = x_corrected[:, no2_idx, :-1] > high_no2_threshold
        low_nox_mask = x_corrected[:, nox_idx, :-1] <= low_nox_threshold + self.margin
        violation_mask = high_no2_mask & low_nox_mask
        
        x_corrected[:, nox_idx, :-1][violation_mask] = low_nox_threshold + self.margin + 0.01
        
        return x_corrected
    
    def _ensure_high_flow_peak(self, x, threshold=1.0):
        x_corrected = x.copy()
        flow_idx = self.feature_idx['total_flow']
        
        for i in range(x_corrected.shape[0]):
            max_flow = np.max(x_corrected[i, flow_idx, :])
            adj_threshold = threshold + self.margin
            if max_flow < adj_threshold:
                peak_idx = np.argmax(x_corrected[i, flow_idx, :])
                x_corrected[i, flow_idx, peak_idx] = adj_threshold + 0.1
        
        return x_corrected
    
    def _ensure_low_speed_dip(self, x, threshold=-0.5):
        x_corrected = x.copy()
        speed_idx = self.feature_idx['avg_speed']
        
        for i in range(x_corrected.shape[0]):
            min_speed = np.min(x_corrected[i, speed_idx, :])
            adj_threshold = threshold - self.margin
            if min_speed >= adj_threshold:
                dip_idx = np.argmin(x_corrected[i, speed_idx, :])
                x_corrected[i, speed_idx, dip_idx] = adj_threshold - 0.1
        
        return x_corrected
    
    def _ensure_high_aq_peak(self, x, threshold=1.0):
        x_corrected = x.copy()
        no2_idx = self.feature_idx['aq_NO2']
        
        for i in range(x_corrected.shape[0]):
            max_no2 = np.max(x_corrected[i, no2_idx, :])
            adj_threshold = threshold + self.margin
            if max_no2 < adj_threshold:
                peak_idx = np.argmax(x_corrected[i, no2_idx, :])
                x_corrected[i, no2_idx, peak_idx] = adj_threshold + 0.1
        
        return x_corrected
    
    def _maintain_flow_speed_diff(self, x, min_diff=-2.0):
        x_corrected = x.copy()
        flow_idx = self.feature_idx['total_flow']
        speed_idx = self.feature_idx['avg_speed']
        
        diff = x_corrected[:, flow_idx, :] - x_corrected[:, speed_idx, :]
        violation_mask = diff < min_diff - self.margin
        
        x_corrected[:, speed_idx, :][violation_mask] = x_corrected[:, flow_idx, :][violation_mask] - min_diff + self.margin + 0.1
        
        return x_corrected
    
    def _maintain_aq_diff(self, x, min_diff=-1.0):
        x_corrected = x.copy()
        no2_idx = self.feature_idx['aq_NO2']
        nox_idx = self.feature_idx['aq_NOx']
        
        diff = x_corrected[:, no2_idx, :] - x_corrected[:, nox_idx, :]
        violation_mask = diff < min_diff - self.margin
        
        x_corrected[:, nox_idx, :][violation_mask] = x_corrected[:, no2_idx, :][violation_mask] - min_diff + self.margin + 0.1
        
        return x_corrected
    
    def _until_high_flow(self, x, phi_threshold=-0.583, psi_threshold=2.030):
        x_corrected = x.copy()
        flow_idx = self.feature_idx['total_flow']
        
        for i in range(x_corrected.shape[0]):
            adj_phi = phi_threshold + self.margin
            
            spike_times = np.where(x_corrected[i, flow_idx, :] > psi_threshold)[0]
            
            if len(spike_times) == 0:
                peak_time = np.argmax(x_corrected[i, flow_idx, :])
                x_corrected[i, flow_idx, peak_time] = psi_threshold + 0.1
                spike_times = [peak_time]
            
            until_time = spike_times[0]
            x_corrected[i, flow_idx, :until_time] = np.maximum(
                x_corrected[i, flow_idx, :until_time], adj_phi
            )
        
        return x_corrected
    
    def _until_congestion(self, x, phi_threshold=0.848, psi_threshold=0.848):
        x_corrected = x.copy()
        speed_idx = self.feature_idx['avg_speed']
        
        for i in range(x_corrected.shape[0]):
            adj_phi = phi_threshold + self.margin
            
            congestion_times = np.where(x_corrected[i, speed_idx, :] < psi_threshold)[0]
            
            if len(congestion_times) == 0:
                congestion_time = np.argmin(x_corrected[i, speed_idx, :])
                x_corrected[i, speed_idx, congestion_time] = psi_threshold - 0.1
                congestion_times = [congestion_time]
            
            until_time = congestion_times[0]
            x_corrected[i, speed_idx, :until_time] = np.maximum(
                x_corrected[i, speed_idx, :until_time], adj_phi
            )
        
        return x_corrected
    
    def _until_no2_spike(self, x, phi_threshold=0.688, psi_threshold=0.688):
        x_corrected = x.copy()
        no2_idx = self.feature_idx['aq_NO2']
        
        for i in range(x_corrected.shape[0]):
            adj_phi = phi_threshold - self.margin
            
            spike_times = np.where(x_corrected[i, no2_idx, :] > psi_threshold)[0]
            
            if len(spike_times) == 0:
                spike_time = np.argmax(x_corrected[i, no2_idx, :])
                x_corrected[i, no2_idx, spike_time] = psi_threshold + 0.1
                spike_times = [spike_time]
            
            until_time = spike_times[0]
            x_corrected[i, no2_idx, :until_time] = np.minimum(
                x_corrected[i, no2_idx, :until_time], adj_phi
            )
        
        return x_corrected
    
    def _until_coupling_break(self, x, phi_threshold=1.268, psi_threshold=1.268):
        x_corrected = x.copy()
        flow_idx = self.feature_idx['total_flow']
        speed_idx = self.feature_idx['avg_speed']
        
        for i in range(x_corrected.shape[0]):
            adj_phi = phi_threshold + self.margin
            
            diff = x_corrected[i, flow_idx, :] - x_corrected[i, speed_idx, :]
            break_times = np.where(diff < psi_threshold)[0]
            
            if len(break_times) == 0:
                break_time = np.argmin(diff)
                adjustment = psi_threshold - diff[break_time] - 0.1
                x_corrected[i, speed_idx, break_time] += adjustment
                break_times = [break_time]
            
            until_time = break_times[0]
            for t in range(until_time):
                current_diff = x_corrected[i, flow_idx, t] - x_corrected[i, speed_idx, t]
                if current_diff < adj_phi:
                    x_corrected[i, speed_idx, t] = x_corrected[i, flow_idx, t] - adj_phi - 0.01
        
        return x_corrected
    
    def _until_nox_inversion(self, x, phi_threshold=-0.187, psi_threshold=-0.187):
        x_corrected = x.copy()
        no2_idx = self.feature_idx['aq_NO2']
        nox_idx = self.feature_idx['aq_NOx']
        
        for i in range(x_corrected.shape[0]):
            adj_phi = phi_threshold + self.margin
            
            diff = x_corrected[i, no2_idx, :] - x_corrected[i, nox_idx, :]
            inversion_times = np.where(diff < psi_threshold)[0]
            
            if len(inversion_times) == 0:
                inversion_time = np.argmin(diff)
                adjustment = psi_threshold - diff[inversion_time] - 0.1
                x_corrected[i, nox_idx, inversion_time] -= adjustment
                inversion_times = [inversion_time]
            
            until_time = inversion_times[0]
            for t in range(until_time):
                current_diff = x_corrected[i, no2_idx, t] - x_corrected[i, nox_idx, t]
                if current_diff < adj_phi:
                    x_corrected[i, nox_idx, t] = x_corrected[i, no2_idx, t] - adj_phi - 0.01
        
        return x_corrected
    
    def _until_speed_outage(self, x, phi_threshold=-1.798, psi_threshold=-1.798):
        x_corrected = x.copy()
        speed_idx = self.feature_idx['avg_speed']
        
        for i in range(x_corrected.shape[0]):
            adj_phi = phi_threshold + self.margin
            
            outage_times = np.where(x_corrected[i, speed_idx, :] < psi_threshold)[0]
            
            if len(outage_times) == 0:
                outage_time = np.argmin(x_corrected[i, speed_idx, :])
                x_corrected[i, speed_idx, outage_time] = psi_threshold - 0.1
                outage_times = [outage_time]
            
            until_time = outage_times[0]
            x_corrected[i, speed_idx, :until_time] = np.maximum(
                x_corrected[i, speed_idx, :until_time], adj_phi
            )
        
        return x_corrected
    
    def _until_maint_window(self, x, phi_threshold=-0.583, psi_threshold=-0.583, maint_start=5, maint_end=12):
        x_corrected = x.copy()
        flow_idx = self.feature_idx['total_flow']
        
        for i in range(x_corrected.shape[0]):
            adj_phi = phi_threshold + self.margin
            
            outage_times = np.where(x_corrected[i, flow_idx, maint_start:maint_end+1] < psi_threshold)[0]
            
            if len(outage_times) == 0:
                relative_min = np.argmin(x_corrected[i, flow_idx, maint_start:maint_end+1])
                absolute_time = maint_start + relative_min
                x_corrected[i, flow_idx, absolute_time] = psi_threshold - 0.1
                outage_times = [relative_min]
            
            until_time = maint_start + outage_times[0]
            x_corrected[i, flow_idx, maint_start:until_time] = np.maximum(
                x_corrected[i, flow_idx, maint_start:until_time], adj_phi
            )
        
        return x_corrected
    
    def _until_o3_surge(self, x, coupling_threshold=0.187, psi_threshold=-0.869):
        x_corrected = x.copy()
        no2_idx = self.feature_idx['aq_NO2']
        nox_idx = self.feature_idx['aq_NOx']
        o3_idx = self.feature_idx['aq_O3']
        
        for i in range(x_corrected.shape[0]):
            adj_coupling = coupling_threshold + self.margin
            
            surge_times = np.where(x_corrected[i, o3_idx, :] > psi_threshold)[0]
            
            if len(surge_times) == 0:
                surge_time = np.argmax(x_corrected[i, o3_idx, :])
                x_corrected[i, o3_idx, surge_time] = psi_threshold + 0.1
                surge_times = [surge_time]
            
            until_time = surge_times[0]
            for t in range(until_time):
                current_diff = x_corrected[i, no2_idx, t] - (x_corrected[i, nox_idx, t] + adj_coupling)
                if current_diff > 0:
                    x_corrected[i, no2_idx, t] = x_corrected[i, nox_idx, t] + adj_coupling
        
        return x_corrected
    
    def get_logic_names(self):
        return [name for name in self.corrections.keys() if self._has_required_features(name)]
    
    def get_default_params(self, logic_name):
        if self.learned_params is None:
            logger.warning(f"Thresholds not learned yet for {logic_name}! Returning empty params.")
            return {}
        learned = self._get_learned_params_for_logic(logic_name)
        return learned if learned else {}
    
    def test_correction(self, x, logic_name, params=None):
        x_corrected = self.correct(x, logic_name, params=params)
        diff = np.abs(x_corrected - x).mean()
        logger.info(f"Logic: {logic_name}, Mean change: {diff:.6f}")
        if params:
            logger.info(f"  Parameters: {params}")
        return x_corrected
    
    def enforce_bounds(self, x, feature_name, lower_bound=None, upper_bound=None):
        if feature_name not in self.feature_idx:
            logger.warning(f"Feature {feature_name} not found, returning original data")
            return x.copy()
        
        x_corrected = x.copy()
        feature_idx = self.feature_idx[feature_name]
        
        if lower_bound is not None:
            x_corrected[:, feature_idx, :] = np.maximum(x_corrected[:, feature_idx, :], lower_bound)
        
        if upper_bound is not None:
            x_corrected[:, feature_idx, :] = np.minimum(x_corrected[:, feature_idx, :], upper_bound)
        
        return x_corrected
    
    def enforce_multi_feature_bounds(self, x, bounds_dict):
        x_corrected = x.copy()
        for feature_name, (lower, upper) in bounds_dict.items():
            x_corrected = self.enforce_bounds(x_corrected, feature_name, lower, upper)
        return x_corrected