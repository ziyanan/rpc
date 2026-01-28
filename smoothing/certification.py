import numpy as np
from scipy import stats
from scipy.special import ndtri


def confidence_lower_bound(count_A, n_samples, alpha=0.001):
    if count_A == 0:
        return 0.0
    
    if count_A == n_samples:
        return (alpha) ** (1/n_samples)
    
    p_lower = stats.beta.ppf(alpha, count_A, n_samples - count_A + 1)
    
    return p_lower


def compute_certified_radius(p_A, p_B, sigma, n_samples, alpha=0.001, return_diagnostics=False):
    if p_A > p_B:
        prediction = 1
        count_A = int(round(p_A * n_samples))
        p_hat = p_A
    else:
        prediction = 0
        count_A = int(round(p_B * n_samples))
        p_hat = p_B
    
    min_samples_required = max(10, int(np.ceil(-np.log(alpha))))
    if n_samples < min_samples_required:
        if return_diagnostics:
            return 0.0, prediction, {
                'p_hat': p_hat,
                'p_lower': 0.0,
                'count': count_A,
                'n_samples': n_samples,
                'failed_reason': f'insufficient_samples (need >={min_samples_required})',
                'alpha': alpha
            }
        return 0.0, prediction
    
    if p_hat <= 0.5:
        if return_diagnostics:
            return 0.0, prediction, {
                'p_hat': p_hat,
                'p_lower': 0.0,
                'count': count_A,
                'n_samples': n_samples,
                'failed_reason': 'no_majority',
                'alpha': alpha
            }
        return 0.0, prediction
    
    p_lower = confidence_lower_bound(count_A, n_samples, alpha)
    
    if p_lower <= 0.5:
        if return_diagnostics:
            return 0.0, prediction, {
                'p_hat': p_hat,
                'p_lower': p_lower,
                'count': count_A,
                'n_samples': n_samples,
                'failed_reason': 'lower_bound_below_threshold',
                'alpha': alpha
            }
        return 0.0, prediction
    
    radius = (sigma / 2.0) * (ndtri(p_lower) - ndtri(1.0 - p_lower))
    
    radius = max(0.0, radius)
    
    if return_diagnostics:
        return radius, prediction, {
            'p_hat': p_hat,
            'p_lower': p_lower,
            'count': count_A,
            'n_samples': n_samples,
            'alpha': alpha,
            'certified': radius > 0
        }
    
    return radius, prediction
