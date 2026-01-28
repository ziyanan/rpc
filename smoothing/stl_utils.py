import numpy as np
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.stl_robustness import stl_robustness
from analysis.stl_formulas import Always, Eventually, STLUntil, Atomic, STLAnd, STLOr, STLNot


def _max_atomic_w_length(formula):
    if isinstance(formula, Atomic):
        return len(formula.w) if getattr(formula, "w", None) is not None else 0
    elif isinstance(formula, (Always, Eventually, STLNot, STLUntil)):
        return _max_atomic_w_length(formula.child)
    elif isinstance(formula, (STLAnd, STLOr)):
        return max(_max_atomic_w_length(formula.left), _max_atomic_w_length(formula.right))
    else:
        return 0


def compute_stl_robustness(y_pred, formula, feature_names=None):
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    original_shape = y_pred.shape
    if y_pred.ndim == 2:
        if y_pred.shape[0] > y_pred.shape[1]:
            y_pred = y_pred.T
    num_features = y_pred.shape[1] if y_pred.ndim == 2 else None
    max_w_len = _max_atomic_w_length(formula)
    if num_features is not None and max_w_len == num_features * 3:
        deltas = np.diff(y_pred, axis=0)
        deltas = np.concatenate([deltas, deltas[-1:, :]], axis=0)
        abs_deltas = np.abs(deltas)
        y_pred = np.concatenate([y_pred, deltas, abs_deltas], axis=1)
    
    try:
        rho = stl_robustness(formula, y_pred, t=0)
        return float(rho)
    except Exception as e:
        print(f"Warning: STL robustness computation failed: {e}")
        print(f"  Original shape: {original_shape}, After transpose: {y_pred.shape}")
        print(f"  Formula weight shape: {formula.w.shape if hasattr(formula, 'w') else 'N/A'}")
        import traceback
        traceback.print_exc()
        return -np.inf


def batch_stl_robustness(y_preds, formula, feature_names=None):
    if isinstance(y_preds, torch.Tensor):
        y_preds = y_preds.detach().cpu().numpy()
    
    rho_scores = []
    for i in range(len(y_preds)):
        rho = compute_stl_robustness(y_preds[i], formula, feature_names)
        rho_scores.append(rho)
    
    return np.array(rho_scores)


SAT_LABEL = 1
VIOL_LABEL = 0


def binary_classification(rho, threshold=0.0):
    if np.isscalar(rho):
        return SAT_LABEL if rho >= threshold else VIOL_LABEL
    return np.where(rho >= threshold, SAT_LABEL, VIOL_LABEL)


def majority_vote(classifications):
    classifications = np.array(classifications, dtype=int)
    n = len(classifications)
    
    count_satisfied = np.sum(classifications)
    count_violated = n - count_satisfied
    
    p_A = count_satisfied / n
    p_B = count_violated / n
    
    prediction = SAT_LABEL if p_A > p_B else VIOL_LABEL
    
    return prediction, p_A, p_B
