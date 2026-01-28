import numpy as np
from .stl_formulas import Formula, Atomic, STLNot, STLAnd, STLOr, Always, Eventually, STLUntil

def stl_robustness(formula: Formula, traj: np.ndarray, t: int = 0) -> float:
    T = traj.shape[0]
    
    if isinstance(formula, Atomic):
        if t >= T:
            return -np.inf
        val = np.dot(formula.w, traj[t])
        if formula.relop in [">=", ">"]:
            return val - formula.c
        else:
            return formula.c - val
    
    elif isinstance(formula, STLNot):
        return -stl_robustness(formula.child, traj, t)
    
    elif isinstance(formula, STLAnd):
        return min(stl_robustness(formula.left, traj, t),
                   stl_robustness(formula.right, traj, t))
    
    elif isinstance(formula, STLOr):
        return max(stl_robustness(formula.left, traj, t),
                   stl_robustness(formula.right, traj, t))
    
    elif isinstance(formula, Always):
        t_start = max(0, t + formula.t_start)
        t_end = min(T - 1, t + formula.t_end)
        if t_start > t_end:
            return np.inf
        return min(stl_robustness(formula.child, traj, ti) 
                   for ti in range(t_start, t_end + 1))
    
    elif isinstance(formula, Eventually):
        t_start = max(0, t + formula.t_start)
        t_end = min(T - 1, t + formula.t_end)
        if t_start > t_end:
            return -np.inf
        return max(stl_robustness(formula.child, traj, ti) 
                   for ti in range(t_start, t_end + 1))
    
    elif isinstance(formula, STLUntil):
        t_start = max(0, t + formula.t_start)
        t_end = min(T - 1, t + formula.t_end)
        if t_start > t_end:
            return -np.inf
        best = -np.inf
        for t2 in range(t_start, t_end + 1):
            rho_right = stl_robustness(formula.right, traj, t2)
            if t2 == t:
                rho_left_min = np.inf
            else:
                rho_left_min = min(stl_robustness(formula.left, traj, ti) 
                                   for ti in range(t, t2))
            best = max(best, min(rho_right, rho_left_min))
        return best
    
    else:
        raise TypeError(f"Unknown formula type: {type(formula)}")


def lipschitz_stl(formula: Formula) -> float:
    if isinstance(formula, Atomic):
        return float(np.linalg.norm(formula.w, ord=2))
    elif isinstance(formula, STLNot):
        return lipschitz_stl(formula.child)
    elif isinstance(formula, (STLAnd, STLOr)):
        return max(lipschitz_stl(formula.left), lipschitz_stl(formula.right))
    elif isinstance(formula, (Always, Eventually)):
        return lipschitz_stl(formula.child)
    elif isinstance(formula, STLUntil):
        return max(lipschitz_stl(formula.left), lipschitz_stl(formula.right))
    else:
        raise TypeError(f"Unknown formula type: {type(formula)}")


def empirical_lipschitz_stl(
    formula: Formula,
    L_rho_theory: float,
    n_samples: int = 1000,
    eps: float = 1e-3,
    T: int = 16,
    d: int = 4,
    seed: int = 42
):
    rng = np.random.default_rng(seed)
    max_ratio = 0.0
    violations = 0

    for _ in range(n_samples):
        Y = rng.uniform(low=-1.0, high=1.0, size=(T, d))
        delta = rng.normal(size=(T, d))
        delta = delta / np.linalg.norm(delta) * eps
        Yp = Y + delta

        rho_Y = stl_robustness(formula, Y)
        rho_Yp = stl_robustness(formula, Yp)

        if np.isinf(rho_Y) or np.isinf(rho_Yp):
            continue

        num = abs(rho_Yp - rho_Y)
        denom = np.linalg.norm(Yp - Y)
        ratio = num / denom if denom > 0 else 0.0

        max_ratio = max(max_ratio, ratio)
        if ratio > L_rho_theory * 1.01:
            violations += 1

    return max_ratio, violations, n_samples

