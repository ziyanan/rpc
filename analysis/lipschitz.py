import torch
import torch.nn as nn
from net.seq import RMSNorm

def spectral_norm_matrix(matrix, iters=20):
    w = matrix.float()
    u = torch.randn(w.shape[0], 1, device=w.device)
    for _ in range(iters):
        v = torch.nn.functional.normalize(torch.matmul(w.t(), u), dim=0)
        u = torch.nn.functional.normalize(torch.matmul(w, v), dim=0)
    return torch.matmul(torch.matmul(u.t(), w), v).item()

def spectral_norm_conv1d(weight, iters=20):
    w = weight.reshape(weight.shape[0], -1).float()
    return spectral_norm_matrix(w, iters)

def analyze_lipschitz(model):
    linear_norms = []
    conv_norms = []
    has_layernorm = False
    has_rmsnorm = False
    has_attention = False
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            sn = spectral_norm_matrix(module.weight.data)
            linear_norms.append((name, sn))
        elif isinstance(module, nn.Conv1d):
            sn = spectral_norm_conv1d(module.weight.data)
            conv_norms.append((name, sn))
        elif isinstance(module, nn.LayerNorm):
            has_layernorm = True
        elif isinstance(module, RMSNorm):
            has_rmsnorm = True
        elif isinstance(module, nn.MultiheadAttention):
            has_attention = True
    
    return linear_norms, conv_norms, has_layernorm, has_rmsnorm, has_attention

def empirical_lipschitz(model, n_samples=100, eps=1e-3, logic_name=None):
    model.eval()
    device = next(model.parameters()).device
    max_ratio = 0.0
    with torch.no_grad():
        for _ in range(n_samples):
            x1 = torch.randn(1, 7, 64, device=device)
            delta = torch.randn_like(x1)
            delta = delta / delta.norm() * eps
            x2 = x1 + delta
            
            y1 = model(x1, logic_name=logic_name, add_noise=False)
            y2 = model(x2, logic_name=logic_name, add_noise=False)
            
            ratio = (y1 - y2).norm().item() / delta.norm().item()
            max_ratio = max(max_ratio, ratio)
    return max_ratio

