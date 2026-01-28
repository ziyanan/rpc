#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import json
import argparse
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from net.net import LogicGuidedDiffusionForecast
from logics.stl_attack_repair import AttackGenerator, STLGuidedRepairer, TeLExLearner
from smoothing.stl_utils import compute_stl_robustness
from smoothing.test_smoothing import create_meaningful_stl_formulas


def load_model(model_path, device='cpu'):
    checkpoint = torch.load(model_path, map_location=device)
    
    in_channels = checkpoint.get('in_channels', 7)
    forecast_horizon = checkpoint.get('forecast_horizon', 6)
    
    model_dir = os.path.dirname(model_path)
    config_path = os.path.join(model_dir, 'config.json')
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        model_config = config['model']
        diffusion_hidden = model_config.get('diffusion_hidden', 32)
        diffusion_levels = model_config.get('diffusion_levels', 2)
        seq_d_model = model_config.get('seq_d_model', 64)
        seq_n_heads = model_config.get('seq_n_heads', 2)
        seq_n_layers = model_config.get('seq_n_layers', 2)
        num_diffusion_steps = model_config.get('num_diffusion_steps', 10)
        use_conditioned = model_config.get('use_conditioned_diffusion', True)
    else:
        diffusion_hidden = 32
        diffusion_levels = 2
        seq_d_model = 64
        seq_n_heads = 2
        seq_n_layers = 2
        num_diffusion_steps = 10
        use_conditioned = True
    
    model = LogicGuidedDiffusionForecast(
        in_channels=in_channels,
        diffusion_hidden=diffusion_hidden,
        diffusion_levels=diffusion_levels,
        seq_d_model=seq_d_model,
        seq_n_heads=seq_n_heads,
        seq_n_layers=seq_n_layers,
        forecast_horizon=forecast_horizon,
        num_diffusion_steps=num_diffusion_steps,
        use_conditioned_diffusion=use_conditioned
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, use_conditioned


def compute_sample_errors(y_pred, y_true, metric='mae'):
    if metric == 'mae':
        errors = np.mean(np.abs(y_pred - y_true), axis=(1, 2))  # (N,)
    elif metric == 'rmse':
        errors = np.sqrt(np.mean((y_pred - y_true) ** 2, axis=(1, 2)))  # (N,)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    return errors


def compute_sample_robustness(y_pred, stl_formulas, feature_names):
    n_samples = y_pred.shape[0]
    robustness_scores = []
    
    for i in range(n_samples):
        y_sample = y_pred[i].T
        
        sample_robustness = []
        for formula_name, formula in stl_formulas.items():
            rho = compute_stl_robustness(y_sample, formula, feature_names)
            sample_robustness.append(rho)
        
        avg_rho = np.mean(sample_robustness)
        robustness_scores.append(avg_rho)
    
    return np.array(robustness_scores)


def compute_sample_robustness_single(y_pred, formula, feature_names):
    n_samples = y_pred.shape[0]
    robustness = np.zeros(n_samples)

    for i in range(n_samples):
        y_sample = y_pred[i].T  # (horizon, features)
        robustness[i] = compute_stl_robustness(
            y_sample, formula, feature_names
        )

    return robustness


def compute_violation_rate_among_accurate(errors, robustness, threshold_type='top20', verbose=True):
    if threshold_type == 'top20':
        threshold = np.percentile(errors, 20)
    elif threshold_type == 'median':
        threshold = np.median(errors)
    else:
        raise ValueError(f"Unknown threshold_type: {threshold_type}")
    
    accurate_mask = errors <= threshold
    n_accurate = np.sum(accurate_mask)
    
    if n_accurate == 0:
        violation_rate = 0.0
        n_violations = 0
    else:
        accurate_robustness = robustness[accurate_mask]
        n_violations = np.sum(accurate_robustness < 0)
        violation_rate = n_violations / n_accurate
    
    if verbose:
        print(f"\nViolation Rate Among Accurate Predictions:")
        print(f"  Threshold type: {threshold_type}")
        print(f"  Threshold (τ): {threshold:.6f}")
        print(f"  Accurate predictions (e ≤ τ): {n_accurate}/{len(errors)} ({100*n_accurate/len(errors):.1f}%)")
        print(f"  Violations among accurate (ρ < 0): {n_violations}/{n_accurate} ({100*violation_rate:.1f}%)")
        print(f"  P(ρ<0 | e≤τ) = {violation_rate:.4f}")
    
    return violation_rate, threshold, n_accurate, n_violations


def plot_error_vs_robustness(errors, robustness,
                             threshold=None, save_path='plots/error_vs_robustness.png'):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.scatter(errors, robustness, 
              alpha=0.6, s=30, label='Baseline', 
              color='blue', edgecolors='darkblue', linewidths=0.5)
    
    if threshold is not None:
        bad_mask = (errors <= threshold) & (robustness < 0)
        if np.any(bad_mask):
            ax.scatter(errors[bad_mask], robustness[bad_mask],
                      s=100, marker='x', color='red', linewidths=2,
                      label='Accurate but Violating', zorder=10)
    
    ax.axhline(y=0, color='black', linestyle='--', linewidth=2, 
              label='Violation Boundary (ρ=0)', zorder=5)
    
    if threshold is not None:
        ax.axvline(x=threshold, color='gray', linestyle=':', linewidth=1.5,
                  label=f'Accuracy Threshold (τ={threshold:.4f})', alpha=0.7)
    
    ax.set_xlabel('Numeric Error (MAE)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Semantic Robustness (ρ)', fontsize=12, fontweight='bold')
    ax.set_title('Error vs Robustness: Accurate but Semantically Wrong?', 
                fontsize=14, fontweight='bold')
    
    handles, labels = ax.get_legend_handles_labels()
    
    seen = set()
    unique_handles = []
    unique_labels = []
    for handle, label in zip(handles, labels):
        if label not in seen:
            seen.add(label)
            unique_handles.append(handle)
            unique_labels.append(label)
    
    ax.legend(unique_handles, unique_labels, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10)
    ax.grid(True, alpha=0.3)
    
    n_bad = np.sum((errors <= threshold) & (robustness < 0)) if threshold else 0
    
    textstr = f'Low error (≤{threshold:.4f}) but ρ<0: {n_bad}'
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {save_path}")


def plot_error_vs_robustness_semantic(
    errors, rho,
    threshold,
    save_path='plots/error_vs_robustness_semantic.png'
):
    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.5])

    ax_scatter = fig.add_subplot(gs[0])
    ax_bar = fig.add_subplot(gs[1])

    ax_scatter.scatter(errors, rho, s=25, alpha=0.6,
                      color='tab:blue', edgecolors='k', linewidths=0.3)

    ax_scatter.axhline(0, color='black', linestyle='--', linewidth=1.5)
    ax_scatter.axvline(threshold, color='gray', linestyle=':', linewidth=1.5)

    ax_scatter.fill_betweenx(
        y=[min(rho.min(), -0.5), 0],
        x1=errors.min(),
        x2=threshold,
        color='red',
        alpha=0.08
    )

    ax_scatter.set_title('Baseline (Unconditioned)', fontsize=13, fontweight='bold')
    ax_scatter.set_xlabel('Numeric Error (MAE)')
    ax_scatter.set_ylabel('Semantic Robustness (ρ)')
    ax_scatter.grid(alpha=0.3)

    def compute_violation_rate(errors, rho):
        mask = errors <= threshold
        if np.sum(mask) == 0:
            return 0.0
        return np.mean(rho[mask] < 0)

    vr = compute_violation_rate(errors, rho)

    ax_bar.bar(['Baseline'], [vr * 100], color='tab:blue', alpha=0.8)
    ax_bar.set_ylabel('Violation Rate among Accurate (%)')
    ax_bar.set_title('Hidden Semantic Failures', fontweight='bold')
    ax_bar.set_ylim(0, vr * 120 + 1)
    ax_bar.grid(axis='y', alpha=0.3)

    ax_bar.text(0, vr * 100 + 0.5, f'{vr*100:.1f}%',
                ha='center', fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Semantic robustness plot saved to {save_path}")