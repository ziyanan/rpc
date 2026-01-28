#!/usr/bin/env python3

import os
import sys
import torch
import numpy as np
import argparse
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from smoothing import STLRandomizedSmoother
from net.net import LogicGuidedDiffusionForecast
from analysis.stl_formulas import Always, Eventually, Atomic, STLAnd, STLOr, STLNot
from telex import synth
import telex.stl as telex_stl
import telex.parametrizer as parametrizer
from smoothing.test_smoothing import create_meaningful_stl_formulas as base_create_meaningful_stl_formulas


def load_model(model_path, device='cpu', force_conditioned=None):
    checkpoint = torch.load(model_path, map_location=device)
    
    in_channels = checkpoint.get('in_channels', 7)
    forecast_horizon = checkpoint.get('forecast_horizon', 6)
    
    # Try to load config.json from the same directory if available
    model_dir = os.path.dirname(model_path)
    config_path = os.path.join(model_dir, 'config.json')
    
    if os.path.exists(config_path):
        import json
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
        seq_model_type = model_config.get('seq_model_type', 'seq2seq')
        # loaded config.json
    else:
        # Use new reduced capacity defaults
        diffusion_hidden = 32
        diffusion_levels = 2
        seq_d_model = 64
        seq_n_heads = 2
        seq_n_layers = 2
        num_diffusion_steps = 10
        use_conditioned = True
        seq_model_type = 'seq2seq'
        # fallback defaults
    
    # Allow manual override
    if force_conditioned is not None:
        use_conditioned = (force_conditioned == 'true')
        print(f"OVERRIDE: use_conditioned_diffusion={use_conditioned} (forced via --force-conditioned)")
    
    model = LogicGuidedDiffusionForecast(
        in_channels=in_channels,
        diffusion_hidden=diffusion_hidden,
        diffusion_levels=diffusion_levels,
        seq_model_type=seq_model_type,
        seq_d_model=seq_d_model,
        seq_n_heads=seq_n_heads,
        seq_n_layers=seq_n_layers,
        forecast_horizon=forecast_horizon,
        num_diffusion_steps=num_diffusion_steps,
        use_conditioned_diffusion=use_conditioned
    )
    
    # Try loading with strict=True first, fall back to mapping/flexible loading
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError as e:
        error_msg = str(e)
        state_dict = checkpoint['model_state_dict']
        
        # Check if it's old transformer architecture
        if 'seq2seq.encoder.layers' in error_msg or 'seq2seq.decoder.layers' in error_msg:
            print("⚠️  Detected old transformer architecture, mapping keys...")
            new_state_dict = {}
            
            # Map old keys to new keys
            for key, value in state_dict.items():
                new_key = key
                # Map encoder/decoder layer structure
                new_key = new_key.replace('seq2seq.encoder.layers.', 'seq2seq.encoder_layers.')
                new_key = new_key.replace('seq2seq.decoder.layers.', 'seq2seq.decoder_layers.')
                new_key = new_key.replace('.multihead_attn.', '.cross_attn.')
                new_key = new_key.replace('.linear1.', '.ffn.0.')
                new_key = new_key.replace('.linear2.', '.ffn.3.')
                
                # Map positional encoding
                if key == 'seq2seq.pos_encoder.pe' or key == 'seq2seq.pos_decoder.pe':
                    # Skip these - new version uses learnable pos_embedding instead
                    continue
                new_key = new_key.replace('seq2seq.decoder_start_token', 'seq2seq.start_token')
                
                # Skip norm biases (new LayerNorm doesn't have bias)
                if '.norm' in new_key and '.bias' in new_key:
                    continue
                # Skip encoder norm2 (structure difference)
                if '.norm2.' in new_key and 'encoder_layers' in new_key:
                    continue
                    
                new_state_dict[new_key] = value
            
            # Load with strict=False
            missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
            print(f"Loaded with key mapping (missing: {len(missing_keys)}, unexpected: {len(unexpected_keys)})")
        
        # Check if it's missing condition_weight (old model before this parameter was added)
        elif 'condition_weight' in error_msg:
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            print(f"Loaded with strict=False (missing: {len(missing_keys)}, unexpected: {len(unexpected_keys)})")
        
        else:
            raise e
    
    model.to(device)
    model.eval()
    
    print(f"Model loaded from {model_path}")
    return model


def load_test_data(device='cpu'):
    ds_dir = 'pytorch_datasets'
    
    X_test = np.load(os.path.join(ds_dir, 'X_clean_test.npy'))
    Y_test = np.load(os.path.join(ds_dir, 'Y_test.npy'))
    
    # Transpose to (N, features, timesteps)
    X_test = np.transpose(X_test, (0, 2, 1))
    Y_test = np.transpose(Y_test, (0, 2, 1))
    
    # Limit for testing
    max_samples = 500
    X_test = X_test[:max_samples]
    Y_test = Y_test[:max_samples]
    
    X_test = torch.FloatTensor(X_test).to(device)
    Y_test = torch.FloatTensor(Y_test).to(device)
    
    return X_test, Y_test


def extract_gt_bounds_with_telex(Y_test, feature_names, horizon=6, optmethod='gradient'):
    if isinstance(Y_test, torch.Tensor):
        Y_test = Y_test.cpu().numpy()
    
    bounds = {}
    telex_formulas = {}
    
    print("\n" + "="*80)
    print("GROUND TRUTH BOUNDS EXTRACTION")
    print("="*80)
    print(f"Extracting bounds from {Y_test.shape[0]} samples x {horizon} timesteps")
    print(f"Using true min/max across ALL samples and ALL timesteps")
    print(f"These bounds define what values the GT data actually achieves\n")
    
    lower_template = f"G[0,{horizon-1}](x >= a? -10;10)"
    upper_template = f"G[0,{horizon-1}](x <= a? -10;10)"
    
    for idx, feature_name in enumerate(feature_names):
        print(f"\n[{idx+1}/{len(feature_names)}] Processing feature: {feature_name}")
        
        # Extract feature data: (N, timesteps)
        feature_data = Y_test[:, idx, :]
        
        # Compute statistics
        mean_val = np.mean(feature_data)
        std_val = np.std(feature_data)
        min_val = np.min(feature_data)
        max_val = np.max(feature_data)
        
        print(f"  Statistics: mean={mean_val:.4f}, std={std_val:.4f}")
        print(f"  True bounds: min={min_val:.4f}, max={max_val:.4f}")
        
        # Use true min/max as bounds
        lower_bound = min_val
        upper_bound = max_val
        
        print(f"  Final STL bounds: [{lower_bound:.4f}, {upper_bound:.4f}]")
        
        bounds[feature_name] = (lower_bound, upper_bound)
        telex_formulas[feature_name] = {
            'lower': None,
            'upper': None
        }
    
    print("\n" + "="*80)
    print("GROUND TRUTH BOUNDS SUMMARY")
    print("="*80)
    print("These are the TRUE min/max values observed in ground truth")
    print(f"across {Y_test.shape[0]} samples and {horizon} timesteps per sample\n")
    for feature_name, (lower, upper) in bounds.items():
        print(f"{feature_name:>15s}: [{lower:>8.4f}, {upper:>8.4f}]")
    print("\n" + "="*80 + "\n")
    
    return bounds, telex_formulas


def compute_temporal_derivative_threshold(Y_data, feature_idx, percentile=95):
    # Y_data: (N, features, timesteps)
    feature_data = Y_data[:, feature_idx, :]  # (N, timesteps)
    
    # Compute temporal differences |y_{t+1} - y_t|
    diffs = np.abs(np.diff(feature_data, axis=1))  # (N, timesteps-1)
    
    # Use percentile (not max!) to get meaningful threshold
    threshold = np.percentile(diffs, percentile)
    return float(threshold)


def create_meaningful_stl_formulas(Y_data, feature_names, horizon=6, verbose=False):
    feature_idx_map = {name: idx for idx, name in enumerate(feature_names)}
    num_features = len(feature_names)
    formulas = {}
    
    # Handle both single sample and batch
    if Y_data.ndim == 2:
        # Single sample: (num_features, horizon) -> (1, num_features, horizon)
        Y_data = Y_data[np.newaxis, :]
        single_sample = True
    else:
        single_sample = False
    
    # ========================================================================
    # Formula Type 1: CROSS-SIGNAL CONSISTENCY (Absolute difference constraints)
    # ========================================================================
    if 'avg_speed' in feature_idx_map and 'avg_occupancy' in feature_idx_map:
        speed_idx = feature_idx_map['avg_speed']
        occ_idx = feature_idx_map['avg_occupancy']
        formula_name = "traffic_consistency_speed_occ_diff"

        try:
            traces = []
            for i in range(Y_data.shape[0]):
                traces.append({
                    'avg_speed': Y_data[i, speed_idx, :].tolist(),
                    'avg_occupancy': Y_data[i, occ_idx, :].tolist()
                })

            template = f"G[0,{horizon-1}]({{avg_speed - avg_occupancy}} <= a? -100;100)"

            from telex import synth
            stlsyn, value, dur = synth.synthSTLParam(template, traces, optmethod='gradient')

            threshold = float(stlsyn.subformula.bound)

            if verbose:
                print(
                    f"{'speed_occ_diff':>15s}: "
                    f"G[0,{horizon-1}](|avg_speed - avg_occupancy| <= {threshold:.3f}) [TeLEx-hard]"
                )

            w_pos = np.zeros(num_features)
            w_pos[speed_idx] = 1.0
            w_pos[occ_idx] = -1.0

            w_neg = np.zeros(num_features)
            w_neg[speed_idx] = -1.0
            w_neg[occ_idx] = 1.0

            diff_pos = Atomic(w=w_pos, c=threshold, feature_idx=None, relop="<=")
            diff_neg = Atomic(w=w_neg, c=threshold, feature_idx=None, relop="<=")

            formulas[formula_name] = Always(
                STLAnd(diff_pos, diff_neg),
                t_start=0,
                t_end=horizon-1
            )

        except Exception as e:
            raise RuntimeError(
                f"Type-1 STL generation failed for {formula_name}: {e}"
            ) from e
    
    if 'total_flow' in feature_idx_map and 'avg_speed' in feature_idx_map:
        flow_idx = feature_idx_map['total_flow']
        speed_idx = feature_idx_map['avg_speed']
        formula_name = "traffic_consistency_flow_speed_diff"

        try:
            traces = []
            for i in range(Y_data.shape[0]):
                traces.append({
                    'total_flow': Y_data[i, flow_idx, :].tolist(),
                    'avg_speed': Y_data[i, speed_idx, :].tolist()
                })

            template = f"G[0,{horizon-1}]({{total_flow - avg_speed}} <= a? -100;100)"

            from telex import synth
            stlsyn, value, dur = synth.synthSTLParam(
                template,
                traces,
                optmethod='gradient'
            )

            threshold = float(stlsyn.subformula.bound)

            if verbose:
                print(
                    f"{'flow_speed_diff':>15s}: "
                    f"G[0,{horizon-1}](|total_flow - avg_speed| <= {threshold:.3f}) [TeLEx-hard]"
                )

            w_pos = np.zeros(num_features)
            w_pos[flow_idx] = 1.0
            w_pos[speed_idx] = -1.0

            w_neg = np.zeros(num_features)
            w_neg[flow_idx] = -1.0
            w_neg[speed_idx] = 1.0

            diff_pos = Atomic(w=w_pos, c=threshold, feature_idx=None, relop="<=")
            diff_neg = Atomic(w=w_neg, c=threshold, feature_idx=None, relop="<=")

            formulas[formula_name] = Always(
                STLAnd(diff_pos, diff_neg),
                t_start=0,
                t_end=horizon-1
            )

        except Exception as e:
            raise RuntimeError(
                f"Type-1 STL generation failed for {formula_name}: {e}"
            ) from e
    
    # ========================================================================
    # Formula Type 2: REASONABLE BOUNDS (Absolute min/max)
    # ========================================================================
    for feature_name in feature_names:
        if feature_name not in feature_idx_map:
            continue

        idx = feature_idx_map[feature_name]
        feature_data = Y_data[:, idx, :]  # (n_samples, horizon)

        try:
            traces = []
            for i in range(Y_data.shape[0]):
                traces.append({
                    feature_name: feature_data[i, :].tolist()
                })

            from telex import synth

            lower_template = f"G[0,{horizon-1}]({feature_name} >= a? -100;100)"
            stlsyn_lower, _, _ = synth.synthSTLParam(
                lower_template,
                traces,
                optmethod="gradient"
            )

            min_val = float(stlsyn_lower.subformula.bound)

            upper_template = f"G[0,{horizon-1}]({feature_name} <= b? -100;100)"
            stlsyn_upper, _, _ = synth.synthSTLParam(
                upper_template,
                traces,
                optmethod="gradient"
            )

            max_val = float(stlsyn_upper.subformula.bound)

            w_pos = np.zeros(num_features)
            w_pos[idx] = 1.0
            w_neg = np.zeros(num_features)
            w_neg[idx] = -1.0

            # Split into TWO separate formulas: lower bound and upper bound
            formula_name_lower = f"{feature_name}_reasonable_range_lower"
            lower_atomic = Atomic(w=w_pos, c=min_val, feature_idx=idx, relop=">=")
            formulas[formula_name_lower] = Always(
                lower_atomic,
                t_start=0,
                t_end=horizon-1
            )

            formula_name_upper = f"{feature_name}_reasonable_range_upper"
            upper_atomic = Atomic(w=w_neg, c=-max_val, feature_idx=idx, relop=">=")
            formulas[formula_name_upper] = Always(
                upper_atomic,
                t_start=0,
                t_end=horizon-1
            )

            if verbose:
                print(
                    f"{feature_name:>15s}: "
                    f"G[0,{horizon-1}]({feature_name} >= {min_val:.3f}) [TeLEx-hard, lower]"
                )
                print(
                    f"{'':>15s}: "
                    f"G[0,{horizon-1}]({feature_name} <= {max_val:.3f}) [TeLEx-hard, upper]"
                )

            # TEMPORARILY REMOVED: Window formulas (commented out)
            pass

            if horizon > 1:
                # Split Eventually into TWO separate formulas
                formula_name_eventually_lower = f"{feature_name}_reasonable_range_eventually_lower"
                formulas[formula_name_eventually_lower] = Eventually(
                    Atomic(w=w_pos, c=min_val, feature_idx=idx, relop=">="),
                    t_start=0,
                    t_end=horizon-1
                )

                formula_name_eventually_upper = f"{feature_name}_reasonable_range_eventually_upper"
                formulas[formula_name_eventually_upper] = Eventually(
                    Atomic(w=w_neg, c=-max_val, feature_idx=idx, relop=">="),
                    t_start=0,
                    t_end=horizon-1
                )

                if verbose:
                    print(
                        f"{'':>15s}: "
                        f"F[0,{horizon-1}]({feature_name} >= {min_val:.3f}) [TeLEx-hard, eventually lower]"
                    )
                    print(
                        f"{'':>15s}: "
                        f"F[0,{horizon-1}]({feature_name} <= {max_val:.3f}) [TeLEx-hard, eventually upper]"
                    )

        except Exception as e:
            raise RuntimeError(
                f"Type-2 STL generation failed for {feature_name}: {e}"
            ) from e
    
    # ========================================================================
    # Formula Type 3: COMPLEX TEMPORAL LOGIC (Derivative-based, oscillations, anti-smoothing, cross-signal sync)
    # ========================================================================
    derivatives = np.diff(Y_data, axis=2)  # (n_samples, num_features, horizon-1)
    
    for feature_name in feature_names:
        if feature_name not in feature_idx_map:
            continue
        
        idx = feature_idx_map[feature_name]
        feature_data = Y_data[:, idx, :]  # (n_samples, horizon)
        feature_derivatives = derivatives[:, idx, :]  # (n_samples, horizon-1)
        
        w_pos = np.zeros(num_features)
        w_pos[idx] = 1.0
        w_neg = np.zeros(num_features)
        w_neg[idx] = -1.0
        
        # Formula 1: Range Coverage - Predictions must span sufficient range
        # Learn GT range (max - min), require predictions span at least threshold
        # This catches flat predictions that don't vary enough
        try:
            traces = []
            for i in range(Y_data.shape[0]):
                gt_range = np.max(feature_data[i, :]) - np.min(feature_data[i, :])
                traces.append({'range': [gt_range]})
            
            # Learn range threshold (use 50th percentile of GT ranges as minimum)
            range_template = "G[0,0](range >= r? 0;10)"
            from telex import synth
            stlsyn_range, _, _ = synth.synthSTLParam(range_template, traces, optmethod='gradient')
            range_threshold = float(stlsyn_range.subformula.bound) * 0.6  # Require 60% of learned range
            
            # Encode as: F[0,H-1](feature >= baseline + threshold/2) AND F[0,H-1](feature <= baseline - threshold/2)
            # This ensures predictions span at least the threshold range
            # Approximate baseline as middle of GT range
            baseline = np.mean([np.mean(feature_data[i, :]) for i in range(Y_data.shape[0])])
            high_atomic = Atomic(w=w_pos, c=baseline + range_threshold/2, feature_idx=idx, relop=">=")
            low_atomic = Atomic(w=w_neg, c=-(baseline - range_threshold/2), feature_idx=idx, relop=">=")
            formulas[f"{feature_name}_range_coverage"] = STLAnd(
                Eventually(high_atomic, t_start=0, t_end=horizon-1),
                Eventually(low_atomic, t_start=0, t_end=horizon-1)
            )
            if verbose:
                print(f"{feature_name:>15s}: F[0,{horizon-1}](≥{baseline + range_threshold/2:.3f}) AND F[0,{horizon-1}](≤{baseline - range_threshold/2:.3f}) [Range Coverage, threshold={range_threshold:.3f}]")
        except Exception as e:
            if verbose:
                print(f"ERROR: Range coverage formula failed for {feature_name}: {e}")
        
        # Formula 2: Derivative Magnitude - Predictions must have significant changes
        # Learn typical |ΔGT| from ground truth, require predictions have |Δ| exceeding threshold
        if horizon > 1:
            try:
                traces = []
                for i in range(Y_data.shape[0]):
                    abs_derivs = np.abs(feature_derivatives[i, :])
                    traces.append({f'abs_delta_{feature_name}': abs_derivs.tolist()})
                
                # Learn typical |Δ| magnitude from GT (use 30th percentile as minimum threshold)
                deriv_template = f"G[0,{horizon-2}](abs_delta_{feature_name} >= d? 0;10)"
                from telex import synth
                stlsyn_deriv, _, _ = synth.synthSTLParam(deriv_template, traces, optmethod='gradient')
                deriv_threshold = float(stlsyn_deriv.subformula.bound) * 0.5  # Require 50% of learned magnitude
                
                # Encode as: F[0,H-2](|Δ| >= threshold), meaning at least one significant change
                # Approximate |Δ| >= threshold as: (x[t+1] - x[t] >= threshold) OR (x[t] - x[t+1] >= threshold)
                # Use Eventually to check if ANY timestep has significant change
                increase_atomic = Atomic(w=w_pos, c=deriv_threshold, feature_idx=idx, relop=">=")
                decrease_atomic = Atomic(w=w_neg, c=deriv_threshold, feature_idx=idx, relop=">=")
                formulas[f"{feature_name}_derivative_magnitude"] = Eventually(
                    STLOr(increase_atomic, decrease_atomic),
                    t_start=0,
                    t_end=horizon-2
                )
                if verbose:
                    print(f"{feature_name:>15s}: F[0,{horizon-2}](|Δ{feature_name}| >= {deriv_threshold:.3f}) [Derivative Magnitude]")
            except Exception as e:
                if verbose:
                    print(f"ERROR: Derivative magnitude formula failed for {feature_name}: {e}")
        
        # Formula 3: Sustained Change - Check for consecutive increases or decreases
        # Learn threshold for sustained change, require at least 2 consecutive steps in same direction
        if horizon > 2:
            try:
                traces = []
                for i in range(Y_data.shape[0]):
                    derivs = feature_derivatives[i, :]
                    traces.append({f'delta_{feature_name}': derivs.tolist()})
                
                # Learn positive change threshold
                pos_template = f"G[0,{horizon-2}](delta_{feature_name} >= dp? 0;10)"
                from telex import synth
                stlsyn_pos, _, _ = synth.synthSTLParam(pos_template, traces, optmethod='gradient')
                pos_threshold = float(stlsyn_pos.subformula.bound) * 0.3
                
                # Encode as: F[0,H-3](G[0,1](Δ >= threshold)) - sustained increase for 2 steps
                pos_atomic = Atomic(w=w_pos, c=pos_threshold, feature_idx=idx, relop=">=")
                sustained_increase = Always(pos_atomic, t_start=0, t_end=1)
                formulas[f"{feature_name}_sustained_change"] = Eventually(
                    sustained_increase,
                    t_start=0,
                    t_end=horizon-3
                )
                if verbose:
                    print(f"{feature_name:>15s}: F[0,{horizon-3}](G[0,1](Δ{feature_name} >= {pos_threshold:.3f})) [Sustained Change]")
            except Exception as e:
                if verbose:
                    print(f"ERROR: Sustained change formula failed for {feature_name}: {e}")
        
        # Formula 5: Anti-smoothing (KEEP - good formula)
        # ¬F[0,H-3](G[0,2](|Δfeature| ≤ ε))
        if horizon > 3:
            try:
                traces = []
                for i in range(Y_data.shape[0]):
                    deriv_signal = []
                    for t in range(horizon - 1):
                        deriv_signal.append(abs(feature_data[i, t+1] - feature_data[i, t]))
                    traces.append({f'abs_delta_{feature_name}': deriv_signal})
                
                # Learn epsilon (small slope threshold) for |Δ| ≤ ε
                eps_template = f"G[0,2](abs_delta_{feature_name} <= eps? 0;10)"
                from telex import synth
                stlsyn_eps, _, _ = synth.synthSTLParam(eps_template, traces, optmethod='gradient')
                eps = float(stlsyn_eps.subformula.bound)
                
                # Build: G[0,2](|Δ| ≤ ε) approximated as: (x[t+1] - x[t] <= eps) AND (x[t] - x[t+1] <= eps)
                # Simplified: check if consecutive values are very close
                eps_pos = Atomic(w=w_pos, c=eps, feature_idx=idx, relop="<=")
                eps_neg = Atomic(w=w_neg, c=-eps, feature_idx=idx, relop=">=")
                smooth_always = Always(STLAnd(eps_pos, eps_neg), t_start=0, t_end=2)
                smooth_eventually = Eventually(smooth_always, t_start=0, t_end=horizon-3)
                formulas[f"{feature_name}_anti_smoothing"] = STLNot(smooth_eventually)
                if verbose:
                    print(f"{feature_name:>15s}: ¬F[0,{horizon-3}](G[0,2](|Δ{feature_name}| <= {eps:.3f})) [Anti-Smoothing]")
            except Exception as e:
                if verbose:
                    print(f"ERROR: Anti-smoothing formula failed for {feature_name}: {e}")
    
    # Formula 4: Cross-signal timing consistency
    # G[1,H-1]((Δtotal_flow ≤ d) → F[0,1](Δavg_speed ≤ ds))
    if 'total_flow' in feature_idx_map and 'avg_speed' in feature_idx_map:
        flow_idx = feature_idx_map['total_flow']
        speed_idx = feature_idx_map['avg_speed']
        
        try:
            traces = []
            for i in range(Y_data.shape[0]):
                flow_deriv = []
                speed_deriv = []
                for t in range(horizon - 1):
                    flow_deriv.append(Y_data[i, flow_idx, t+1] - Y_data[i, flow_idx, t])
                    speed_deriv.append(Y_data[i, speed_idx, t+1] - Y_data[i, speed_idx, t])
                traces.append({
                    'delta_total_flow': flow_deriv,
                    'delta_avg_speed': speed_deriv
                })
            
            # Learn d (flow drop threshold)
            flow_drop_template = f"G[0,{horizon-2}](delta_total_flow <= d? -100;0)"
            from telex import synth
            stlsyn_d, _, _ = synth.synthSTLParam(flow_drop_template, traces, optmethod='gradient')
            d = float(stlsyn_d.subformula.bound)
            
            # Learn ds (speed drop threshold)
            speed_drop_template = f"G[0,{horizon-2}](delta_avg_speed <= ds? -100;0)"
            stlsyn_ds, _, _ = synth.synthSTLParam(speed_drop_template, traces, optmethod='gradient')
            ds = float(stlsyn_ds.subformula.bound)
            
            # Build formula: G[0,H-2]((Δflow ≤ d) → F[0,1](Δspeed ≤ ds))
            w_flow_neg = np.zeros(num_features)
            w_flow_neg[flow_idx] = -1.0
            w_speed_neg = np.zeros(num_features)
            w_speed_neg[speed_idx] = -1.0
            
            flow_drop_atomic = Atomic(w=w_flow_neg, c=-d, feature_idx=flow_idx, relop=">=")
            speed_drop_atomic = Atomic(w=w_speed_neg, c=-ds, feature_idx=speed_idx, relop=">=")
            speed_drop_eventually = Eventually(speed_drop_atomic, t_start=0, t_end=1)
            formulas["flow_speed_sync"] = Always(
                STLOr(STLNot(flow_drop_atomic), speed_drop_eventually),
                t_start=0,
                t_end=horizon-2
            )
            if verbose:
                print(f"{'flow_speed_sync':>15s}: G[0,{horizon-2}]((Δtotal_flow <= {d:.3f}) -> F[0,1](Δavg_speed <= {ds:.3f})) [Cross-Signal Sync]")
        except Exception as e:
            if verbose:
                print(f"ERROR: Cross-signal sync formula failed: {e}")
    
    if verbose:
        print("="*80 + "\n")
        print(f"Total formulas created: {len(formulas)}")
        print("These formulas can be violated by predictions, showing conditioning benefit!\n")
    
    # Separate directly enforced vs other formulas
    directly_enforced = {}
    other_formulas = {}
    
    # Directly enforced: Type 3 (reasonable bounds) - enforced by learned bounds in repair
    # Also Type 2 correlation formulas (flow_speed_correlation, aq_correlation) - enforced in repair
    for formula_name, formula in formulas.items():
        if 'reasonable_range' in formula_name:
            # Type 3: Reasonable bounds (enforced by learned bounds)
            directly_enforced[formula_name] = formula
        elif 'traffic_consistency_high_flow_not_crawl' in formula_name:
            # Type 2: flow_speed_correlation (enforced in repair)
            directly_enforced[formula_name] = formula
        else:
            # Type 2 (high_speed_low_occ) and Type 4 (complex temporal) are NOT directly enforced
            other_formulas[formula_name] = formula
    
    return formulas, directly_enforced, other_formulas


def create_meaningful_stl_formulas(Y_data, feature_names, horizon=6, verbose=False, X_data=None):
    # Override with shared logic generator to stay in sync with net_training scripts
    return base_create_meaningful_stl_formulas(
        Y_data, feature_names, horizon=horizon, verbose=verbose, X_data=X_data
    )


def test_single_sample(smoother, x, y_gt=None, alpha=0.001):
    print("\n" + "="*80)
    print("SINGLE SAMPLE CERTIFICATION TEST")
    print("="*80)
    
    # Standard prediction
    y_pred = smoother.predict(x)
    print(f"Standard prediction shape: {y_pred.shape}")
    
    # Check ground truth STL satisfaction if provided
    if y_gt is not None:
        from smoothing.stl_utils import compute_stl_robustness, binary_classification
        
        print("\nGround Truth STL Satisfaction:")
        for formula_name, formula in smoother.stl_formulas.items():
            rho_gt = compute_stl_robustness(y_gt[0], formula)
            sat_gt = binary_classification(np.array([rho_gt]))[0]
            result_gt = "SATISFIED" if sat_gt == 1 else "VIOLATED"
            print(f"  {formula_name:35s}: {result_gt:10s} (rho={rho_gt:.4f})")
    
    # Smoothed prediction
    print("\nComputing smoothed predictions...")
    smooth_results = smoother.predict_smooth(x)
    
    print("\nSmoothed Prediction STL Satisfaction:")
    for formula_name, classifications in smooth_results['smoothed_classifications'].items():
        p_sat = smooth_results['proportion_satisfied'][formula_name][0]
        result = "SATISFIED" if classifications[0] == 1 else "VIOLATED"
        print(f"  {formula_name:35s}: {result:10s} (p={p_sat:.3f})")
    
    # Certification
    print(f"\nCertifying with α={alpha} (confidence={1-alpha:.4f})...")
    cert_results = smoother.certify(x, alpha=alpha, verbose=False)
    
    print("\nCertification Results:")
    for formula_name, certifications in cert_results.items():
        cert = certifications[0]
        diag = cert['diagnostics']
        
        print(f"\n  {formula_name}:")
        print(f"    Prediction:        {cert['prediction']}")
        print(f"    Empirical p_hat:   {diag['p_hat']:.4f} (count={diag['count']}/{diag['n_samples']})")
        print(f"    Lower bound p_L:   {diag['p_lower']:.4f}")
        print(f"    Certified Radius:  {cert['certified_radius']:.4f}")
        
        if 'failed_reason' in diag:
            print(f"    FAILED: {diag['failed_reason']}")
        elif cert['certified_radius'] > 0:
            print(f"    CERTIFIED: Any L2 perturbation <= {cert['certified_radius']:.4f} maintains {cert['prediction']}")
        else:
            print(f"    NOT CERTIFIED")


def test_batch_certification(smoother, X_test, alpha=0.001, batch_size=32):
    print("\n" + "="*80)
    print("BATCH CERTIFICATION TEST")
    print("="*80)
    
    # Certify batch
    aggregate_results = smoother.certify_batch(
        X_test,
        alpha=alpha,
        batch_size=batch_size
    )
    
    # Print aggregate statistics
    print("\nAggregate Statistics:")
    print("-" * 100)
    print(f"{'Formula':<25} {'Sat%':>7} {'Cert%':>7} {'Mean':>8} {'Median':>8} {'P75':>8} {'Std':>8} {'p̂':>7} {'p_L':>7}")
    print("-" * 100)
    
    for formula_name, stats in aggregate_results.items():
        print(
            f"{formula_name:<25} "
            f"{stats['satisfaction_rate']:>6.1%} "
            f"{stats['certified_rate']:>6.1%} "
            f"{stats['avg_certified_radius']:>8.4f} "
            f"{stats['median_certified_radius']:>8.4f} "
            f"{stats['p75_certified_radius']:>8.4f} "
            f"{stats['radius_std']:>8.4f} "
            f"{stats['avg_p_hat']:>7.3f} "
            f"{stats['avg_p_lower']:>7.3f}"
        )
        
        # Show failures if any
        if stats['failed_certifications']:
            for reason, count in stats['failed_certifications'].items():
                print(f"  └─ Failed ({reason}): {count} samples")
    
    return aggregate_results


def test_logic_conditioning_comparison(smoother, x, alpha=0.001):
    print("\n" + "="*80)
    print("LOGIC CONDITIONING COMPARISON (Same Model)")
    print("="*80)
    
    # Test unconditioned
    print("\nCertifying WITHOUT logic conditioning...")
    cert_uncond = smoother.certify(x, alpha=alpha, logic_name=None)
    
    # Test with traffic_flow_stable logic
    print("\nCertifying WITH logic conditioning (traffic_flow_stable)...")
    cert_cond = smoother.certify(x, alpha=alpha, logic_name='traffic_flow_stable')
    
    # Compare
    print("\nComparison:")
    print("-" * 80)
    print(f"{'Formula':<30} {'Uncond R':>12} {'Cond R':>12} {'Improvement':>12}")
    print("-" * 80)
    
    for formula_name in cert_uncond.keys():
        r_uncond = cert_uncond[formula_name][0]['certified_radius']
        r_cond = cert_cond[formula_name][0]['certified_radius']
        improvement = r_cond - r_uncond
        
        print(
            f"{formula_name:<30} "
            f"{r_uncond:>12.4f} "
            f"{r_cond:>12.4f} "
            f"{improvement:>+12.4f}"
        )
    
    print("-" * 80)


def save_comparison_results(results_cond, results_uncond, test_type, attack_type=None, 
                            epsilon=None, sigma=0.1, n_samples=100, alpha=0.001, num_test_samples=500):
    eval_dir = 'eval_results'
    os.makedirs(eval_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Prepare data for CSV
    csv_rows = []
    csv_rows.append(
        "Formula,"
        "Uncond_Mean,Cond_Mean,Improvement,"
        "Uncond_Median,Cond_Median,"
        "Uncond_P75,Cond_P75,"
        "Uncond_Cert_Percent,Cond_Cert_Percent"
    )
    
    for formula_name in results_cond.keys():
        uncond_acr = results_uncond[formula_name]['avg_certified_radius']
        cond_acr = results_cond[formula_name]['avg_certified_radius']
        improvement = cond_acr - uncond_acr
        uncond_median = results_uncond[formula_name]['median_certified_radius']
        cond_median = results_cond[formula_name]['median_certified_radius']
        uncond_p75 = results_uncond[formula_name]['p75_certified_radius']
        cond_p75 = results_cond[formula_name]['p75_certified_radius']
        uncond_cert_rate = results_uncond[formula_name]['certified_rate']
        cond_cert_rate = results_cond[formula_name]['certified_rate']
        
        csv_rows.append(
            f"{formula_name},"
            f"{uncond_acr:.6f},{cond_acr:.6f},{improvement:+.6f},"
            f"{uncond_median:.6f},{cond_median:.6f},"
            f"{uncond_p75:.6f},{cond_p75:.6f},"
            f"{uncond_cert_rate:.2%},{cond_cert_rate:.2%}"
        )

    # Add aggregate row to CSV
    if results_cond:
        uncond_means = [v['avg_certified_radius'] for v in results_uncond.values()]
        cond_means = [v['avg_certified_radius'] for v in results_cond.values()]
        uncond_medians = [v['median_certified_radius'] for v in results_uncond.values()]
        cond_medians = [v['median_certified_radius'] for v in results_cond.values()]
        uncond_p75s = [v['p75_certified_radius'] for v in results_uncond.values()]
        cond_p75s = [v['p75_certified_radius'] for v in results_cond.values()]
        uncond_cert_rates = [v['certified_rate'] for v in results_uncond.values()]
        cond_cert_rates = [v['certified_rate'] for v in results_cond.values()]

        agg_uncond_mean = float(np.mean(uncond_means))
        agg_cond_mean = float(np.mean(cond_means))
        agg_improvement = agg_cond_mean - agg_uncond_mean
        agg_uncond_median = float(np.mean(uncond_medians))
        agg_cond_median = float(np.mean(cond_medians))
        agg_uncond_p75 = float(np.mean(uncond_p75s))
        agg_cond_p75 = float(np.mean(cond_p75s))
        agg_uncond_cert = float(np.mean(uncond_cert_rates))
        agg_cond_cert = float(np.mean(cond_cert_rates))

        csv_rows.append(
            f"AGGREGATED (Average),"
            f"{agg_uncond_mean:.6f},{agg_cond_mean:.6f},{agg_improvement:+.6f},"
            f"{agg_uncond_median:.6f},{agg_cond_median:.6f},"
            f"{agg_uncond_p75:.6f},{agg_cond_p75:.6f},"
            f"{agg_uncond_cert:.2%},{agg_cond_cert:.2%}"
        )
    
    # Save CSV
    csv_filename = f"{eval_dir}/comparison_{test_type}_{timestamp}.csv"
    with open(csv_filename, 'w') as f:
        f.write('\n'.join(csv_rows))
    
    # Prepare data for JSON
    json_data = {
        'test_type': test_type,
        'timestamp': timestamp,
        'test_config': {
            'sigma': sigma,
            'n_samples': n_samples,
            'alpha': alpha,
            'num_test_samples': num_test_samples,
            'attack_type': attack_type,
            'epsilon': epsilon
        },
        'results': {}
    }
    
    for formula_name in results_cond.keys():
        json_data['results'][formula_name] = {
            'unconditioned': {
            'avg_certified_radius': float(results_uncond[formula_name]['avg_certified_radius']),
            'median_certified_radius': float(results_uncond[formula_name]['median_certified_radius']),
            'p75_certified_radius': float(results_uncond[formula_name]['p75_certified_radius']),
            'certified_rate': float(results_uncond[formula_name]['certified_rate']),
            'radius_std': float(results_uncond[formula_name]['radius_std'])
            },
            'conditioned': {
            'avg_certified_radius': float(results_cond[formula_name]['avg_certified_radius']),
            'median_certified_radius': float(results_cond[formula_name]['median_certified_radius']),
            'p75_certified_radius': float(results_cond[formula_name]['p75_certified_radius']),
            'certified_rate': float(results_cond[formula_name]['certified_rate']),
            'radius_std': float(results_cond[formula_name]['radius_std'])
            },
            'improvement': {
                'acr_improvement': float(results_cond[formula_name]['avg_certified_radius'] - 
                                        results_uncond[formula_name]['avg_certified_radius']),
                'cert_rate_improvement': float(results_cond[formula_name]['certified_rate'] - 
                                              results_uncond[formula_name]['certified_rate'])
            }
        }

    if results_cond:
        json_data['aggregate'] = {
            'unconditioned': {
                'mean': float(np.mean([v['avg_certified_radius'] for v in results_uncond.values()])),
                'median': float(np.mean([v['median_certified_radius'] for v in results_uncond.values()])),
                'p75': float(np.mean([v['p75_certified_radius'] for v in results_uncond.values()])),
                'certified_rate': float(np.mean([v['certified_rate'] for v in results_uncond.values()])),
            },
            'conditioned': {
                'mean': float(np.mean([v['avg_certified_radius'] for v in results_cond.values()])),
                'median': float(np.mean([v['median_certified_radius'] for v in results_cond.values()])),
                'p75': float(np.mean([v['p75_certified_radius'] for v in results_cond.values()])),
                'certified_rate': float(np.mean([v['certified_rate'] for v in results_cond.values()])),
            }
        }
    
    # Save JSON
    json_filename = f"{eval_dir}/comparison_{test_type}_{timestamp}.json"
    with open(json_filename, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"\nResults saved to:")
    print(f"  CSV: {csv_filename}")
    print(f"  JSON: {json_filename}")


def compare_two_models(model_conditioned, model_unconditioned, X_test, Y_test, 
                       sigma, n_samples, alpha, device, feature_names, batch_size=32,
                       use_adversarial=False, attack_type='gaussian', epsilon=1.5):
    test_type = "ADVERSARIAL" if use_adversarial else "CLEAN"
    print("\n" + "="*80)
    print(f"TWO-MODEL COMPARISON - {test_type} DATA")
    print("="*80 + "\n")
    
    num_samples = len(X_test)
    
    # Store results for each formula across all samples
    from collections import defaultdict
    results_cond_all = defaultdict(lambda: {'radii': [], 'certified_count': 0})
    results_uncond_all = defaultdict(lambda: {'radii': [], 'certified_count': 0})
    
    print(f"Certifying {num_samples} samples (each with its own STL bounds)...\n")
    
    # Prepare adversarial attack if needed
    if use_adversarial:
        from logics.stl_attack_repair import AttackGenerator, TeLExLearner, STLGuidedRepairer
        
        X_test_np = X_test.cpu().numpy() if isinstance(X_test, torch.Tensor) else X_test
        
        print("Learning STL bounds from clean data for repair...")
        learner = TeLExLearner(feature_names, timesteps=X_test_np.shape[2])
        learned_bounds = learner.learn_bounds_simple(X_test_np, feature_names, verbose=False)
        
        # Attack all test data
        print(f"Attacking test data with {attack_type} attack (epsilon={epsilon})...")
        attacker = AttackGenerator(attack_type=attack_type, epsilon=epsilon, seed=None)
        X_attacked_np = attacker.attack(X_test_np)
        X_attacked = torch.FloatTensor(X_attacked_np).to(device)
        
        # Prepare repairer for conditioned model
        repairer = STLGuidedRepairer(feature_names, learned_bounds=learned_bounds)
    else:
        X_attacked = X_test
        repairer = None
        learned_bounds = None
    
    # Process each sample individually with its own STL bounds
    from tqdm import tqdm
    for idx in tqdm(range(num_samples), desc="Processing samples", ncols=80):
        # Extract per-sample ground truth bounds
        y_sample = Y_test[idx:idx+1]  # (1, features, timesteps)
        x_sample_clean = X_test[idx:idx+1]
        
        if use_adversarial:
            x_sample_attacked = X_attacked[idx:idx+1]
            
            # For conditioned model: Repair CLEAN data to certify around clean input
            x_clean_np = x_sample_clean.cpu().numpy()
            x_repaired_np = repairer.repair(x_clean_np, method='comprehensive', verbose=False)
            x_repaired = torch.FloatTensor(x_repaired_np).to(device)
            
            # Certify robustness around clean input (per paper)
            x_cond_input = x_sample_clean
            x_cond_condition = x_repaired
            x_uncond_input = x_sample_clean
        else:
            # Clean data: both models use clean input
            x_cond_input = x_sample_clean
            x_cond_condition = None
            x_uncond_input = x_sample_clean
        
        # Convert sample to numpy for formula creation (always use clean GT for STL bounds)
        y_np = y_sample.cpu().numpy() if isinstance(y_sample, torch.Tensor) else y_sample
        
        # Create STL formulas for THIS sample using meaningful (non-tautological) constraints
        # Note: y_np shape is (1, features, timesteps) - the function expects (N, features, timesteps)
        sample_stl_formulas, _, _ = create_meaningful_stl_formulas(
            y_np,
            feature_names,
            horizon=y_sample.shape[2]
        )
        # Smoothing eval: keep bounds + windowed; drop trend formulas
        sample_stl_formulas = {
            name: formula
            for name, formula in sample_stl_formulas.items()
            if "_trend" not in name
        }
        
        # Create smoothers with sample-specific STL
        # For conditioned model: need to pass x_condition during certification
        smoother_cond = STLRandomizedSmoother(
            model=model_conditioned,
            stl_formulas=sample_stl_formulas,
            sigma=sigma,
            n_samples=n_samples,
            device=device,
            feature_names=feature_names
        )
        
        smoother_uncond = STLRandomizedSmoother(
            model=model_unconditioned,
            stl_formulas=sample_stl_formulas,
            sigma=sigma,
            n_samples=n_samples,
            device=device,
            feature_names=feature_names
        )
        
        # Certify this sample with both models
        # For conditioned model: pass x_condition if adversarial
        if use_adversarial and x_cond_condition is not None:
            cert_cond = smoother_cond.certify(x_cond_input, alpha=alpha, verbose=False, x_condition=x_cond_condition)
        else:
            cert_cond = smoother_cond.certify(x_cond_input, alpha=alpha, verbose=False)
        
        cert_uncond = smoother_uncond.certify(x_uncond_input, alpha=alpha, verbose=False)
        
        # Collect results
        for formula_name in sample_stl_formulas.keys():
            r_cond = cert_cond[formula_name][0]['certified_radius']
            r_uncond = cert_uncond[formula_name][0]['certified_radius']
            
            results_cond_all[formula_name]['radii'].append(r_cond)
            results_uncond_all[formula_name]['radii'].append(r_uncond)
            
            if r_cond > 0:
                results_cond_all[formula_name]['certified_count'] += 1
            if r_uncond > 0:
                results_uncond_all[formula_name]['certified_count'] += 1
    
    # Aggregate results
    print("\n" + "="*80)
    print("AGGREGATING RESULTS")
    print("="*80)
    
    results_cond = {}
    results_uncond = {}
    
    for formula_name in results_cond_all.keys():
        radii_cond = results_cond_all[formula_name]['radii']
        radii_uncond = results_uncond_all[formula_name]['radii']
        
        results_cond[formula_name] = {
            'avg_certified_radius': np.mean(radii_cond),
            'median_certified_radius': np.median(radii_cond),
            'p75_certified_radius': np.percentile(radii_cond, 75),
            'certified_rate': results_cond_all[formula_name]['certified_count'] / num_samples,
            'radius_std': np.std(radii_cond)
        }
        
        results_uncond[formula_name] = {
            'avg_certified_radius': np.mean(radii_uncond),
            'median_certified_radius': np.median(radii_uncond),
            'p75_certified_radius': np.percentile(radii_uncond, 75),
            'certified_rate': results_uncond_all[formula_name]['certified_count'] / num_samples,
            'radius_std': np.std(radii_uncond)
        }
    
    # Compare aggregate results
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    print("-" * 120)
    print(
        f"{'Formula':<25} "
        f"{'Uncond Mean':>12} {'Cond Mean':>12} {'Improvement':>12} "
        f"{'Uncond Med':>11} {'Cond Med':>11} "
        f"{'Uncond P75':>11} {'Cond P75':>11} "
        f"{'Uncond Cert%':>12} {'Cond Cert%':>12}"
    )
    print("-" * 120)
    
    # Collect values for aggregation
    uncond_acrs = []
    cond_acrs = []
    uncond_medians = []
    cond_medians = []
    uncond_p75s = []
    cond_p75s = []
    improvements = []
    uncond_cert_rates = []
    cond_cert_rates = []
    
    for formula_name in results_cond.keys():
        uncond_acr = results_uncond[formula_name]['avg_certified_radius']
        cond_acr = results_cond[formula_name]['avg_certified_radius']
        improvement = cond_acr - uncond_acr
        uncond_median = results_uncond[formula_name]['median_certified_radius']
        cond_median = results_cond[formula_name]['median_certified_radius']
        uncond_p75 = results_uncond[formula_name]['p75_certified_radius']
        cond_p75 = results_cond[formula_name]['p75_certified_radius']

        uncond_cert_rate = results_uncond[formula_name]['certified_rate']
        cond_cert_rate = results_cond[formula_name]['certified_rate']
        
        uncond_acrs.append(uncond_acr)
        cond_acrs.append(cond_acr)
        uncond_medians.append(uncond_median)
        cond_medians.append(cond_median)
        uncond_p75s.append(uncond_p75)
        cond_p75s.append(cond_p75)
        improvements.append(improvement)
        uncond_cert_rates.append(uncond_cert_rate)
        cond_cert_rates.append(cond_cert_rate)
        
        print(
            f"{formula_name:<25} "
            f"{uncond_acr:>12.4f} "
            f"{cond_acr:>12.4f} "
            f"{improvement:>+12.4f} "
            f"{uncond_median:>11.4f} "
            f"{cond_median:>11.4f} "
            f"{uncond_p75:>11.4f} "
            f"{cond_p75:>11.4f} "
            f"{uncond_cert_rate:>11.1%} "
            f"{cond_cert_rate:>11.1%}"
        )
    
    # Print aggregated summary
    print("-" * 120)
    avg_uncond_acr = np.mean(uncond_acrs)
    avg_cond_acr = np.mean(cond_acrs)
    avg_uncond_median = np.mean(uncond_medians)
    avg_cond_median = np.mean(cond_medians)
    avg_uncond_p75 = np.mean(uncond_p75s)
    avg_cond_p75 = np.mean(cond_p75s)
    avg_improvement = np.mean(improvements)
    avg_uncond_cert_rate = np.mean(uncond_cert_rates)
    avg_cond_cert_rate = np.mean(cond_cert_rates)
    
    print(
        f"{'AGGREGATED (Average)':<25} "
        f"{avg_uncond_acr:>12.4f} "
        f"{avg_cond_acr:>12.4f} "
        f"{avg_improvement:>+12.4f} "
        f"{avg_uncond_median:>11.4f} "
        f"{avg_cond_median:>11.4f} "
        f"{avg_uncond_p75:>11.4f} "
        f"{avg_cond_p75:>11.4f} "
        f"{avg_uncond_cert_rate:>11.1%} "
        f"{avg_cond_cert_rate:>11.1%}"
    )
    print("-" * 120)
    print("="*80 + "\n")
    
    # Save results to file
    save_comparison_results(
        results_cond, results_uncond,
        test_type="adversarial" if use_adversarial else "clean",
        attack_type=attack_type if use_adversarial else None,
        epsilon=epsilon if use_adversarial else None,
        sigma=sigma,
        n_samples=n_samples,
        alpha=alpha,
        num_test_samples=num_samples
    )
    
    return results_cond, results_uncond


def main():
    parser = argparse.ArgumentParser(description='Test STL Randomized Smoothing')
    parser.add_argument('--model', type=str, default='training_runs/run_20260105_180917/best_model.pth',
                       help='Path to conditioned diffusion checkpoint (if comparing two)')
    parser.add_argument('--model-unconditioned', type=str, default='training_runs/run_20251222_121743/best_model.pth',
                       help='Path to unconditioned diffusion checkpoint for comparison')
    parser.add_argument('--sigma', type=float, default=0.1,
                       help='Noise standard deviation')
    parser.add_argument('--n_samples', type=int, default=100,
                       help='Number of Monte Carlo samples')
    parser.add_argument('--alpha', type=float, default=0.001,
                       help='Confidence level (1-alpha)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for processing')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu or cuda)')
    parser.add_argument('--force-conditioned', type=str, default=None, choices=['true', 'false'],
                       help='Override model conditioning (default: auto-detect from config.json)')
    parser.add_argument('--telex-optmethod', type=str, default='gradient',
                       help='TeLEx optimization method (default: gradient)')
    parser.add_argument('--attack-type', type=str, default='gaussian',
                       choices=['gaussian', 'uniform', 'false_high', 'false_low', 'temporal', 'targeted_feature'],
                       help='Attack type for adversarial testing (default: gaussian)')
    parser.add_argument('--epsilon', type=float, default=1.5,
                       help='Attack epsilon/magnitude (default: 1.5)')
    parser.add_argument('--use-adversarial', type=str, default='true', choices=['true', 'false'],
                       help='Whether to run adversarial test in two-model comparison (default: true)')
    
    args = parser.parse_args()
    if not os.path.exists(args.model):
        print(f"Model not found: {args.model}")
        print("Please train the model first using: python net_training/train_net_model.py")
        return
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    comparing_two_models = args.model_unconditioned is not None
    
    if comparing_two_models:
        if not os.path.exists(args.model_unconditioned):
            print(f"Unconditioned model not found: {args.model_unconditioned}")
            return

        model_conditioned = load_model(args.model, device, force_conditioned=args.force_conditioned)
        model_unconditioned = load_model(args.model_unconditioned, device, force_conditioned=args.force_conditioned)
    else:
        model = load_model(args.model, device, force_conditioned=args.force_conditioned)
        model_conditioned = None
        model_unconditioned = None
    
    X_test, Y_test = load_test_data(device)
    
    feature_names = ["total_flow", "avg_occupancy", "avg_speed", 
                    "aq_NO2", "aq_NOx", "aq_O3", "aq_PM25"]
    
    Y_test_np = Y_test.cpu().numpy() if isinstance(Y_test, torch.Tensor) else Y_test
    gt_stl_formulas, _, _ = create_meaningful_stl_formulas(
        Y_test_np, 
        feature_names,
        horizon=Y_test.shape[2]  # Number of forecast timesteps
    )
    
    if comparing_two_models:
        use_adversarial = (args.use_adversarial == 'true')
        print("\n" + "="*80)
        print(f"TEST: {'ADVERSARIAL' if use_adversarial else 'CLEAN'}")
        print("="*80)
        compare_two_models(
            model_conditioned=model_conditioned,
            model_unconditioned=model_unconditioned,
            X_test=X_test,
            Y_test=Y_test,
            sigma=args.sigma,
            n_samples=args.n_samples,
            alpha=args.alpha,
            device=device,
            feature_names=feature_names,
            batch_size=args.batch_size,
            use_adversarial=use_adversarial,
            attack_type=args.attack_type,
            epsilon=args.epsilon
        )
        
        print("\nTwo-model comparison complete!")
        return
    
    print("="*80)
    print("INITIALIZING STL RANDOMIZED SMOOTHER")
    print("="*80)
    print(f"Sigma:           {args.sigma} (Gaussian noise std for smoothing)")
    print(f"N_samples:       {args.n_samples} (Monte Carlo samples per certification)")
    print(f"Alpha:           {args.alpha} (Confidence level: {1-args.alpha:.4f})")
    print(f"STL Formulas:    {len(gt_stl_formulas)} formulas (GT-based, {Y_test.shape[2]} timesteps)")
    print(f"Formula Types:   Always (G) and Eventually (F) over forecast horizon")
    print("="*80 + "\n")
    
    smoother = STLRandomizedSmoother(
        model=model,
        stl_formulas=gt_stl_formulas,
        sigma=args.sigma,
        n_samples=args.n_samples,
        device=device,
        feature_names=feature_names
    )
    
    # Batch certification test
    test_batch_certification(smoother, X_test, args.alpha, args.batch_size)
    
    print("\n" + "="*80)
    print("Done.")
    print("="*80)


if __name__ == '__main__':
    main()
