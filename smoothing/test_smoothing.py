#!/usr/bin/env python3

import numpy as np
from analysis.stl_formulas import Always, Eventually, Atomic, STLAnd, STLOr, STLNot


def _safe_percentile(arr, q, default):
    arr = np.asarray(arr).reshape(-1)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float(default)
    return float(np.percentile(arr, q))


def create_meaningful_stl_formulas(Y_data, feature_names, horizon=6, verbose=False, X_data=None):
    feature_idx_map = {name: idx for idx, name in enumerate(feature_names)}
    num_features = len(feature_names)
    formulas = {}

    # Normalize shapes
    if Y_data.ndim == 2:
        Y_data = Y_data[np.newaxis, :]

    data_source = X_data if X_data is not None else Y_data  # (N, F, T)

    # ----------------------
    # Type 2: Reasonable bounds
    # ----------------------
    for feature_name in feature_names:
        if feature_name not in feature_idx_map:
            continue
        idx = feature_idx_map[feature_name]
        feature_data = data_source[:, idx, :]
        gt_min = float(np.min(feature_data))
        gt_max = float(np.max(feature_data))
        width = gt_max - gt_min
        if width < 1e-6:
            relaxed_min = gt_min - 1e-3
            relaxed_max = gt_max + 1e-3
        else:
            relax = 0.05 * width
            relaxed_min = gt_min - relax
            relaxed_max = gt_max + relax
        w_pos = np.zeros(num_features); w_pos[idx] = 1.0
        w_neg = np.zeros(num_features); w_neg[idx] = -1.0
        formulas[f"{feature_name}_reasonable_range_lower"] = Always(
            Atomic(w=w_pos, c=relaxed_min, feature_idx=idx, relop=">="), t_start=0, t_end=horizon-1
        )
        formulas[f"{feature_name}_reasonable_range_upper"] = Always(
            Atomic(w=w_neg, c=-relaxed_max, feature_idx=idx, relop=">="), t_start=0, t_end=horizon-1
        )
        if horizon > 1:
            formulas[f"{feature_name}_reasonable_range_eventually_lower"] = Eventually(
                Atomic(w=w_pos, c=gt_min, feature_idx=idx, relop=">="), t_start=0, t_end=horizon-1
            )
            formulas[f"{feature_name}_reasonable_range_eventually_upper"] = Eventually(
                Atomic(w=w_neg, c=-gt_max, feature_idx=idx, relop=">="), t_start=0, t_end=horizon-1
            )

        # Windowed 3-step bounds over GT horizon (e.g., [0,2], [3,5])
        for start in range(0, horizon, 3):
            end = start + 2
            if end >= horizon:
                break
            # Collect data from GT window
            y_win = Y_data[:, idx, start:end+1]
            # If X_data provided, also include same indices from X_data (if available)
            if X_data is not None and X_data.shape[2] > end:
                x_win = X_data[:, idx, start:end+1]
                window_data = np.concatenate([y_win, x_win], axis=1)
            else:
                window_data = y_win

            win_min = float(np.min(window_data))
            win_max = float(np.max(window_data))
            width_w = win_max - win_min
            if width_w < 1e-6:
                relaxed_min_w = win_min - 1e-3
                relaxed_max_w = win_max + 1e-3
            else:
                relax_w = 0.1 * width_w
                relaxed_min_w = win_min - relax_w
                relaxed_max_w = win_max + relax_w

            formulas[f"{feature_name}_window_{start}_{end}_lower"] = Always(
                Atomic(w=w_pos, c=relaxed_min_w, feature_idx=idx, relop=">="),
                t_start=start, t_end=end
            )
            formulas[f"{feature_name}_window_{start}_{end}_upper"] = Always(
                Atomic(w=w_neg, c=-relaxed_max_w, feature_idx=idx, relop=">="),
                t_start=start, t_end=end
            )
            formulas[f"{feature_name}_window_{start}_{end}_eventually_lower"] = Eventually(
                Atomic(w=w_pos, c=win_min, feature_idx=idx, relop=">="),
                t_start=start, t_end=end
            )
            formulas[f"{feature_name}_window_{start}_{end}_eventually_upper"] = Eventually(
                Atomic(w=w_neg, c=-win_max, feature_idx=idx, relop=">="),
                t_start=start, t_end=end
            )

    # If too short for temporal windows, stop here
    if data_source.shape[2] < 4:
        return formulas, [], []

    # ----------------------
    # Augmented signals (delta, abs_delta)
    # ----------------------
    USE_AUGMENTED_SIGNALS = True
    aug_feature_names = list(feature_names)
    aug_feature_idx_map = dict(feature_idx_map)
    if USE_AUGMENTED_SIGNALS:
        delta_names = [f"delta_{fn}" for fn in feature_names]
        abs_delta_names = [f"abs_delta_{fn}" for fn in feature_names]
        base_F = len(aug_feature_names)
        for k, name in enumerate(delta_names):
            aug_feature_idx_map[name] = base_F + k
            aug_feature_names.append(name)
        base_F2 = len(aug_feature_names)
        for k, name in enumerate(abs_delta_names):
            aug_feature_idx_map[name] = base_F2 + k
            aug_feature_names.append(name)
        deltas = np.diff(data_source, axis=2)
        deltas_padded = np.concatenate([deltas, deltas[:, :, -1:,]], axis=2)
        abs_deltas_padded = np.abs(deltas_padded)
        data_source = np.concatenate([data_source, deltas_padded, abs_deltas_padded], axis=1)
        num_aug_features = len(aug_feature_names)
    else:
        num_aug_features = num_features

    # ----------------------
    # Type 3: temporal formulas (dynamic behavior)
    # ----------------------
    SKIP_SHAPE_LOGICS = True  # set False to re-enable drop/anti-smoothing/peak-trough/temporal-sync

    for feature_name in feature_names:
        idx = feature_idx_map[feature_name]

        if not SKIP_SHAPE_LOGICS:
            # Drop-then-recovery
            all_deltas = np.diff(data_source[:, idx, :], axis=1).reshape(-1)
            d_drop = _safe_percentile(all_deltas[all_deltas < 0], 20, default=-0.1)
            d_rise = _safe_percentile(all_deltas[all_deltas > 0], 80, default=0.1)
            if USE_AUGMENTED_SIGNALS:
                d_idx = aug_feature_idx_map[f"delta_{feature_name}"]
                w_delta = np.zeros(num_aug_features); w_delta[d_idx] = 1.0
                drop_cond = Atomic(w=w_delta, c=d_drop, feature_idx=d_idx, relop="<=")
                rise_cond = Atomic(w=w_delta, c=d_rise, feature_idx=d_idx, relop=">=")
                formulas[f"{feature_name}_drop_recovery"] = Always(
                    STLOr(STLNot(drop_cond), Eventually(rise_cond, t_start=1, t_end=2)),
                    t_start=0, t_end=horizon-3
                )

            # Anti-smoothing
            all_abs = np.abs(np.diff(data_source[:, idx, :], axis=1)).reshape(-1)
            eps = _safe_percentile(all_abs, 30, default=0.01)
            if USE_AUGMENTED_SIGNALS:
                a_idx = aug_feature_idx_map[f"abs_delta_{feature_name}"]
                w_abs = np.zeros(num_aug_features); w_abs[a_idx] = 1.0
                flat_step = Atomic(w=w_abs, c=eps, feature_idx=a_idx, relop="<=")
                flat_window = Always(flat_step, t_start=0, t_end=2)
                formulas[f"{feature_name}_anti_smoothing"] = STLNot(
                    Eventually(flat_window, t_start=0, t_end=horizon-4)
                )

            # Peak-to-trough
            all_values = data_source[:, idx, :].reshape(-1)
            high_thr = _safe_percentile(all_values, 75, default=float(np.max(all_values)))
            low_thr = _safe_percentile(all_values, 25, default=float(np.min(all_values)))
            w_pos = np.zeros(num_features); w_pos[idx] = 1.0
            w_neg = np.zeros(num_features); w_neg[idx] = -1.0
            peak = Atomic(w=w_pos, c=high_thr, feature_idx=idx, relop=">=")
            trough = Atomic(w=w_neg, c=-low_thr, feature_idx=idx, relop=">=")
            formulas[f"{feature_name}_peak_trough"] = Always(
                STLOr(STLNot(peak), Eventually(trough, t_start=1, t_end=2)),
                t_start=0, t_end=horizon-3
            )

        # Trend direction (net increase/decrease from start to end of GT horizon)
        gt_start = float(data_source[:, idx, 0].mean())
        gt_end = float(data_source[:, idx, horizon-1].mean())
        net_trend = gt_end - gt_start
        trend_tol = 0.0  # simple comparison
        if USE_AUGMENTED_SIGNALS:
            d_idx = aug_feature_idx_map[f"delta_{feature_name}"]
            w_delta = np.zeros(num_aug_features); w_delta[d_idx] = 1.0
            if net_trend >= 0:
                # Expect non-negative deltas somewhere across horizon
                formulas[f"{feature_name}_trend_up"] = Eventually(
                    Atomic(w=w_delta, c=trend_tol, feature_idx=d_idx, relop=">="),
                    t_start=0, t_end=horizon-2
                )
            else:
                formulas[f"{feature_name}_trend_down"] = Eventually(
                    Atomic(w=w_delta, c=-trend_tol, feature_idx=d_idx, relop="<="),
                    t_start=0, t_end=horizon-2
                )

            if horizon >= 4:
                gt_mid = float(data_source[:, idx, 3].mean())
                net_trend_0_3 = gt_mid - gt_start
                if net_trend_0_3 >= 0:
                    formulas[f"{feature_name}_trend_0_3_up"] = Eventually(
                        Atomic(w=w_delta, c=trend_tol, feature_idx=d_idx, relop=">="),
                        t_start=0, t_end=min(2, horizon-2)
                    )
                else:
                    formulas[f"{feature_name}_trend_0_3_down"] = Eventually(
                        Atomic(w=w_delta, c=-trend_tol, feature_idx=d_idx, relop="<="),
                        t_start=0, t_end=min(2, horizon-2)
                    )

            if horizon >= 6:
                gt_t3 = float(data_source[:, idx, 3].mean())
                gt_t5 = float(data_source[:, idx, 5].mean())
                net_trend_3_5 = gt_t5 - gt_t3
                if net_trend_3_5 >= 0:
                    formulas[f"{feature_name}_trend_3_5_up"] = Eventually(
                        Atomic(w=w_delta, c=trend_tol, feature_idx=d_idx, relop=">="),
                        t_start=3, t_end=min(4, horizon-2)
                    )
                else:
                    formulas[f"{feature_name}_trend_3_5_down"] = Eventually(
                        Atomic(w=w_delta, c=-trend_tol, feature_idx=d_idx, relop="<="),
                        t_start=3, t_end=min(4, horizon-2)
                    )

    if not SKIP_SHAPE_LOGICS and 'total_flow' in feature_idx_map and 'avg_speed' in feature_idx_map and horizon > 3:
        flow_idx = feature_idx_map['total_flow']
        speed_idx = feature_idx_map['avg_speed']
        flow_deltas = np.diff(data_source[:, flow_idx, :], axis=1).reshape(-1)
        speed_deltas = np.diff(data_source[:, speed_idx, :], axis=1).reshape(-1)
        d_drop = _safe_percentile(flow_deltas[flow_deltas < 0], 20, default=-0.1)
        ds_drop = _safe_percentile(speed_deltas[speed_deltas < 0], 20, default=-0.1)
        if USE_AUGMENTED_SIGNALS:
            df_idx = aug_feature_idx_map["delta_total_flow"]
            ds_idx = aug_feature_idx_map["delta_avg_speed"]
            w_df = np.zeros(num_aug_features); w_df[df_idx] = 1.0
            w_ds = np.zeros(num_aug_features); w_ds[ds_idx] = 1.0
            flow_drop = Atomic(w=w_df, c=d_drop, feature_idx=df_idx, relop="<=")
            speed_resp = Atomic(w=w_ds, c=ds_drop, feature_idx=ds_idx, relop="<=")
            formulas["flow_speed_temporal_sync"] = Always(
                STLOr(STLNot(flow_drop), Eventually(speed_resp, t_start=0, t_end=1)),
                t_start=0, t_end=horizon-3
            )

    return formulas, [], []
