import json
import os
from collections import Counter
from typing import List, Optional

import numpy as np
import torch

from logics.stl_attack_repair import AttackGenerator, STLGuidedRepairer, TeLExLearner
from net.net import LogicGuidedDiffusionForecast
from net.baselines.transformer_baseline import TemporalTransformerBaseline
from smoothing.stl_smoother import STLRandomizedSmoother
from smoothing.stl_utils import compute_stl_robustness
from smoothing.test_smoothing import create_meaningful_stl_formulas

from .control import FixedTimeController, apply_traffic_light_actions, select_top_tls_ids, get_controlled_edges
from .metrics import compute_step_metrics, summarize_metrics
from .plotting import plot_timeseries
from .sumo_collect import (
    _get_net_path,
    _init_traci,
    _load_edge_lanes,
    _scan_active_edges,
    _select_lanes_from_edges,
    _write_actuated_net,
    _write_augmented_sumocfg,
    _write_detectors_file,
)


FEATURE_NAMES = ["total_flow", "avg_speed", "avg_occupancy"]
SCALER_FALLBACK = {"mean_": [0.0, 0.0, 0.0], "scale_": [1.0, 1.0, 1.0]}
USE_AUTO_THRESHOLD = False
AUTO_THRESHOLD_PERCENTILE = 80


def _load_model(model_path: str, device: str = "cpu") -> LogicGuidedDiffusionForecast:
    checkpoint = torch.load(model_path, map_location=device)
    model_dir = os.path.dirname(model_path)
    config_path = os.path.join(model_dir, "config.json")
    model_cfg = {}
    model_type = None
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = json.load(f)
            model_cfg = cfg.get("model", {})
            model_type = model_cfg.get("model_type")

    if model_type == "transformer_baseline":
        model = TemporalTransformerBaseline(
            in_channels=checkpoint.get("in_channels", len(FEATURE_NAMES)),
            d_model=model_cfg.get("d_model", 128),
            n_heads=model_cfg.get("n_heads", 4),
            n_encoder_layers=model_cfg.get("n_encoder_layers", 3),
            n_decoder_layers=model_cfg.get("n_decoder_layers", 3),
            dim_feedforward=model_cfg.get("dim_feedforward", 512),
            forecast_horizon=checkpoint.get("forecast_horizon", model_cfg.get("forecast_horizon", 3)),
            dropout=model_cfg.get("dropout", 0.1),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        return model

    model = LogicGuidedDiffusionForecast(
        in_channels=checkpoint.get("in_channels", len(FEATURE_NAMES)),
        diffusion_hidden=model_cfg.get("diffusion_hidden", 32),
        diffusion_levels=model_cfg.get("diffusion_levels", 2),
        seq_d_model=model_cfg.get("seq_d_model", 64),
        seq_n_heads=model_cfg.get("seq_n_heads", 2),
        seq_n_layers=model_cfg.get("seq_n_layers", 2),
        forecast_horizon=checkpoint.get("forecast_horizon", 3),
        num_diffusion_steps=model_cfg.get("num_diffusion_steps", 10),
        use_conditioned_diffusion=model_cfg.get("use_conditioned_diffusion", True),
        seq_model_type=model_cfg.get("seq_model_type", "seq2seq"),
        seq_dim_feedforward=model_cfg.get("seq_dim_feedforward", 256),
    )
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as exc:
        msg = str(exc)
        if "condition_weight" in msg:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            raise
    model.to(device)
    model.eval()
    return model


def _load_scaler_stats(model_path: str) -> dict:
    model_dir = os.path.dirname(model_path)
    candidates = [
        os.path.join(model_dir, "stats.json"),
        os.path.join(model_dir, "timeseries_meta.json"),
        os.path.join("nyc_simulation", "data", "timeseries_meta.json"),
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        with open(path, "r") as f:
            data = json.load(f)
        if "scaler" in data:
            return data["scaler"]
        if "mean_" in data and "scale_" in data:
            return {"mean_": data["mean_"], "scale_": data["scale_"]}
    return SCALER_FALLBACK


def _normalize_series(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (x - mean[:, None]) / scale[:, None]


def _inverse_transform(y: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return y * scale[None, :] + mean[None, :]


def _build_detectors(
    sumocfg_path: str,
    out_dir: str,
    detector_ids: Optional[List[str]],
    end_time: Optional[int],
    sumo_home: Optional[str],
    seed: int,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    sumo_dir = os.path.dirname(sumocfg_path)
    detectors_path = os.path.join(out_dir, "additional_detectors.xml")
    augmented_cfg = os.path.join(out_dir, "augmented.sumocfg")

    if detector_ids is None:
        net_path = _get_net_path(sumocfg_path)
        edge_lanes = _load_edge_lanes(net_path)
        active_edges = _scan_active_edges(sumocfg_path, seed, sumo_home)
        detector_ids = _select_lanes_from_edges(edge_lanes, active_edges, max_det=4)
        if not detector_ids:
            all_lanes = []
            for lanes in edge_lanes.values():
                all_lanes.extend(lanes)
            detector_ids = all_lanes[:4]
        if not detector_ids:
            raise RuntimeError("No lanes found in network; cannot create detectors.")

    _write_detectors_file(detector_ids, detectors_path)
    net_path = _get_net_path(sumocfg_path)
    actuated_net = os.path.join(out_dir, "actuated.net.xml")
    _write_actuated_net(net_path, actuated_net)
    _, cfg_path = _write_augmented_sumocfg(
        sumocfg_path, detectors_path, augmented_cfg, end_time, net_override_path=actuated_net
    )
    return cfg_path


def run_closed_loop(
    sumocfg_path: str,
    model_path: str,
    baseline_path: str,
    out_dir: str,
    detector_ids: Optional[List[str]] = None,
    dt_seconds: int = 60,
    attack_type: str = "none",
    epsilon: float = 1.0,
    sigma: float = 0.1,
    n0: int = 100,
    n_samples: int = 1000,
    alpha: float = 0.001,
    epsilon_cert: float = 0.08,
    seed: int = 42,
    mode: str = "logic",
    sumo_home: Optional[str] = None,
    end_time: Optional[int] = None,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    cfg_path = _build_detectors(sumocfg_path, out_dir, detector_ids, end_time, sumo_home, seed)
    traci = _init_traci(cfg_path, seed, sumo_home)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    np.random.seed(seed)
    torch.manual_seed(seed)

    logic_model = _load_model(model_path, device=device)
    baseline_model = _load_model(baseline_path, device=device)
    scaler = _load_scaler_stats(model_path)
    scaler_mean = np.asarray(scaler.get("mean_", SCALER_FALLBACK["mean_"]), dtype=np.float32)
    scaler_scale = np.asarray(scaler.get("scale_", SCALER_FALLBACK["scale_"]), dtype=np.float32)
    scaler_scale = np.where(scaler_scale == 0, 1.0, scaler_scale)

    controller = FixedTimeController(green_extension=5.0, congestion_threshold=0.03)

    tls_ids = select_top_tls_ids(traci, top_n=10)
    controlled_edges = get_controlled_edges(traci, tls_ids)
    print(f"Selected {len(tls_ids)} traffic lights for control")
    print(f"Monitoring {len(controlled_edges)} controlled edges")

    history = []
    rows = []
    timestep = 0
    last_decision_time = None
    decision_count = 0

    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            timestep += 1
            sim_time = traci.simulation.getTime()
            if end_time is not None and sim_time >= end_time:
                break

            det_ids = traci.lanearea.getIDList()
            flows = [traci.lanearea.getLastStepVehicleNumber(d) for d in det_ids]
            speeds = [traci.lanearea.getLastStepMeanSpeed(d) for d in det_ids]
            occs = [traci.lanearea.getLastStepOccupancy(d) for d in det_ids]

            if last_decision_time is None:
                last_decision_time = sim_time
            if sim_time - last_decision_time >= dt_seconds:
                last_decision_time = sim_time
                state = np.array([np.mean(flows), np.mean(speeds), np.mean(occs)], dtype=np.float32)
                history.append(state)
                if len(history) < 6:
                    continue
                history = history[-6:]

                x_history_raw = np.stack(history, axis=0)
                x_input_raw = x_history_raw.copy()[None, :, :].transpose(0, 2, 1)
                x_input_norm = _normalize_series(x_input_raw, scaler_mean, scaler_scale)

                if attack_type != "none":
                    attacker = AttackGenerator(attack_type=attack_type, epsilon=epsilon, seed=seed)
                    x_attacked = attacker.attack(x_input_norm.copy())
                else:
                    x_attacked = x_input_norm.copy()
                x_clean = x_input_norm.copy()

                use_logic = (mode == "logic")
                model = logic_model if use_logic else baseline_model
                x_condition = None
                x_cert_center = x_clean.copy()

                if use_logic:
                    learner = TeLExLearner(FEATURE_NAMES, timesteps=x_input_norm.shape[2])
                    learned_bounds = learner.learn_bounds_simple(x_input_norm, FEATURE_NAMES, verbose=False)
                    repairer = STLGuidedRepairer(FEATURE_NAMES, learned_bounds=learned_bounds)
                    x_repaired_np = repairer.repair(x_attacked, method="comprehensive", verbose=False)
                    x_condition = torch.FloatTensor(x_repaired_np).to(device)
                    x_cert_center = x_repaired_np

                x_tensor = torch.FloatTensor(x_attacked).to(device)
                with torch.no_grad():
                    if use_logic:
                        y_pred = model(x_tensor, x_condition=x_condition, add_noise=False)
                    else:
                        y_pred = model(x_tensor, x_condition=None, add_noise=False)

                y_pred_norm = y_pred.cpu().numpy()[0].T
                y_pred_phys = _inverse_transform(y_pred_norm, scaler_mean, scaler_scale)

                y_context = x_history_raw[-3:].T[None, :, :]
                formulas, _, _ = create_meaningful_stl_formulas(
                    y_context, FEATURE_NAMES, horizon=3, verbose=False
                )
                formulas = {
                    name: formula
                    for name, formula in formulas.items()
                    if "_trend" not in name
                }

                rho_values = []
                for formula in formulas.values():
                    rho = compute_stl_robustness(y_pred_phys, formula, FEATURE_NAMES)
                    rho_values.append(rho)
                rho_min = float(np.min(rho_values)) if rho_values else 0.0

                if n_samples < 500:
                    print(f"Warning: n_samples={n_samples} may be unstable for certification.")
                smoother = STLRandomizedSmoother(
                    model=model,
                    stl_formulas=formulas,
                    sigma=sigma,
                    n_samples=n_samples,
                    device=device,
                    feature_names=FEATURE_NAMES,
                )

                x_cert_tensor = torch.FloatTensor(x_cert_center).to(device)

                temp_n = smoother.n_samples
                smoother.n_samples = n0
                pre = smoother.predict_smooth(x_cert_tensor, x_condition=x_condition, return_all=True)
                smoother.n_samples = temp_n
                cert = smoother.certify(
                    x_cert_tensor,
                    alpha=alpha,
                    x_condition=x_condition,
                    n_samples=n_samples,
                    verbose=False,
                )

                gate_formula_names = [
                    name for name in formulas.keys()
                    if (("reasonable_range" in name) or ("eventually" in name))
                    and not ("window" in name and "lower" in name and "eventually" not in name)
                ]
                if not gate_formula_names:
                    gate_formula_names = list(formulas.keys())

                radii = [cert[name][0]["certified_radius"] for name in gate_formula_names if name in cert]

                if cert:
                    pre_p_hat = []
                    for formula_name in gate_formula_names:
                        classifications = np.stack(pre["all_classifications"][formula_name])
                        pre_p_hat.append(float(np.mean(classifications[:, 0])))
                    p_hat = min(pre_p_hat) if pre_p_hat else 0.0
                else:
                    p_hat = 0.0
                p_lower = min(
                    cert[name][0]["diagnostics"]["p_lower"]
                    for name in gate_formula_names
                    if name in cert
                ) if cert else 0.0
                if radii:
                    radius_perc25 = float(np.percentile(radii, 25))
                else:
                    print("ERROR: No radii found")
                    radius_perc25 = 0.0
                is_certified = radius_perc25 >= epsilon_cert

                if USE_AUTO_THRESHOLD:
                    recent_occ = [h[2] for h in history]
                    if recent_occ:
                        controller.congestion_threshold = float(
                            np.percentile(recent_occ, AUTO_THRESHOLD_PERCENTILE)
                        )
                else:
                    controller.congestion_threshold = 3.0 if float(state[2]) > 1.5 else 0.3

                pred_occ_raw = float(y_pred_norm[-1, 2])
                pred_occ_physical = float(y_pred_phys[-1, 2])
                extend_green = controller.select_action(
                    predicted_occupancy=pred_occ_physical,
                    is_certified=is_certified,
                )
                use_fallback = not is_certified
                if use_fallback:
                    action = "fallback"
                else:
                    action = apply_traffic_light_actions(traci, tls_ids, extend_green, 5.0)
                decision_count += 1
                print(
                    f"[t={sim_time:.0f}] state_occ_raw={float(state[2]):.3f} "
                    f"pred_occ_raw={pred_occ_raw:.3f} pred_occ_physical={pred_occ_physical:.3f} "
                    f"cert={int(is_certified)} r_perc25={radius_perc25:.4f} p_lower={p_lower:.3f} "
                    f"extend={int(extend_green)}"
                )

                step_metrics = compute_step_metrics(traci, controlled_edges=controlled_edges)
                rows.append(
                    {
                        "time": timestep,
                        "total_flow": float(state[0]),
                        "avg_speed": float(state[1]),
                        "avg_occupancy": float(state[2]),
                        "rho": rho_min,
                        "certified_radius": radius_perc25,
                        "p_hat": p_hat,
                        "p_lower": p_lower,
                        "control_action": action,
                        "fallback": 1.0 if use_fallback else 0.0,
                        **step_metrics,
                    }
                )
    finally:
        traci.close()

    if rows:
        csv_path = os.path.join(out_dir, "closed_loop_log.csv")
        with open(csv_path, "w") as f:
            headers = list(rows[0].keys())
            f.write(",".join(headers) + "\n")
            for r in rows:
                f.write(",".join(str(r[h]) for h in headers) + "\n")

        metrics_rows = rows[-600:] if len(rows) > 600 else rows
        summary = summarize_metrics(metrics_rows, epsilon_cert)
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        plot_timeseries(metrics_rows, out_dir, epsilon_cert)
        action_counts = Counter(r["control_action"] for r in metrics_rows)
        print(f"Control action counts: {dict(action_counts)}")
