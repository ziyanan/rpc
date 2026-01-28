import argparse
import os

from .sumo_collect import collect_sumo_data
from .sim_loop import run_closed_loop
from .dataset import build_windows_from_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="NYC SUMO simulation pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Collect
    collect_parser = subparsers.add_parser("collect", help="Run SUMO and collect detector data")
    collect_parser.add_argument("--sumocfg", required=True, help="Path to SUMO .sumocfg")
    collect_parser.add_argument("--out-dir", default="nyc_simulation/data", help="Output directory")
    collect_parser.add_argument("--detectors", default="auto", help="Detector IDs comma-separated or 'auto'")
    collect_parser.add_argument("--dt", type=int, default=60, help="Aggregation interval in seconds")
    collect_parser.add_argument("--seed", type=int, default=42, help="SUMO random seed")
    collect_parser.add_argument("--sumo-home", default=None, help="SUMO installation path")
    collect_parser.add_argument("--end-time", type=int, default=None, help="Override SUMO end time (seconds)")

    # Build windows
    windows_parser = subparsers.add_parser("build", help="Build sliding windows from collected CSV")
    windows_parser.add_argument("--csv", required=True, help="Collected CSV path")
    windows_parser.add_argument("--out-dir", default="nyc_simulation/data", help="Output directory")
    windows_parser.add_argument("--seq-len", type=int, default=6, help="History length")
    windows_parser.add_argument("--horizon", type=int, default=3, help="Prediction horizon")
    windows_parser.add_argument("--stride", type=int, default=1, help="Window stride")

    # Train using existing pipeline
    train_parser = subparsers.add_parser("train", help="Train LogicGuidedDiffusionForecast on SUMO dataset")
    train_parser.add_argument("--data-dir", required=True, help="Dataset directory with X_clean_*.npy")
    train_parser.add_argument("--use-conditioned", default="true", choices=["true", "false"])
    train_parser.add_argument("--use-adv-training", default="true", choices=["true", "false"])
    train_parser.add_argument("--attack-type", default="uniform")
    train_parser.add_argument("--attack-prob", default="0.5")
    train_parser.add_argument("--epochs", default="50")

    # Simulate
    sim_parser = subparsers.add_parser("simulate", help="Closed-loop SUMO simulation")
    sim_parser.add_argument("--sumocfg", required=True, help="Path to SUMO .sumocfg")
    sim_parser.add_argument("--model-path", required=True, help="Path to trained model (logic-guided)")
    sim_parser.add_argument("--baseline-path", required=True, help="Path to baseline model")
    sim_parser.add_argument("--detectors", default="auto", help="Detector IDs comma-separated or 'auto'")
    sim_parser.add_argument("--dt", type=int, default=60, help="Control timestep in seconds")
    sim_parser.add_argument("--attack-type", default="none", help="Attack type or 'none'")
    sim_parser.add_argument("--epsilon", type=float, default=1.0, help="Attack epsilon")
    sim_parser.add_argument("--sigma", type=float, default=0.1, help="Smoothing sigma")
    sim_parser.add_argument("--n0", type=int, default=100, help="N0 samples for smoothing")
    sim_parser.add_argument("--n", type=int, default=1000, help="N samples for smoothing")
    sim_parser.add_argument("--alpha", type=float, default=0.001, help="Certification alpha")
    sim_parser.add_argument("--epsilon-cert", type=float, default=0.08, help="Certification threshold")
    sim_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    sim_parser.add_argument("--out-dir", default="nyc_simulation/outputs", help="Output directory")
    sim_parser.add_argument("--mode", default="logic", choices=["logic", "baseline"],
                            help="Which model to use for prediction")
    sim_parser.add_argument("--sumo-home", default=None, help="SUMO installation path")
    sim_parser.add_argument("--end-time", type=int, default=None, help="Override SUMO end time (seconds)")

    args = parser.parse_args()

    if args.command == "collect":
        detector_ids = None if args.detectors == "auto" else args.detectors.split(",")
        os.makedirs(args.out_dir, exist_ok=True)
        collect_sumo_data(
            sumocfg_path=args.sumocfg,
            out_dir=args.out_dir,
            detector_ids=detector_ids,
            dt_seconds=args.dt,
            seed=args.seed,
            sumo_home=args.sumo_home,
            end_time=args.end_time,
        )
    elif args.command == "build":
        os.makedirs(args.out_dir, exist_ok=True)
        build_windows_from_csv(
            csv_path=args.csv,
            out_dir=args.out_dir,
            seq_len=args.seq_len,
            horizon=args.horizon,
            stride=args.stride,
        )
    elif args.command == "train":
        import subprocess
        env = os.environ.copy()
        env["DATASET_DIR"] = args.data_dir
        env["USE_CONDITIONED"] = args.use_conditioned
        env["USE_ADV_TRAINING"] = args.use_adv_training
        env["ATTACK_TYPE"] = args.attack_type
        env["ATTACK_PROB"] = args.attack_prob
        env["EPOCHS"] = args.epochs
        subprocess.run(["python3.9", "net_training/train_net_model.py"], check=True, env=env)
    elif args.command == "simulate":
        detector_ids = None if args.detectors == "auto" else args.detectors.split(",")
        os.makedirs(args.out_dir, exist_ok=True)
        run_closed_loop(
            sumocfg_path=args.sumocfg,
            model_path=args.model_path,
            baseline_path=args.baseline_path,
            out_dir=args.out_dir,
            detector_ids=detector_ids,
            dt_seconds=args.dt,
            attack_type=args.attack_type,
            epsilon=args.epsilon,
            sigma=args.sigma,
            n0=args.n0,
            n_samples=args.n,
            alpha=args.alpha,
            epsilon_cert=args.epsilon_cert,
            seed=args.seed,
            mode=args.mode,
            sumo_home=args.sumo_home,
            end_time=args.end_time,
        )


if __name__ == "__main__":
    main()
