import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from net.net import LogicGuidedDiffusionForecast
from logics.stl_attack_repair import AttackGenerator, STLGuidedRepairer, TeLExLearner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _set_requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag


def _make_optimizer(params, lr: float) -> optim.Optimizer:
    return optim.AdamW(params, lr=lr, weight_decay=1e-4)


def _make_warmup_cosine_scheduler(optimizer: optim.Optimizer, total_steps: int, warmup_frac: float = 0.075):
    warmup_steps = max(1, int(total_steps * warmup_frac))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
        return float(0.5 * (1 + np.cos(np.pi * progress)))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _load_dataset(ds_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    X_train = np.load(os.path.join(ds_dir, "X_clean_train.npy"))
    X_val = np.load(os.path.join(ds_dir, "X_clean_val.npy"))
    Y_train = np.load(os.path.join(ds_dir, "Y_train.npy"))
    Y_val = np.load(os.path.join(ds_dir, "Y_val.npy"))

    with open(os.path.join(ds_dir, "timeseries_meta.json")) as f:
        meta = json.load(f)

    max_train = int(os.environ.get("MAX_TRAIN_SAMPLES", "50000"))
    max_val = int(os.environ.get("MAX_VAL_SAMPLES", "15000"))
    if len(X_train) > max_train:
        X_train = X_train[:max_train]
        Y_train = Y_train[:max_train]
    if len(X_val) > max_val:
        X_val = X_val[:max_val]
        Y_val = Y_val[:max_val]

    X_train_t = np.transpose(X_train, (0, 2, 1))
    X_val_t = np.transpose(X_val, (0, 2, 1))
    Y_train_t = np.transpose(Y_train, (0, 2, 1))
    Y_val_t = np.transpose(Y_val, (0, 2, 1))

    return X_train_t, Y_train_t, X_val_t, Y_val_t, meta


class StagedTrainer:
    def __init__(
        self,
        model: LogicGuidedDiffusionForecast,
        device: torch.device,
        feature_names,
        use_conditioned_diffusion: bool,
        adv_epsilon: float,
    ):
        self.model = model.to(device)
        self.device = device
        self.feature_names = feature_names
        self.use_conditioned_diffusion = use_conditioned_diffusion
        self.criterion = nn.MSELoss()

        self.attack_types = ["gaussian", "uniform", "false_high", "false_low", "temporal"]
        self.attackers = {
            attack_type: AttackGenerator(attack_type=attack_type, epsilon=adv_epsilon, seed=None)
            for attack_type in self.attack_types
        }
        self.attack_weights = None
        self.attack_distribution_config = None

    def configure_attack_distribution(self, attack_type_config: str) -> None:
        attack_type_config = attack_type_config.strip().lower()
        if attack_type_config == "uniform":
            self.attack_weights = None
            self.attack_distribution_config = "uniform (20% each type)"
            return

        if attack_type_config in self.attack_types:
            self.attack_weights = {attack_type_config: 1.0}
            self.attack_distribution_config = f"100% {attack_type_config}"
            return

        if ":" not in attack_type_config:
            raise ValueError(
                f"Invalid ATTACK_TYPE={attack_type_config}. Use 'uniform', one of {self.attack_types}, "
                "or weighted format 'type1:w1,type2:w2'."
            )

        pairs = attack_type_config.split(",")
        weights = {}
        total = 0.0
        for pair in pairs:
            atype, w = pair.split(":")
            atype = atype.strip()
            if atype not in self.attack_types:
                raise ValueError(f"Unknown attack type: {atype}. Must be one of {self.attack_types}")
            wv = float(w.strip())
            weights[atype] = wv
            total += wv
        if not np.isclose(total, 1.0, atol=1e-3):
            raise ValueError(f"Weights must sum to 1.0, got {total}")
        self.attack_weights = weights
        self.attack_distribution_config = ", ".join([f"{k}:{v:.0%}" for k, v in weights.items()])

    def _sample_attack_type(self) -> str:
        if self.attack_weights is None:
            return str(np.random.choice(self.attack_types))
        types = list(self.attack_weights.keys())
        weights = list(self.attack_weights.values())
        return str(np.random.choice(types, p=weights))

    def _maybe_attack_and_condition(
        self, batch_x: torch.Tensor, use_adversarial_training: bool, attack_prob: float
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if (not use_adversarial_training) or (np.random.random() >= attack_prob):
            return batch_x, None

        batch_x_np = batch_x.detach().cpu().numpy()
        attack_type = self._sample_attack_type()
        attacker = self.attackers[attack_type]
        x_attacked_np = attacker.attack(batch_x_np)

        if self.use_conditioned_diffusion:
            learner = TeLExLearner(self.feature_names, timesteps=batch_x_np.shape[2])
            learned_bounds = learner.learn_bounds_simple(batch_x_np, self.feature_names, verbose=False)
            repairer = STLGuidedRepairer(self.feature_names, learned_bounds=learned_bounds)
            x_repaired_np = repairer.repair(x_attacked_np, method="comprehensive", verbose=False)
            x_input = torch.FloatTensor(x_attacked_np).to(self.device)
            x_condition = torch.FloatTensor(x_repaired_np).to(self.device)
            return x_input, x_condition

        x_input = torch.FloatTensor(x_attacked_np).to(self.device)
        return x_input, None

    def stage1_train_diffusion(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        save_dir: str,
        epochs: int,
        batch_size: int,
        lr: float,
        use_adversarial_training: bool,
        attack_prob: float,
        patience: int,
    ) -> Dict[str, Any]:
        os.makedirs(save_dir, exist_ok=True)

        _set_requires_grad(self.model.diffusion, True)
        _set_requires_grad(self.model.seq2seq, False)
        self.model.condition_weight.requires_grad = False

        optimizer = _make_optimizer(self.model.diffusion.parameters(), lr=lr)

        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train)),
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_val)),
            batch_size=batch_size,
            shuffle=False,
        )

        total_steps = epochs * len(train_loader)
        scheduler = _make_warmup_cosine_scheduler(optimizer, total_steps=total_steps, warmup_frac=0.075)

        best_val = float("inf")
        best_epoch = 0
        patience_counter = 0
        train_losses = []
        val_losses = []
        global_step = 0

        best_ckpt = os.path.join(save_dir, "stage1_best.pth")
        final_ckpt = os.path.join(save_dir, "stage1_final.pth")

        for epoch in range(epochs):
            self.model.train()
            running = 0.0
            pbar = tqdm(train_loader, desc=f"Stage1 {epoch+1}/{epochs} [train]", leave=False)
            for (batch_x,) in pbar:
                batch_x = batch_x.to(self.device)
                x_input, x_condition = self._maybe_attack_and_condition(batch_x, use_adversarial_training, attack_prob)

                optimizer.zero_grad()
                if self.use_conditioned_diffusion and x_condition is not None:
                    _, x_recon = self.model.diffusion(x_input, add_noise=True, x_logic=x_condition)
                else:
                    _, x_recon = self.model.diffusion(x_input, add_noise=True, x_logic=None)

                loss = self.criterion(x_recon, x_input)
                loss.backward()
                optimizer.step()
                scheduler.step()
                global_step += 1
                running += float(loss.item())
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

            train_loss = running / max(1, len(train_loader))
            train_losses.append(train_loss)

            self.model.eval()
            v = 0.0
            with torch.no_grad():
                vpbar = tqdm(val_loader, desc=f"Stage1 {epoch+1}/{epochs} [val]", leave=False)
                for (batch_x,) in vpbar:
                    batch_x = batch_x.to(self.device)
                    if self.use_conditioned_diffusion:
                        x_logic = batch_x
                        _, x_recon = self.model.diffusion(batch_x, add_noise=False, x_logic=x_logic)
                    else:
                        _, x_recon = self.model.diffusion(batch_x, add_noise=False, x_logic=None)
                    loss = self.criterion(x_recon, batch_x)
                    v += float(loss.item())
                    vpbar.set_postfix({"val_loss": f"{loss.item():.4f}"})

            val_loss = v / max(1, len(val_loader))
            val_losses.append(val_loss)

            logger.info(
                f"[Stage1] epoch {epoch+1}/{epochs} train={train_loss:.6f} val={val_loss:.6f} lr={scheduler.get_last_lr()[0]:.2e}"
            )

            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch + 1
                patience_counter = 0
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "stage": 1,
                        "epoch": epoch + 1,
                        "best_val_loss": best_val,
                    },
                    best_ckpt,
                )
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"[Stage1] early stop at epoch {epoch+1} (best epoch {best_epoch})")
                    break

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "stage": 1,
                "epoch": len(train_losses),
                "best_val_loss": best_val,
            },
            final_ckpt,
        )

        return {
            "stage1": {
                "best_epoch": best_epoch,
                "best_val_loss": best_val,
                "train_loss": train_losses,
                "val_loss": val_losses,
                "best_ckpt": best_ckpt,
                "final_ckpt": final_ckpt,
                "global_steps": global_step,
            }
        }

    def stage2_train_predictor(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: np.ndarray,
        Y_val: np.ndarray,
        save_dir: str,
        epochs: int,
        batch_size: int,
        lr: float,
        use_adversarial_training: bool,
        attack_prob: float,
        patience: int,
        add_noise: bool,
    ) -> Dict[str, Any]:
        os.makedirs(save_dir, exist_ok=True)

        _set_requires_grad(self.model.diffusion, False)
        _set_requires_grad(self.model.seq2seq, True)
        self.model.condition_weight.requires_grad = True

        params = list(self.model.seq2seq.parameters()) + [self.model.condition_weight]
        optimizer = _make_optimizer(params, lr=lr)

        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(Y_train)),
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(Y_val)),
            batch_size=batch_size,
            shuffle=False,
        )

        total_steps = epochs * len(train_loader)
        scheduler = _make_warmup_cosine_scheduler(optimizer, total_steps=total_steps, warmup_frac=0.075)

        best_val = float("inf")
        best_epoch = 0
        patience_counter = 0
        train_losses = []
        val_losses = []
        global_step = 0

        best_ckpt = os.path.join(save_dir, "best_model.pth")
        final_ckpt = os.path.join(save_dir, "final_model.pth")

        for epoch in range(epochs):
            self.model.train()
            running = 0.0
            pbar = tqdm(train_loader, desc=f"Stage2 {epoch+1}/{epochs} [train]", leave=False)
            for batch_x, batch_y in pbar:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                x_input, x_condition = self._maybe_attack_and_condition(batch_x, use_adversarial_training, attack_prob)

                optimizer.zero_grad()
                with torch.no_grad():
                    if self.use_conditioned_diffusion and x_condition is not None:
                        _, x_recon = self.model.diffusion(x_input, add_noise=add_noise, x_logic=x_condition)
                        x_for_pred = torch.sigmoid(self.model.condition_weight) * x_recon + (
                            1 - torch.sigmoid(self.model.condition_weight)
                        ) * x_condition
                    elif self.use_conditioned_diffusion:
                        x_logic = x_input
                        _, x_recon = self.model.diffusion(x_input, add_noise=add_noise, x_logic=x_logic)
                        x_for_pred = torch.sigmoid(self.model.condition_weight) * x_recon + (
                            1 - torch.sigmoid(self.model.condition_weight)
                        ) * x_logic
                    else:
                        _, x_recon = self.model.diffusion(x_input, add_noise=add_noise, x_logic=None)
                        x_for_pred = x_recon

                y_pred = self.model.seq2seq(x_for_pred)
                loss = self.criterion(y_pred, batch_y)
                loss.backward()
                optimizer.step()
                scheduler.step()
                global_step += 1
                running += float(loss.item())
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

            train_loss = running / max(1, len(train_loader))
            train_losses.append(train_loss)

            self.model.eval()
            v = 0.0
            with torch.no_grad():
                vpbar = tqdm(val_loader, desc=f"Stage2 {epoch+1}/{epochs} [val]", leave=False)
                for batch_x, batch_y in vpbar:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    if self.use_conditioned_diffusion:
                        x_logic = batch_x
                        _, x_recon = self.model.diffusion(batch_x, add_noise=False, x_logic=x_logic)
                        alpha = torch.sigmoid(self.model.condition_weight)
                        x_for_pred = alpha * x_recon + (1 - alpha) * x_logic
                    else:
                        _, x_recon = self.model.diffusion(batch_x, add_noise=False, x_logic=None)
                        x_for_pred = x_recon
                    y_pred = self.model.seq2seq(x_for_pred)
                    loss = self.criterion(y_pred, batch_y)
                    v += float(loss.item())
                    vpbar.set_postfix({"val_loss": f"{loss.item():.4f}"})

            val_loss = v / max(1, len(val_loader))
            val_losses.append(val_loss)

            logger.info(
                f"[Stage2] epoch {epoch+1}/{epochs} train={train_loss:.6f} val={val_loss:.6f} lr={scheduler.get_last_lr()[0]:.2e}"
            )

            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch + 1
                patience_counter = 0
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "stage": 2,
                        "epoch": epoch + 1,
                        "best_val_loss": best_val,
                    },
                    best_ckpt,
                )
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"[Stage2] early stop at epoch {epoch+1} (best epoch {best_epoch})")
                    break

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "stage": 2,
                "epoch": len(train_losses),
                "best_val_loss": best_val,
            },
            final_ckpt,
        )

        return {
            "stage2": {
                "best_epoch": best_epoch,
                "best_val_loss": best_val,
                "train_loss": train_losses,
                "val_loss": val_losses,
                "best_ckpt": best_ckpt,
                "final_ckpt": final_ckpt,
                "global_steps": global_step,
            }
        }


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("training_runs", f"staged_run_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_dir = os.environ.get("DATASET_DIR", "pytorch_datasets")

    X_train, Y_train, X_val, Y_val, meta = _load_dataset(ds_dir)
    feature_names = meta.get("features", []) or [f"f{i}" for i in range(X_train.shape[1])]
    horizon = int(meta.get("horizon", Y_train.shape[2] if Y_train.ndim == 3 else 16))
    in_channels = int(len(feature_names)) if feature_names else int(X_train.shape[1])

    use_conditioned_diffusion = os.environ.get("USE_CONDITIONED", "true").lower() == "true"

    model = LogicGuidedDiffusionForecast(
        in_channels=in_channels,
        diffusion_hidden=int(os.environ.get("DIFFUSION_HIDDEN", "32")),
        diffusion_levels=int(os.environ.get("DIFFUSION_LEVELS", "2")),
        seq_model_type=os.environ.get("SEQ_MODEL_TYPE", "seq2seq"),
        seq_d_model=int(os.environ.get("SEQ_D_MODEL", "64")),
        seq_n_heads=int(os.environ.get("SEQ_N_HEADS", "2")),
        seq_n_layers=int(os.environ.get("SEQ_N_LAYERS", "2")),
        seq_dim_feedforward=int(os.environ.get("SEQ_DIM_FEEDFORWARD", "512")),
        forecast_horizon=int(os.environ.get("FORECAST_HORIZON", str(horizon))),
        num_diffusion_steps=int(os.environ.get("NUM_DIFFUSION_STEPS", "10")),
        dropout=float(os.environ.get("DROPOUT", "0.1")),
        feature_names=feature_names,
        margin=0.0,
        use_conditioned_diffusion=use_conditioned_diffusion,
    )

    adv_eps = float(os.environ.get("ADV_EPS", "1.5"))
    trainer = StagedTrainer(
        model=model,
        device=device,
        feature_names=feature_names,
        use_conditioned_diffusion=use_conditioned_diffusion,
        adv_epsilon=adv_eps,
    )

    use_adv_training = os.environ.get("USE_ADV_TRAINING", "true").lower() == "true"
    attack_prob = float(os.environ.get("ATTACK_PROB", "0.5"))
    attack_type_config = os.environ.get("ATTACK_TYPE", "uniform")
    trainer.configure_attack_distribution(attack_type_config)

    default_epochs = 50 if device.type == "cuda" else 20
    default_batch = 128 if device.type == "cuda" else 64

    stage1_epochs = int(os.environ.get("STAGE1_EPOCHS", str(default_epochs)))
    stage2_epochs = int(os.environ.get("STAGE2_EPOCHS", str(default_epochs)))
    batch_size = int(os.environ.get("BATCH_SIZE", str(default_batch)))

    stage1_lr = float(os.environ.get("STAGE1_LR", os.environ.get("LR", "2e-4")))
    stage2_lr = float(os.environ.get("STAGE2_LR", os.environ.get("LR", "2e-4")))

    stage1_patience = int(os.environ.get("STAGE1_PATIENCE", "10"))
    stage2_patience = int(os.environ.get("STAGE2_PATIENCE", "10"))

    stage2_add_noise = os.environ.get("STAGE2_ADD_NOISE", "false").lower() == "true"

    config = {
        "timestamp": timestamp,
        "device": str(device),
        "data": {
            "dataset_dir": ds_dir,
            "train_samples": int(X_train.shape[0]),
            "val_samples": int(X_val.shape[0]),
            "input_shape": list(X_train.shape[1:]),
            "output_shape": list(Y_train.shape[1:]),
            "features": feature_names,
        },
        "model": {
            "use_conditioned_diffusion": use_conditioned_diffusion,
            "diffusion_hidden": int(os.environ.get("DIFFUSION_HIDDEN", "32")),
            "diffusion_levels": int(os.environ.get("DIFFUSION_LEVELS", "2")),
            "seq_model_type": os.environ.get("SEQ_MODEL_TYPE", "seq2seq"),
            "seq_d_model": int(os.environ.get("SEQ_D_MODEL", "64")),
            "seq_n_heads": int(os.environ.get("SEQ_N_HEADS", "2")),
            "seq_n_layers": int(os.environ.get("SEQ_N_LAYERS", "2")),
            "seq_dim_feedforward": int(os.environ.get("SEQ_DIM_FEEDFORWARD", "512")),
            "num_diffusion_steps": int(os.environ.get("NUM_DIFFUSION_STEPS", "10")),
            "dropout": float(os.environ.get("DROPOUT", "0.1")),
            "forecast_horizon": int(os.environ.get("FORECAST_HORIZON", str(horizon))),
        },
        "training": {
            "use_adversarial_training": use_adv_training,
            "attack_prob": attack_prob if use_adv_training else 0.0,
            "attack_type": attack_type_config if use_adv_training else "N/A",
            "attack_distribution": trainer.attack_distribution_config if use_adv_training else "N/A",
            "adv_epsilon": adv_eps,
            "batch_size": batch_size,
            "stage1": {
                "epochs": stage1_epochs,
                "lr": stage1_lr,
                "patience": stage1_patience,
                "objective": "reconstruction_only",
                "target": "x_input",
                "scheduler": "warmup+cosine",
            },
            "stage2": {
                "epochs": stage2_epochs,
                "lr": stage2_lr,
                "patience": stage2_patience,
                "objective": "forecast_only",
                "add_noise": stage2_add_noise,
                "scheduler": "warmup+cosine",
            },
        },
    }

    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Save dir: {save_dir}")
    logger.info(f"Device: {device}")
    logger.info(f"Data: train={X_train.shape} val={X_val.shape} y_train={Y_train.shape} y_val={Y_val.shape}")
    logger.info(f"Stage1: epochs={stage1_epochs} batch={batch_size} lr={stage1_lr} patience={stage1_patience}")
    logger.info(f"Stage2: epochs={stage2_epochs} batch={batch_size} lr={stage2_lr} patience={stage2_patience} add_noise={stage2_add_noise}")
    if use_adv_training:
        logger.info(f"Adv train: prob={attack_prob} eps={adv_eps} dist={trainer.attack_distribution_config}")

    stage1_dir = os.path.join(save_dir, "stage1")
    stage2_dir = os.path.join(save_dir, "stage2")
    os.makedirs(stage1_dir, exist_ok=True)
    os.makedirs(stage2_dir, exist_ok=True)

    hist1 = trainer.stage1_train_diffusion(
        X_train=X_train,
        X_val=X_val,
        save_dir=stage1_dir,
        epochs=stage1_epochs,
        batch_size=batch_size,
        lr=stage1_lr,
        use_adversarial_training=use_adv_training,
        attack_prob=attack_prob,
        patience=stage1_patience,
    )

    stage1_best = hist1["stage1"]["best_ckpt"]
    ckpt = torch.load(stage1_best, map_location=device)
    trainer.model.load_state_dict(ckpt["model_state_dict"])
    logger.info(f"Loaded Stage1 best: {stage1_best}")

    hist2 = trainer.stage2_train_predictor(
        X_train=X_train,
        Y_train=Y_train,
        X_val=X_val,
        Y_val=Y_val,
        save_dir=stage2_dir,
        epochs=stage2_epochs,
        batch_size=batch_size,
        lr=stage2_lr,
        use_adversarial_training=use_adv_training,
        attack_prob=attack_prob,
        patience=stage2_patience,
        add_noise=stage2_add_noise,
    )

    summary = {
        "timestamp": timestamp,
        "save_dir": save_dir,
        "stage1": {
            "best_epoch": hist1["stage1"]["best_epoch"],
            "best_val_loss": hist1["stage1"]["best_val_loss"],
            "total_epochs": len(hist1["stage1"]["train_loss"]),
        },
        "stage2": {
            "best_epoch": hist2["stage2"]["best_epoch"],
            "best_val_loss": hist2["stage2"]["best_val_loss"],
            "total_epochs": len(hist2["stage2"]["train_loss"]),
        },
        "config": config,
    }
    with open(os.path.join(save_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 80)
    logger.info("STAGED TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Stage1 best: {os.path.join(stage1_dir, 'stage1_best.pth')}")
    logger.info(f"Stage2 best: {os.path.join(stage2_dir, 'best_model.pth')}")


if __name__ == "__main__":
    main()

