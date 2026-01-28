import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
import logging
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from net.baselines.transformer_baseline import TemporalTransformerBaseline
from logics.stl_attack_repair import AttackGenerator
from smoothing.test_smoothing import create_meaningful_stl_formulas
from smoothing.stl_utils import batch_stl_robustness

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransformerBaselineTrainer:
    def __init__(
        self,
        in_channels: int = 7,
        d_model: int = 128,
        n_heads: int = 4,
        n_encoder_layers: int = 3,
        n_decoder_layers: int = 3,
        dim_feedforward: int = 512,
        forecast_horizon: int = 6,
        dropout: float = 0.1,
        device: str = 'cpu',
        lr: float = 2e-4,
        feature_names=None,
        use_stl_teacher: bool = False,
        stl_weight: float = 0.1
    ):
        self.device = torch.device(device)
        self.in_channels = in_channels
        self.forecast_horizon = forecast_horizon
        self.feature_names = feature_names or []
        self.use_stl_teacher = use_stl_teacher
        self.stl_weight = stl_weight
        
        self.model = TemporalTransformerBaseline(
            in_channels=in_channels,
            d_model=d_model,
            n_heads=n_heads,
            n_encoder_layers=n_encoder_layers,
            n_decoder_layers=n_decoder_layers,
            dim_feedforward=dim_feedforward,
            forecast_horizon=forecast_horizon,
            dropout=dropout
        )
        
        self.model.to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.criterion = nn.MSELoss()
        self.base_lr = lr
        self.scheduler = None  # Will be initialized in train() when we know total steps
        
        self.attack_types = ['gaussian', 'uniform', 'false_high', 'false_low', 'temporal', 
                            'spike', 'dropout', 'spoofing', 'clipping', 'jitter']
        self.attackers = {
            attack_type: AttackGenerator(attack_type=attack_type, epsilon=1.5, seed=None)
            for attack_type in self.attack_types
        }
        
        self.attack_weights = None
        self.attack_distribution_config = None
        
        logger.info(f"Model initialized on {self.device}")
    
    def configure_attack_distribution(self, attack_type_config: str):
        attack_type_config = attack_type_config.strip().lower()
        
        if attack_type_config == "uniform":
            self.attack_weights = None
            self.attack_distribution_config = "uniform (equal probability)"
        elif attack_type_config in self.attack_types:
            self.attack_weights = {attack_type_config: 1.0}
            self.attack_distribution_config = f"100% {attack_type_config}"
        elif ":" in attack_type_config:
            self.attack_weights = {}
            pairs = attack_type_config.split(",")
            total_weight = 0.0
            
            for pair in pairs:
                if ":" not in pair:
                    raise ValueError(f"Invalid weighted format: {pair}. Expected 'type:weight'")
                
                attack_type, weight_str = pair.split(":")
                attack_type = attack_type.strip()
                
                if attack_type not in self.attack_types:
                    raise ValueError(f"Unknown attack type: {attack_type}. Must be one of {self.attack_types}")
                
                try:
                    weight = float(weight_str.strip())
                except ValueError:
                    raise ValueError(f"Invalid weight: {weight_str}. Must be a float.")
                
                if weight < 0 or weight > 1:
                    raise ValueError(f"Weight must be in [0, 1], got {weight}")
                
                self.attack_weights[attack_type] = weight
                total_weight += weight
            
            if not np.isclose(total_weight, 1.0, atol=1e-3):
                raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
            
            weight_strs = [f"{k}:{v:.0%}" for k, v in self.attack_weights.items()]
            self.attack_distribution_config = ", ".join(weight_strs)
        else:
            raise ValueError(
                f"Invalid attack_type_config: {attack_type_config}. "
                f"Must be 'uniform', one of {self.attack_types}, or weighted format 'type1:w1,type2:w2'"
            )
        
        logger.info(f"Attack distribution configured: {self.attack_distribution_config}")
    
    def _sample_attack_type(self) -> str:
        if self.attack_weights is None:
            return np.random.choice(self.attack_types)
        else:
            types = list(self.attack_weights.keys())
            weights = list(self.attack_weights.values())
            return np.random.choice(types, p=weights)
    
    def train(
        self,
        X_train,
        Y_train,
        X_val=None,
        Y_val=None,
        epochs=50,
        batch_size=32,
        use_adversarial_training=False,
        attack_prob=0.5,
        save_dir=None,
        use_stl_teacher=None
    ):
        if use_stl_teacher is not None:
            self.use_stl_teacher = use_stl_teacher
        if save_dir is None:
            save_dir = 'models/latest'
        self.save_dir = save_dir
        self.best_model_path = os.path.join(save_dir, 'best_model.pth')
        self.final_model_path = os.path.join(save_dir, 'final_model.pth')
        
        logger.info("Starting training...")
        logger.info(f"Save directory: {save_dir}")
        logger.info(f"Train data: {X_train.shape}, Targets: {Y_train.shape}")
        logger.info(f"Adversarial training: {use_adversarial_training}")
        if use_adversarial_training:
            logger.info(f"  Attack probability: {attack_prob*100:.0f}%")
            logger.info(f"  Attack types: {self.attack_types}")
        
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        Y_train_tensor = torch.FloatTensor(Y_train).to(self.device)
        
        train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_loader = None
        if X_val is not None and Y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            Y_val_tensor = torch.FloatTensor(Y_val).to(self.device)
            val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        patience = 10
        
        global_step = 0
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]", leave=False)
            for batch_x, batch_y in train_pbar:
                batch_x_np = batch_x.cpu().numpy()
                
                apply_attack = use_adversarial_training and (np.random.random() < attack_prob)
                x_input = batch_x
                
                if apply_attack:
                    attack_type = self._sample_attack_type()
                    attacker = self.attackers[attack_type]
                    x_attacked_np = attacker.attack(batch_x_np)
                    x_input = torch.FloatTensor(x_attacked_np).to(self.device)
                
                self.optimizer.zero_grad()
                
                y_pred = self.model(x_input, x_condition=None)
                loss = self.criterion(y_pred, batch_y)

                if self.use_stl_teacher and len(self.feature_names) == batch_y.shape[1]:
                    batch_y_np = batch_y.detach().cpu().numpy()
                    formulas, _, _ = create_meaningful_stl_formulas(
                        batch_y_np,
                        self.feature_names,
                        horizon=batch_y.shape[2],
                        verbose=False,
                        X_data=None
                    )
                    stl_penalties = []
                    for formula in formulas.values():
                        rho_scores = batch_stl_robustness(
                            y_pred.detach(), formula, feature_names=self.feature_names
                        )
                        rho_tensor = torch.from_numpy(rho_scores).to(self.device)
                        stl_penalties.append(torch.relu(-rho_tensor))
                    if stl_penalties:
                        stl_loss = torch.mean(torch.cat(stl_penalties))
                        loss = loss + self.stl_weight * stl_loss
                
                loss.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                global_step += 1
                
                train_loss += loss.item()
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [val]", leave=False)
                    for batch_x, batch_y in val_pbar:
                        y_pred = self.model(batch_x, x_condition=None)
                        loss = self.criterion(y_pred, batch_y)
                        val_loss += loss.item()
                        val_pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})
                
                val_loss /= len(val_loader)
                val_losses.append(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    patience_counter = 0
                    self.save_model(self.best_model_path)
                    logger.info(f"  New best model saved! Val Loss: {val_loss:.6f}")
                else:
                    patience_counter += 1
                    logger.info(f"  No improvement for {patience_counter} epochs")
                    if patience_counter >= patience:
                        logger.info(f"Early stopping at epoch {epoch+1}. Best was epoch {best_epoch+1}")
                        break
            else:
                val_losses.append(train_loss)
            
            current_lr = self.scheduler.get_last_lr()[0] if self.scheduler is not None else self.base_lr
            logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_losses[-1]:.6f}, LR: {current_lr:.2e}")
        
        if best_epoch > 0:
            logger.info(f"Loading best model from epoch {best_epoch+1}")
            self.load_model(self.best_model_path)
        
        self.save_model(self.final_model_path)
        
        return {
            'train_loss': train_losses, 
            'val_loss': val_losses,
            'best_epoch': best_epoch + 1,
            'best_val_loss': best_val_loss,
            'save_dir': self.save_dir
        }
    
    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'in_channels': self.in_channels,
            'forecast_horizon': self.forecast_horizon,
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
            if self.scheduler is not None:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info(f"Model loaded from {path}")


def plot_training_curves(history, save_path):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', marker='o', markersize=4)
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Val Loss', marker='s', markersize=4)
        
        best_epoch = np.argmin(history['val_loss'])
        plt.axvline(x=best_epoch, color='r', linestyle='--', alpha=0.5, label=f'Best (epoch {best_epoch+1})')
        plt.scatter([best_epoch], [history['val_loss'][best_epoch]], 
                   color='red', s=100, zorder=5, marker='*')
    
    plt.title('Training & Validation Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    epochs = range(1, len(history['train_loss']) + 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss', marker='o', markersize=4)
    if 'val_loss' in history:
        plt.plot(epochs, history['val_loss'], label='Val Loss', marker='s', markersize=4)
    plt.title('Loss Convergence (Log Scale)', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Training curves saved to {save_path}")


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join('training_runs', f'baseline_transformer_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training Run: {timestamp}")
    logger.info(f"Save directory: {save_dir}")
    logger.info(f"Using device: {device}")
    
    ds_dir = os.environ.get('DATASET_DIR', 'pytorch_datasets')
    X_train = np.load(os.path.join(ds_dir, 'X_clean_train.npy'))
    X_val = np.load(os.path.join(ds_dir, 'X_clean_val.npy'))
    Y_train = np.load(os.path.join(ds_dir, 'Y_train.npy'))
    Y_val = np.load(os.path.join(ds_dir, 'Y_val.npy'))
    
    with open(os.path.join(ds_dir, 'timeseries_meta.json')) as f:
        meta = json.load(f)
    
    seq_len = int(meta.get('seq_len', X_train.shape[1]))
    num_features = len(meta.get('features', [])) or X_train.shape[2]
    horizon = int(meta.get('horizon', 16))
    feature_names = meta.get('features', [])
    
    max_train = int(os.environ.get('MAX_TRAIN_SAMPLES', '50000'))
    max_val = int(os.environ.get('MAX_VAL_SAMPLES', '15000'))
    
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
    
    logger.info(f"Dataset loaded: train {X_train_t.shape}, val {X_val_t.shape}")
    logger.info(f"Targets: train {Y_train_t.shape}, val {Y_val_t.shape}")
    logger.info(f"Model architecture: Transformer Baseline")
    
    use_stl_teacher = os.environ.get('USE_STL_TEACHER', 'false').lower() == 'true'
    stl_weight = float(os.environ.get('STL_WEIGHT', '0.1'))

    trainer = TransformerBaselineTrainer(
        in_channels=num_features,
        d_model=int(os.environ.get('D_MODEL', '128')),
        n_heads=int(os.environ.get('N_HEADS', '4')),
        n_encoder_layers=int(os.environ.get('N_ENCODER_LAYERS', '3')),
        n_decoder_layers=int(os.environ.get('N_DECODER_LAYERS', '3')),
        dim_feedforward=int(os.environ.get('DIM_FEEDFORWARD', '512')),
        forecast_horizon=horizon,
        dropout=float(os.environ.get('DROPOUT', '0.1')),
        device=device,
        lr=float(os.environ.get('LR', '2e-4')),
        feature_names=feature_names,
        use_stl_teacher=use_stl_teacher,
        stl_weight=stl_weight
    )
    
    default_epochs = 50 if device.type == 'cuda' else 20
    default_batch = 128 if device.type == 'cuda' else 64
    
    use_adv_training = os.environ.get('USE_ADV_TRAINING', 'true').lower() == 'true'
    attack_prob = float(os.environ.get('ATTACK_PROB', '0.5'))
    attack_type_config = os.environ.get('ATTACK_TYPE', 'uniform')
    
    trainer.configure_attack_distribution(attack_type_config)
    
    config = {
        'timestamp': timestamp,
        'device': str(device),
        'data': {
            'train_samples': len(X_train_t),
            'val_samples': len(X_val_t),
            'input_shape': list(X_train_t.shape[1:]),
            'output_shape': list(Y_train_t.shape[1:]),
            'features': feature_names
        },
        'model': {
            'model_type': 'transformer_baseline',
            'd_model': int(os.environ.get('D_MODEL', '128')),
            'n_heads': int(os.environ.get('N_HEADS', '4')),
            'n_encoder_layers': int(os.environ.get('N_ENCODER_LAYERS', '3')),
            'n_decoder_layers': int(os.environ.get('N_DECODER_LAYERS', '3')),
            'dim_feedforward': int(os.environ.get('DIM_FEEDFORWARD', '512')),
            'dropout': float(os.environ.get('DROPOUT', '0.1')),
            'forecast_horizon': horizon,
        },
        'training': {
            'epochs': int(os.environ.get('EPOCHS', default_epochs)),
            'batch_size': int(os.environ.get('BATCH_SIZE', default_batch)),
            'lr': float(os.environ.get('LR', '2e-4')),
            'adversarial_training': use_adv_training,
            'attack_prob': attack_prob if use_adv_training else 0.0,
            'attack_type': attack_type_config if use_adv_training else 'N/A',
            'attack_distribution': trainer.attack_distribution_config if use_adv_training else 'N/A',
            'stl_teacher': use_stl_teacher,
            'stl_weight': stl_weight if use_stl_teacher else 0.0,
        }
    }
    
    config_path = os.path.join(save_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Configuration saved to {config_path}")
    
    history = trainer.train(
        X_train=X_train_t,
        Y_train=Y_train_t,
        X_val=X_val_t,
        Y_val=Y_val_t,
        epochs=config['training']['epochs'],
        batch_size=config['training']['batch_size'],
        use_adversarial_training=use_adv_training,
        attack_prob=attack_prob,
        save_dir=save_dir
    )
    
    plot_path = os.path.join(save_dir, 'training_curves.png')
    plot_training_curves(history, plot_path)
    
    summary = {
        'timestamp': timestamp,
        'best_epoch': history['best_epoch'],
        'best_val_loss': history['best_val_loss'],
        'final_train_loss': history['train_loss'][-1],
        'final_val_loss': history['val_loss'][-1],
        'total_epochs': len(history['train_loss']),
        'config': config
    }
    
    summary_path = os.path.join(save_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Training summary saved to {summary_path}")
    
    logger.info("="*80)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*80)
    logger.info(f"Best model: {os.path.join(save_dir, 'best_model.pth')}")
    logger.info(f"Final model: {os.path.join(save_dir, 'final_model.pth')}")
    logger.info(f"Training curves: {plot_path}")
    logger.info(f"Best epoch: {history['best_epoch']} (Val Loss: {history['best_val_loss']:.6f})")
    logger.info("="*80)


if __name__ == "__main__":
    main()
