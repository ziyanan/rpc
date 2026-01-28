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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from net.net import LogicGuidedDiffusionForecast
from logics.logic_corrector import SimpleLogicCorrector
from logics.stl_attack_repair import AttackGenerator, STLGuidedRepairer, TeLExLearner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NetModelTrainer:
    def __init__(
        self,
        in_channels: int = 7,
        diffusion_hidden: int = 64,
        diffusion_levels: int = 3,
        seq_d_model: int = 128,
        seq_n_heads: int = 4,
        seq_n_layers: int = 3,
        forecast_horizon: int = 16,
        num_diffusion_steps: int = 10,
        dropout: float = 0.1,
        feature_names: list = None,
        margin: float = 0.0,
        use_conditioned_diffusion: bool = True,
        device: str = 'cpu',
        lr: float = 2e-4,
        reconstruction_weight: float = 0.5,
        seq_model_type: str = 'seq2seq',
        seq_dim_feedforward: int = 512
    ):
        self.device = torch.device(device)
        self.in_channels = in_channels
        self.forecast_horizon = forecast_horizon
        self.reconstruction_weight = reconstruction_weight
        
        if feature_names is None:
            feature_names = ['total_flow', 'avg_occupancy', 'avg_speed', 
                           'aq_NO2', 'aq_NOx', 'aq_O3', 'aq_PM25']
        
        self.model = LogicGuidedDiffusionForecast(
            in_channels=in_channels,
            diffusion_hidden=diffusion_hidden,
            diffusion_levels=diffusion_levels,
            seq_d_model=seq_d_model,
            seq_n_heads=seq_n_heads,
            seq_n_layers=seq_n_layers,
            forecast_horizon=forecast_horizon,
            num_diffusion_steps=num_diffusion_steps,
            dropout=dropout,
            feature_names=feature_names,
            margin=margin,
            use_conditioned_diffusion=use_conditioned_diffusion,
            seq_model_type=seq_model_type,
            seq_dim_feedforward=seq_dim_feedforward
        )
        
        self.model.to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.criterion = nn.MSELoss()
        self.base_lr = lr
        self.scheduler = None 
        self.corrector = SimpleLogicCorrector(feature_names=feature_names, margin=margin)
        self.logic_names = self.corrector.get_logic_names()
        self.feature_names = feature_names
        
        self.attack_types = ['gaussian', 'uniform', 'false_high', 'false_low', 'temporal']
        self.attackers = {
            attack_type: AttackGenerator(attack_type=attack_type, epsilon=1.5, seed=None)
            for attack_type in self.attack_types
        }
        
        self.attack_weights = None
        self.attack_distribution_config = None
        
        logger.info(f"Model initialized on {self.device}")
        logger.info(f"Available logic properties: {len(self.logic_names)}")
        logger.info(f"Reconstruction weight: {reconstruction_weight}")
    
    def configure_attack_distribution(self, attack_type_config: str):
        attack_type_config = attack_type_config.strip().lower()
        
        if attack_type_config == "uniform":
            self.attack_weights = None
            self.attack_distribution_config = "uniform (20% each type)"
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
        use_logic_conditioning=True,
        use_adversarial_training=False,
        attack_prob=0.5,
        save_dir=None
    ):
        if use_logic_conditioning and not self.logic_names:
            logger.info("No compatible logic properties for current features; disabling logic conditioning.")
            use_logic_conditioning = False
        if save_dir is None:
            save_dir = 'models/latest'
        self.save_dir = save_dir
        self.best_model_path = os.path.join(save_dir, 'best_model.pth')
        self.final_model_path = os.path.join(save_dir, 'final_model.pth')
        
        logger.info("Starting training...")
        logger.info(f"Save directory: {save_dir}")
        logger.info(f"Train data: {X_train.shape}, Targets: {Y_train.shape}")
        logger.info(f"Logic conditioning: {use_logic_conditioning}")
        logger.info(f"Adversarial training: {use_adversarial_training}")
        if use_adversarial_training:
            logger.info(f"  Attack probability: {attack_prob*100:.0f}%")
            logger.info(f"  Attack types: {self.attack_types}")
            logger.info(f"  Strategy:")
            logger.info(f"    Conditioned: Input=attacked, Condition=synthesized STL repair")
            logger.info(f"    Unconditioned: Input=attacked, no condition")
        
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
        
        total_steps = epochs * len(train_loader)
        warmup_steps = int(total_steps * 0.075)
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
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
                x_condition = None
                
                if apply_attack:
                    attack_type = self._sample_attack_type()
                    attacker = self.attackers[attack_type]
                    
                    x_attacked_np = attacker.attack(batch_x_np)
                    
                    if use_logic_conditioning:
                        learner = TeLExLearner(self.feature_names, timesteps=batch_x_np.shape[2])
                        learned_bounds = learner.learn_bounds_simple(batch_x_np, self.feature_names, verbose=False)
                        
                        repairer = STLGuidedRepairer(self.feature_names, learned_bounds=learned_bounds)
                        x_repaired_np = repairer.repair(x_attacked_np, method='comprehensive', verbose=False)
                        
                        x_input = torch.FloatTensor(x_attacked_np).to(self.device)
                        x_condition = torch.FloatTensor(x_repaired_np).to(self.device)
                    else:
                        x_input = torch.FloatTensor(x_attacked_np).to(self.device)
                        x_condition = None
                
                self.optimizer.zero_grad()
                
                intermediates = self.model.forward_with_intermediates(
                    x_input, 
                    x_condition=x_condition,
                    add_noise=True
                )
                y_pred = intermediates['y_pred']
                x_recon = intermediates['x_recon']
                
                forecast_loss = self.criterion(y_pred, batch_y)
                reconstruction_loss = self.criterion(x_recon, x_input)
                loss = forecast_loss + self.reconstruction_weight * reconstruction_loss
                
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                global_step += 1
                
                train_loss += loss.item()
                train_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'f_loss': f'{forecast_loss.item():.4f}',
                    'r_loss': f'{reconstruction_loss.item():.4f}'
                })
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [val]", leave=False)
                    for batch_x, batch_y in val_pbar:
                        batch_x_np = batch_x.cpu().numpy()
                        
                        logic_name = None
                        if use_logic_conditioning:
                            logic_ids = np.random.choice(len(self.logic_names), size=len(batch_x_np))
                            logic_name = self.logic_names[logic_ids[0]]

                        intermediates = self.model.forward_with_intermediates(
                            batch_x,
                            logic_name=logic_name,
                            add_noise=False
                        )
                        y_pred = intermediates['y_pred']
                        x_recon = intermediates['x_recon']
                        
                        forecast_loss = self.criterion(y_pred, batch_y)
                        reconstruction_loss = self.criterion(x_recon, batch_x)
                        loss = forecast_loss + self.reconstruction_weight * reconstruction_loss
                        
                        val_loss += loss.item()
                        val_pbar.set_postfix({
                            'val_loss': f'{loss.item():.4f}',
                            'f_loss': f'{forecast_loss.item():.4f}',
                            'r_loss': f'{reconstruction_loss.item():.4f}'
                        })
                
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
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
            'in_channels': self.in_channels,
            'forecast_horizon': self.forecast_horizon,
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Model loaded from {path}")


def plot_training_curves(history, save_path):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', marker='o', markersize=4)
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Val Loss', marker='s', markersize=4)
        
        # Mark best epoch
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
    save_dir = os.path.join('training_runs', f'run_{timestamp}')
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
    
    use_conditioned_diffusion = os.environ.get('USE_CONDITIONED', 'true').lower() == 'true'
    
    logger.info(f"Dataset loaded: train {X_train_t.shape}, val {X_val_t.shape}")
    logger.info(f"Targets: train {Y_train_t.shape}, val {Y_val_t.shape}")
    logger.info(f"Model architecture: {'CONDITIONED' if use_conditioned_diffusion else 'UNCONDITIONED'} diffusion")
    
    trainer = NetModelTrainer(
        in_channels=num_features,
        diffusion_hidden=int(os.environ.get('DIFFUSION_HIDDEN', '32')),
        diffusion_levels=int(os.environ.get('DIFFUSION_LEVELS', '2')),
        seq_d_model=int(os.environ.get('SEQ_D_MODEL', '64')),
        seq_n_heads=int(os.environ.get('SEQ_N_HEADS', '2')),
        seq_n_layers=int(os.environ.get('SEQ_N_LAYERS', '2')),
        forecast_horizon=horizon,
        num_diffusion_steps=int(os.environ.get('NUM_DIFFUSION_STEPS', '10')),
        dropout=float(os.environ.get('DROPOUT', '0.1')),
        feature_names=feature_names,
        margin=0.0,
        use_conditioned_diffusion=use_conditioned_diffusion,
        device=device,
        lr=float(os.environ.get('LR', '2e-4')),
        reconstruction_weight=float(os.environ.get('RECON_WEIGHT', '0.5')),
        seq_model_type=os.environ.get('SEQ_MODEL_TYPE', 'seq2seq'),
        seq_dim_feedforward=int(os.environ.get('SEQ_DIM_FEEDFORWARD', '512'))
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
            'use_conditioned_diffusion': use_conditioned_diffusion,
            'diffusion_hidden': int(os.environ.get('DIFFUSION_HIDDEN', '32')),
            'diffusion_levels': int(os.environ.get('DIFFUSION_LEVELS', '2')),
            'seq_model_type': os.environ.get('SEQ_MODEL_TYPE', 'seq2seq'),
            'seq_d_model': int(os.environ.get('SEQ_D_MODEL', '64')),
            'seq_n_heads': int(os.environ.get('SEQ_N_HEADS', '2')),
            'seq_n_layers': int(os.environ.get('SEQ_N_LAYERS', '2')),
            'seq_dim_feedforward': int(os.environ.get('SEQ_DIM_FEEDFORWARD', '512')),
            'num_diffusion_steps': int(os.environ.get('NUM_DIFFUSION_STEPS', '10')),
            'dropout': float(os.environ.get('DROPOUT', '0.1')),
            'forecast_horizon': horizon,
        },
        'training': {
            'epochs': int(os.environ.get('EPOCHS', default_epochs)),
            'batch_size': int(os.environ.get('BATCH_SIZE', default_batch)),
            'lr': float(os.environ.get('LR', '2e-4')),
            'reconstruction_weight': float(os.environ.get('RECON_WEIGHT', '0.5')),
            'logic_conditioning': True,
            'adversarial_training': use_adv_training,
            'attack_prob': attack_prob if use_adv_training else 0.0,
            'attack_type': attack_type_config if use_adv_training else 'N/A',
            'attack_distribution': trainer.attack_distribution_config if use_adv_training else 'N/A',
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
        use_logic_conditioning=True,
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

