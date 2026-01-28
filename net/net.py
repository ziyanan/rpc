import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union, List
import numpy as np

from .diffusion import DiffusionUNet
from .seq import Seq2SeqForecast, TransformerForecast


class LogicCorrectorWrapper(nn.Module):
    def __init__(self, feature_names=None, margin=0.0):
        super().__init__()
        from logics.logic_corrector import SimpleLogicCorrector
        self.corrector = SimpleLogicCorrector(feature_names=feature_names, margin=margin)
        self.feature_names = self.corrector.feature_names

    def forward(
        self, 
        x: torch.Tensor, 
        logic_name: Optional[Union[str, List[str]]] = None,
        logic_params: Optional[Union[dict, List[dict]]] = None
    ) -> torch.Tensor:
        if logic_name is None:
            return x

        device = x.device
        x_np = x.detach().cpu().numpy()

        if isinstance(logic_name, list):
            params_list = logic_params if isinstance(logic_params, list) else [logic_params] * len(logic_name)
            x_corrected_np = self.corrector.correct_multiple(x_np, logic_name, params_list=params_list)
        else:
            x_corrected_np = self.corrector.correct(x_np, logic_name, params=logic_params)

        x_corrected = torch.from_numpy(x_corrected_np).to(device)
        return x_corrected


class LogicGuidedDiffusionForecast(nn.Module):
    def __init__(
        self,
        in_channels: int,
        diffusion_hidden: int = 64,
        diffusion_levels: int = 3,
        seq_d_model: int = 128,
        seq_n_heads: int = 4,
        seq_n_layers: int = 3,
        forecast_horizon: int = 16,
        num_diffusion_steps: int = 10,
        dropout: float = 0.1,
        feature_names: Optional[List[str]] = None,
        margin: float = 0.0,
        use_conditioned_diffusion: bool = True,
        seq_model_type: str = 'seq2seq',
        seq_dim_feedforward: int = 512
    ):
        super().__init__()
        self.in_channels = in_channels
        self.forecast_horizon = forecast_horizon
        self.use_conditioned_diffusion = use_conditioned_diffusion

        self.diffusion = DiffusionUNet(
            in_channels=in_channels,
            hidden_channels=diffusion_hidden,
            num_levels=diffusion_levels,
            num_diffusion_steps=num_diffusion_steps,
            use_condition=use_conditioned_diffusion
        )

        self.logic_corrector = LogicCorrectorWrapper(
            feature_names=feature_names,
            margin=margin
        )

        if seq_model_type == 'transformer':
            self.seq2seq = TransformerForecast(
                in_channels=in_channels,
                d_model=seq_d_model,
                n_heads=seq_n_heads,
                n_encoder_layers=seq_n_layers,
                n_decoder_layers=seq_n_layers,
                dim_feedforward=seq_dim_feedforward,
                forecast_horizon=forecast_horizon,
                dropout=dropout
            )
        else:
            self.seq2seq = Seq2SeqForecast(
                in_channels=in_channels,
                d_model=seq_d_model,
                n_heads=seq_n_heads,
                n_layers=seq_n_layers,
                forecast_horizon=forecast_horizon,
                dropout=dropout
            )
        
        self.condition_weight = nn.Parameter(torch.tensor(0.3))  # Start with 30% condition, 70% recon

    def forward(
        self,
        x: torch.Tensor,
        x_condition: Optional[torch.Tensor] = None,
        logic_name: Optional[Union[str, List[str]]] = None,
        logic_params: Optional[Union[dict, List[dict]]] = None,
        diffusion_t: Optional[int] = None,
        add_noise: bool = True
    ) -> torch.Tensor:
        if self.use_conditioned_diffusion:
            if x_condition is not None:
                x_logic = x_condition
            elif logic_name is not None:
                x_logic = self.logic_corrector(x, logic_name, logic_params=logic_params)
            else:
                x_logic = x.clone()
            z, x_recon = self.diffusion(x, t=diffusion_t, add_noise=add_noise, x_logic=x_logic)
            
            alpha = torch.sigmoid(self.condition_weight)
            x_for_prediction = alpha * x_recon + (1 - alpha) * x_logic
        else:
            z, x_recon = self.diffusion(x, t=diffusion_t, add_noise=add_noise, x_logic=None)
            if logic_name is not None:
                x_recon = self.logic_corrector(x_recon, logic_name, logic_params=logic_params)
            x_for_prediction = x_recon
        
        y_pred = self.seq2seq(x_for_prediction)
        return y_pred

    def forward_with_intermediates(
        self,
        x: torch.Tensor,
        x_condition: Optional[torch.Tensor] = None,
        logic_name: Optional[Union[str, List[str]]] = None,
        logic_params: Optional[Union[dict, List[dict]]] = None,
        diffusion_t: Optional[int] = None,
        add_noise: bool = True
    ) -> Dict[str, torch.Tensor]:
        x_logic_condition = None
        if self.use_conditioned_diffusion:
            if x_condition is not None:
                x_logic_condition = x_condition
            elif logic_name is not None:
                x_logic_condition = self.logic_corrector(x, logic_name, logic_params=logic_params)
            else:
                x_logic_condition = x.clone()
            z, x_recon = self.diffusion(x, t=diffusion_t, add_noise=add_noise, x_logic=x_logic_condition)
            
            alpha = torch.sigmoid(self.condition_weight)
            x_for_prediction = alpha * x_recon + (1 - alpha) * x_logic_condition
        else:
            z, x_recon = self.diffusion(x, t=diffusion_t, add_noise=add_noise, x_logic=None)
            if logic_name is not None:
                x_recon = self.logic_corrector(x_recon, logic_name, logic_params=logic_params)
            x_for_prediction = x_recon
        
        y_pred = self.seq2seq(x_for_prediction)
        return {
            'z': z,
            'x_recon': x_recon,
            'x_logic_condition': x_logic_condition,
            'y_pred': y_pred
        }
