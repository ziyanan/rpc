import torch
import torch.nn as nn
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TemporalTransformerBaseline(nn.Module):
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
        max_input_len: int = 5000
    ):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.forecast_horizon = forecast_horizon
        
        self.input_proj = nn.Linear(in_channels, d_model)
        
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_input_len, dropout=dropout)
        self.pos_decoder = PositionalEncoding(d_model, max_len=forecast_horizon + 100, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
            activation='gelu'
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_decoder_layers)
        
        self.output_proj = nn.Linear(d_model, in_channels)
        
        self.decoder_start_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def _generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(
        self,
        x: torch.Tensor,
        x_condition: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        input_len = x.shape[2]
        
        x = x.transpose(0, 2).transpose(1, 2)
        x = self.input_proj(x)
        x = self.pos_encoder(x)

        memory = self.encoder(x)
        
        decoder_input = self.decoder_start_token.expand(self.forecast_horizon, batch_size, -1)
        decoder_input = self.pos_decoder(decoder_input)
        
        tgt_mask = self._generate_square_subsequent_mask(self.forecast_horizon, x.device)
        
        output = self.decoder(
            tgt=decoder_input,
            memory=memory,
            tgt_mask=tgt_mask
        )
        
        output = self.output_proj(output)
        
        output = output.transpose(0, 1).transpose(1, 2)
        
        return output

