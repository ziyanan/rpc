import torch
import torch.nn as nn
import math
from typing import Optional


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class SequenceBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dim_ff: int = None, dropout: float = 0.1, is_decoder: bool = False):
        super().__init__()
        dim_ff = dim_ff or d_model * 4
        self.is_decoder = is_decoder

        self.norm1 = RMSNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        if is_decoder:
            self.norm2 = RMSNorm(d_model)
            self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.norm3 = RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.self_attn(h, h, h, attn_mask=tgt_mask)
        x = x + attn_out

        if self.is_decoder and memory is not None:
            h = self.norm2(x)
            cross_out, _ = self.cross_attn(h, memory, memory)
            x = x + cross_out

        h = self.norm3(x)
        x = x + self.ffn(h)
        return x


class Seq2SeqForecast(nn.Module):
    def __init__(
        self,
        in_channels: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        forecast_horizon: int = 16,
        dropout: float = 0.1
    ):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.forecast_horizon = forecast_horizon

        self.input_proj = nn.Linear(in_channels, d_model)

        self.encoder_layers = nn.ModuleList([
            SequenceBlock(d_model, n_heads, dropout=dropout, is_decoder=False)
            for _ in range(n_layers)
        ])

        self.decoder_layers = nn.ModuleList([
            SequenceBlock(d_model, n_heads, dropout=dropout, is_decoder=True)
            for _ in range(n_layers)
        ])

        self.start_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embedding = nn.Parameter(torch.randn(1, forecast_horizon, d_model) * 0.02)

        self.output_proj = nn.Linear(d_model, in_channels)

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = x.transpose(1, 2)

        enc_input = self.input_proj(x)

        memory = enc_input
        for enc_layer in self.encoder_layers:
            memory = enc_layer(memory)

        dec_input = self.start_token.expand(batch_size, self.forecast_horizon, -1)
        dec_input = dec_input + self.pos_embedding

        causal_mask = self._get_causal_mask(self.forecast_horizon, x.device)

        h = dec_input
        for dec_layer in self.decoder_layers:
            h = dec_layer(h, memory=memory, tgt_mask=causal_mask)

        output = self.output_proj(h)
        output = output.transpose(1, 2)

        return output


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


class TransformerForecast(nn.Module):
    def __init__(
        self,
        in_channels: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_encoder_layers: int = 3,
        n_decoder_layers: int = 3,
        dim_feedforward: int = 512,
        forecast_horizon: int = 16,
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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