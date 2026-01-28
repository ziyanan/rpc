import torch
import torch.nn as nn
from typing import Tuple, Optional, List


def get_sigma_schedule(num_steps: int = 10, sigma_min: float = 0.01, sigma_max: float = 1.0) -> torch.Tensor:
    return torch.linspace(sigma_max, sigma_min, num_steps)


def forward_diffusion(x: torch.Tensor, t: int, sigma_schedule: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    sigma = sigma_schedule[t]
    epsilon = torch.randn_like(x)
    x_noisy = x + sigma * epsilon
    return x_noisy, epsilon


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, downsample: bool = True):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.act = nn.SiLU()
        self.pool = nn.MaxPool1d(2) if downsample else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.act(self.conv(x)))


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv1d(in_ch + skip_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-1] != skip.shape[-1]:
            x = nn.functional.interpolate(x, size=skip.shape[-1], mode='nearest')
        x = torch.cat([x, skip], dim=1)
        return self.act(self.conv(x))


class DiffusionUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_levels: int = 3,
        num_diffusion_steps: int = 10,
        sigma_min: float = 0.01,
        sigma_max: float = 1.0,
        use_condition: bool = False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_levels = num_levels
        self.use_condition = use_condition
        self.register_buffer('sigma_schedule', get_sigma_schedule(num_diffusion_steps, sigma_min, sigma_max))

        input_channels = in_channels * 2 if use_condition else in_channels
        self.input_conv = nn.Conv1d(input_channels, hidden_channels, kernel_size=3, padding=1)

        self.encoder = nn.ModuleList()
        enc_channels: List[int] = [hidden_channels]
        ch = hidden_channels
        for i in range(num_levels):
            out_ch = ch * 2
            self.encoder.append(ConvBlock(ch, out_ch, downsample=True))
            enc_channels.append(out_ch)
            ch = out_ch

        self.bottleneck_ch = ch
        self.bottleneck = nn.Sequential(
            nn.Conv1d(ch, ch, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(ch, ch, kernel_size=3, padding=1),
            nn.SiLU()
        )

        self.decoder = nn.ModuleList()
        for i in range(num_levels):
            skip_ch = enc_channels[-(i + 1)]
            out_ch = skip_ch // 2 if i < num_levels - 1 else hidden_channels
            self.decoder.append(UpBlock(ch, skip_ch, out_ch))
            ch = out_ch

        self.output_conv = nn.Conv1d(ch, in_channels, kernel_size=3, padding=1)

    def add_noise(self, x: torch.Tensor, t: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if t is None:
            t = len(self.sigma_schedule) // 2
        t = min(t, len(self.sigma_schedule) - 1)
        return forward_diffusion(x, t, self.sigma_schedule)

    def forward(
        self, 
        x: torch.Tensor, 
        t: Optional[int] = None, 
        add_noise: bool = True,
        x_logic: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if add_noise:
            x_noisy, _ = self.add_noise(x, t)
        else:
            x_noisy = x

        if self.use_condition and x_logic is not None:
            x_combined = torch.cat([x_noisy, x_logic], dim=1)
        else:
            x_combined = x_noisy

        h = self.input_conv(x_combined)

        skips = [h]
        for enc_block in self.encoder:
            h = enc_block(h)
            skips.append(h)

        z = self.bottleneck(h)

        h = z
        for i, dec_block in enumerate(self.decoder):
            skip = skips[-(i + 1)]
            h = dec_block(h, skip)

        x_recon = self.output_conv(h)

        if x_recon.shape[-1] != x.shape[-1]:
            x_recon = nn.functional.interpolate(x_recon, size=x.shape[-1], mode='linear', align_corners=False)

        return z, x_recon
