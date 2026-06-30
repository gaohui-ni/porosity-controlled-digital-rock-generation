import numpy as np
import torch
import torch.nn as nn

from src.models.vqvae3d import ResBlock3D

def sinusoidal_time_embedding(t: torch.Tensor, dim: int, max_period: int = 10000):
    half = dim // 2
    freqs = torch.exp(-np.log(max_period) * torch.arange(0, half, device=t.device).float() / half)
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb

class SelfAttention3D(nn.Module):
    def __init__(self, channels: int, heads: int = 4):
        super().__init__()
        self.mha = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.ln1 = nn.LayerNorm(channels)
        self.ff = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        B, C, D, H, W = x.shape
        N = D * H * W
        x_ = x.view(B, C, N).transpose(1, 2)  # [B,N,C]
        x_ln = self.ln1(x_)
        attn, _ = self.mha(x_ln, x_ln, x_ln)
        x_ = x_ + attn
        x_ = x_ + self.ff(x_)
        return x_.transpose(1, 2).view(B, C, D, H, W)

class UNetLatentCond(nn.Module):
    def __init__(self, channels: int, base: int = 64, time_dim: int = 128):
        super().__init__()
        C = int(channels)
        self.time_dim = int(time_dim)

        self.inc = nn.Sequential(nn.Conv3d(C, base, 3, padding=1), nn.SiLU(), ResBlock3D(base))
        self.down1 = nn.Sequential(nn.Conv3d(base, base*2, 4, stride=2, padding=1), nn.SiLU(), ResBlock3D(base*2))
        self.down2 = nn.Sequential(nn.Conv3d(base*2, base*4, 4, stride=2, padding=1), nn.SiLU(), ResBlock3D(base*4))

        self.attn = SelfAttention3D(base*4, heads=4)

        self.cond_mlp = nn.Sequential(
            nn.Linear(time_dim + 1, base*4),
            nn.SiLU(),
            nn.Linear(base*4, base*8),  # gamma+beta
        )

        self.up1 = nn.Sequential(nn.ConvTranspose3d(base*4, base*2, 4, stride=2, padding=1), nn.SiLU(), ResBlock3D(base*2))
        self.up2 = nn.Sequential(nn.ConvTranspose3d(base*2, base, 4, stride=2, padding=1), nn.SiLU(), ResBlock3D(base))
        self.out = nn.Conv3d(base, C, 3, padding=1)

    def forward(self, x, t, poro_scaled):
        temb = sinusoidal_time_embedding(t, self.time_dim)  # [B,time_dim]
        cond = torch.cat([temb, poro_scaled[:, None]], dim=1)
        film = self.cond_mlp(cond)
        B = x.size(0)
        ch = film.size(1) // 2
        gamma, beta = film[:, :ch], film[:, ch:]
        gamma = gamma.view(B, ch, 1, 1, 1)
        beta = beta.view(B, ch, 1, 1, 1)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        mid = self.attn(x3 * (1.0 + gamma) + beta)

        u1 = self.up1(mid) + x2
        u2 = self.up2(u1) + x1
        return self.out(u2)
