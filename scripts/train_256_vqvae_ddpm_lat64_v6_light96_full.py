# train_256_vqvae_ddpm_lat64_v6_light96.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import time
import json
import traceback
import logging
from contextlib import nullcontext
from dataclasses import dataclass, asdict
from typing import Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

def cuda_autocast(enabled: bool):
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp"):
        return torch.amp.autocast(device_type="cuda", enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


# =========================
# Logging (no tqdm)
# =========================
def setup_logger(save_dir: str, name: str = "train") -> logging.Logger:
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "train.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)

    logger.addHandler(fh)
    logger.addHandler(sh)

    logger.info(f"Logger initialized. Writing to: {log_path}")
    return logger


# =========================
# Config
# =========================
@dataclass
class CFG:
    raw_path: str = "S1.raw"
    raw_shape: Tuple[int, int, int] = (800, 800, 800)
    patch_size: int = 256
    n_samples: int = 1000
    min_porosity: float = 0.0
    max_tries: int = 50

    # down=4 -> latent 64^3 (for fine pores)
    downsample_factor: int = 4  # fixed for this script (4)

    # VQ-VAE
    embedding_dim: int = 32
    num_embeddings: int = 1024
    commitment_cost: float = 0.25
    lr_vae: float = 2e-4
    epochs_vae: int = 80
    batch_size_vae: int = 1
    amp: bool = True

    # VQ-VAE width control (IMPORTANT)
    vae_base_ch: int = 32       # base width
    vae_max_ch: int = 96        # max width (<=96 recommended for down=4 256^3 on 40GB)

    # reconstruction losses
    bce_weight: float = 1.0
    dice_weight: float = 0.2
    pos_weight_mode: str = "batch"  # "fixed" or "batch"
    poro_center: float = 0.13
    pos_weight_clip: float = 20.0

    # DDPM
    lr_ddpm: float = 1e-4
    epochs_ddpm: int = 150
    batch_size_ddpm: int = 1
    n_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02

    # DDPM UNet width
    unet_base: int = 64
    unet_time_dim: int = 128

    # conditioning
    poro_scale: float = 0.02
    poro_clip: float = 5.0

    # runtime
    save_interval: int = 5
    log_interval: int = 20
    save_dir: str = "./ckpt_lat64_vq80_ddpm150_light96"
    device: str = "cuda"
    num_workers: int = 0
    pin_memory: bool = True
    seed: int = 123

    # latent stats for ddpm
    stats_batches: int = 80

    # sampling
    target_porosity: float = 0.13
    sample_seed: int = 0
    clamp_x: float = 15.0
    eps_clip: float = 5.0

    # IMPORTANT: quantile thresholding to match porosity
    use_quantile_threshold: bool = True

    # grad clip (optional)
    grad_clip: float = 1.0


# =========================
# Dataset
# =========================
class Rock256ProbDataset(Dataset):
    """
    raw: uint8, 0=solid matrix, 1=pore space
    return:
      x: float32 in [0,1], [1,256,256,256]
      poro: float32 scalar in [0,1]
    """
    def __init__(self, raw_path: str, raw_shape: Tuple[int, int, int], patch_size: int,
                 n_samples: int, min_porosity: float, max_tries: int):
        self.raw_shape = tuple(raw_shape)
        self.patch_size = int(patch_size)
        self.n_samples = int(n_samples)
        self.min_porosity = float(min_porosity)
        self.max_tries = int(max_tries)

        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"raw not found: {raw_path}")

        self.mm = np.memmap(raw_path, dtype=np.uint8, mode="r", shape=self.raw_shape)

    def __len__(self):
        return self.n_samples

    def _sample_patch(self):
        S = self.patch_size
        X, Y, Z = self.raw_shape
        x0 = np.random.randint(0, X - S + 1)
        y0 = np.random.randint(0, Y - S + 1)
        z0 = np.random.randint(0, Z - S + 1)
        patch = np.array(self.mm[x0:x0+S, y0:y0+S, z0:z0+S], dtype=np.uint8)
        return patch

    def __getitem__(self, idx):
        for i in range(self.max_tries):
            p = self._sample_patch()
            poro = float(np.mean(p == 1))
            if poro >= self.min_porosity or i == self.max_tries - 1:
                x = p.astype(np.float32)
                x = torch.from_numpy(x).unsqueeze(0)  # [1,S,S,S]
                return x, torch.tensor(poro, dtype=torch.float32)


# =========================
# Utils: losses
# =========================
def soft_dice_loss(prob, target, eps=1e-6):
    num = 2 * (prob * target).sum(dim=(1,2,3,4))
    den = (prob + target).sum(dim=(1,2,3,4)) + eps
    return 1 - (num / den).mean()

def make_pos_weight(cfg: CFG, poro_batch: torch.Tensor, device: str):
    if cfg.pos_weight_mode == "batch":
        p = float(poro_batch.mean().detach().cpu())
    else:
        p = float(cfg.poro_center)
    pw = (1.0 - p) / max(p, 1e-6)
    pw = float(np.clip(pw, 1.0, cfg.pos_weight_clip))
    return torch.tensor([pw], device=device, dtype=torch.float32)

def save_cfg_json(cfg: CFG, logger: logging.Logger):
    path = os.path.join(cfg.save_dir, "run_cfg.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)
    logger.info(f"[CFG] saved: {path}")

def save_ckpt(path: str, model: nn.Module, optimizer, epoch: int, extra: Optional[dict], logger: logging.Logger):
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "extra": extra or {},
    }
    torch.save(payload, path)
    logger.info(f"[AutoSave] {os.path.basename(path)} (epoch={epoch})")

def load_ckpt(path: str, model: nn.Module, optimizer, device: str, logger: logging.Logger):
    if not os.path.exists(path):
        return 0, {}
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    logger.info(f"[Resume] {os.path.basename(path)} from epoch={ckpt['epoch']}")
    extra = ckpt.get("extra", {}) or {}
    return int(ckpt["epoch"]) + 1, extra


# =========================
# VQ-VAE blocks (LIGHT)
# =========================
class ResBlock3D(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        g = 8 if c >= 8 else 1
        self.net = nn.Sequential(
            nn.GroupNorm(g, c),
            nn.SiLU(),
            nn.Conv3d(c, c, 3, padding=1),
            nn.GroupNorm(g, c),
            nn.SiLU(),
            nn.Conv3d(c, c, 3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.commitment_cost = float(commitment_cost)

        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)

    def forward(self, inputs):
        # inputs: [B,C,D,H,W]
        B, C, D, H, W = inputs.shape
        assert C == self.embedding_dim

        x = inputs.permute(0, 2, 3, 4, 1).contiguous()   # [B,D,H,W,C]
        flat_x = x.view(-1, self.embedding_dim)          # [N,C]

        emb = self.embeddings.weight                     # [K,C]
        distances = (
            torch.sum(flat_x ** 2, dim=1, keepdim=True) +
            torch.sum(emb ** 2, dim=1) -
            2 * flat_x @ emb.t()
        )                                                 # [N,K]

        encoding_indices = torch.argmin(distances, dim=1)  # [N]
        quantized = self.embeddings(encoding_indices).view(B, D, H, W, C)

        e_latent = F.mse_loss(quantized.detach(), x)
        q_latent = F.mse_loss(quantized, x.detach())
        vq_loss = q_latent + self.commitment_cost * e_latent

        # straight-through
        quantized = x + (quantized - x).detach()
        quantized = quantized.permute(0, 4, 1, 2, 3).contiguous()

        enc_oh = F.one_hot(encoding_indices, self.num_embeddings).float()
        avg_probs = enc_oh.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, vq_loss, perplexity


class VQVAE256Down4Light(nn.Module):
    """
    down=4 only:
      256 -> 128 -> 64 (latent 64^3)
    max_ch is limited (<=96) to fit on 40GB.
    """
    def __init__(self, embedding_dim: int, num_embeddings: int, commitment_cost: float,
                 base_ch: int = 32, max_ch: int = 96):
        super().__init__()
        C = int(embedding_dim)
        b = int(base_ch)
        m = int(max_ch)

        c1 = min(b, m)            # 32
        c2 = min(b * 2, m)        # 64
        c3 = min(b * 3, m)        # 96  (max)

        # Encoder: 256 -> 128 -> 64
        self.encoder = nn.Sequential(
            nn.Conv3d(1, c1, 4, stride=2, padding=1), nn.SiLU(),   # 256->128
            ResBlock3D(c1),
            nn.Conv3d(c1, c2, 4, stride=2, padding=1), nn.SiLU(),  # 128->64
            ResBlock3D(c2),
            nn.Conv3d(c2, c3, 3, padding=1), nn.SiLU(),
            ResBlock3D(c3),
            nn.Conv3d(c3, C, 3, padding=1),  # -> embedding_dim
        )

        self.vq = VectorQuantizer(num_embeddings=num_embeddings, embedding_dim=C, commitment_cost=commitment_cost)

        # Decoder: 64 -> 128 -> 256
        self.decoder = nn.Sequential(
            nn.Conv3d(C, c3, 3, padding=1), nn.SiLU(),
            ResBlock3D(c3),

            nn.ConvTranspose3d(c3, c2, 4, stride=2, padding=1), nn.SiLU(),  # 64->128
            ResBlock3D(c2),

            nn.ConvTranspose3d(c2, c1, 4, stride=2, padding=1), nn.SiLU(),  # 128->256
            ResBlock3D(c1),

            nn.Conv3d(c1, max(16, c1 // 2), 3, padding=1), nn.SiLU(),
            nn.Conv3d(max(16, c1 // 2), 1, 3, padding=1),  # logits
        )

    def forward(self, x):
        z = self.encoder(x)
        z_q, vq_loss, ppl = self.vq(z)
        logits = self.decoder(z_q)
        prob = torch.sigmoid(logits)
        return prob, logits, vq_loss, ppl


# =========================
# DDPM: time embedding + FiLM conditioning on porosity
# =========================
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


# =========================
# Latent stats (per-channel mean/std)
# =========================
@torch.no_grad()
def compute_latent_stats(cfg: CFG, vqvae: VQVAE256Down4Light, dl: DataLoader, logger: logging.Logger):
    device = cfg.device
    vqvae.eval()

    # per-channel EMA mean/var
    C = cfg.embedding_dim
    mean = torch.zeros(C, device=device)
    var = torch.ones(C, device=device)

    batches = 0
    for x, _ in dl:
        x = x.to(device, non_blocking=True)
        z = vqvae.encoder(x)
        z_q, _, _ = vqvae.vq(z)  # [B,C,64,64,64]
        # mean/var over B,D,H,W -> per-channel
        bmean = z_q.float().mean(dim=(0,2,3,4))
        bvar = z_q.float().var(dim=(0,2,3,4), unbiased=False)

        if batches == 0:
            mean = bmean
            var = bvar
        else:
            mean = 0.9 * mean + 0.1 * bmean
            var = 0.9 * var + 0.1 * bvar

        batches += 1
        if batches >= cfg.stats_batches:
            break

    std = torch.sqrt(torch.clamp(var, min=1e-8))

    out = os.path.join(cfg.save_dir, "latent_stats.npz")
    np.savez_compressed(
        out,
        mean=mean.detach().cpu().numpy(),
        std=std.detach().cpu().numpy(),
        downsample_factor=cfg.downsample_factor,
        embedding_dim=cfg.embedding_dim,
        num_embeddings=cfg.num_embeddings,
        vae_base_ch=cfg.vae_base_ch,
        vae_max_ch=cfg.vae_max_ch,
    )
    logger.info(f"[STATS] saved per-channel latent mean/std -> {out}")
    logger.info(f"[STATS] mean[min,max]=[{float(mean.min()):.4f},{float(mean.max()):.4f}] "
                f"std[min,max]=[{float(std.min()):.4f},{float(std.max()):.4f}]")

def load_latent_stats(cfg: CFG, logger: logging.Logger):
    path = os.path.join(cfg.save_dir, "latent_stats.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"latent_stats not found: {path} (run vqvae stage to generate)")
    d = np.load(path)
    if int(d["downsample_factor"]) != int(cfg.downsample_factor):
        raise ValueError(f"latent_stats downsample mismatch: {int(d['downsample_factor'])} vs cfg {cfg.downsample_factor}")
    if int(d["embedding_dim"]) != int(cfg.embedding_dim):
        raise ValueError(f"latent_stats embedding_dim mismatch: {int(d['embedding_dim'])} vs cfg {cfg.embedding_dim}")
    if int(d["num_embeddings"]) != int(cfg.num_embeddings):
        raise ValueError(f"latent_stats num_embeddings mismatch: {int(d['num_embeddings'])} vs cfg {cfg.num_embeddings}")
    mean = torch.tensor(d["mean"], dtype=torch.float32, device=cfg.device)
    std = torch.tensor(d["std"], dtype=torch.float32, device=cfg.device)
    logger.info(f"[STATS] loaded per-channel mean/std: mean[min,max]=[{float(mean.min()):.4f},{float(mean.max()):.4f}] "
                f"std[min,max]=[{float(std.min()):.4f},{float(std.max()):.4f}]")
    return mean, std


# =========================
# Train VQ-VAE
# =========================
def train_vqvae(cfg: CFG, logger: logging.Logger):
    device = cfg.device

    ds = Rock256ProbDataset(cfg.raw_path, cfg.raw_shape, cfg.patch_size, cfg.n_samples, cfg.min_porosity, cfg.max_tries)
    dl = DataLoader(ds, batch_size=cfg.batch_size_vae, shuffle=True,
                    num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)

    vqvae = VQVAE256Down4Light(
        embedding_dim=cfg.embedding_dim,
        num_embeddings=cfg.num_embeddings,
        commitment_cost=cfg.commitment_cost,
        base_ch=cfg.vae_base_ch,
        max_ch=cfg.vae_max_ch
    ).to(device)

    opt = torch.optim.Adam(vqvae.parameters(), lr=cfg.lr_vae)

    # torch.amp.GradScaler is new; keep compatible
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=(cfg.amp and device == "cuda"))
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device == "cuda"))

    latest_path = os.path.join(cfg.save_dir, "vqvae_latest.pth")
    final_path = os.path.join(cfg.save_dir, "vqvae_final.pth")
    start_epoch, _ = load_ckpt(latest_path, vqvae, opt, device, logger)

    save_cfg_json(cfg, logger)

    latent_L = cfg.patch_size // cfg.downsample_factor
    logger.info("=== Stage: VQ-VAE (256->64 latent, LIGHT96) ===")
    logger.info(
        f"patch={cfg.patch_size} latent={latent_L}^3 down={cfg.downsample_factor} | "
        f"batch={cfg.batch_size_vae} | dim={cfg.embedding_dim} | K={cfg.num_embeddings} | "
        f"dice_w={cfg.dice_weight} pos_weight_mode={cfg.pos_weight_mode} amp={cfg.amp} | "
        f"vae_base_ch={cfg.vae_base_ch} vae_max_ch={cfg.vae_max_ch}"
    )

    step = 0
    for epoch in range(start_epoch, cfg.epochs_vae):
        vqvae.train()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()

        bce_sum = dice_sum = vq_sum = ppl_sum = 0.0
        poro_gt_sum = poro_pr_sum = 0.0
        n_batches = 0

        for x, poro_gt in dl:
            x = x.to(device, non_blocking=True)
            poro_gt = poro_gt.to(device, non_blocking=True)

            pos_w = make_pos_weight(cfg, poro_gt, device=device)

            opt.zero_grad(set_to_none=True)
            with cuda_autocast(cfg.amp and device == "cuda"):
                prob, logits, vq_loss, ppl = vqvae(x)
                bce = F.binary_cross_entropy_with_logits(logits, x, pos_weight=pos_w)
                dsc = soft_dice_loss(prob, x) if cfg.dice_weight > 0 else torch.tensor(0.0, device=device)
                loss = cfg.bce_weight * bce + cfg.dice_weight * dsc + vq_loss

            scaler.scale(loss).backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(vqvae.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()

            poro_pr = prob.mean(dim=(1,2,3,4))

            bce_sum += float(bce.detach().cpu())
            dice_sum += float(dsc.detach().cpu())
            vq_sum += float(vq_loss.detach().cpu())
            ppl_sum += float(ppl.detach().cpu())
            poro_gt_sum += float(poro_gt.mean().detach().cpu())
            poro_pr_sum += float(poro_pr.mean().detach().cpu())

            n_batches += 1
            step += 1

            if step % cfg.log_interval == 0:
                mem_gb = torch.cuda.max_memory_allocated() / (1024**3) if device == "cuda" else 0.0
                logger.info(
                    f"[VQ] epoch={epoch+1}/{cfg.epochs_vae} step={step} "
                    f"bce={bce.item():.4f} dice={float(dsc):.4f} vq={vq_loss.item():.4f} ppl={float(ppl):.1f} "
                    f"poro_gt={float(poro_gt.mean())*100:.2f}% poro_pr={float(poro_pr.mean())*100:.2f}% "
                    f"pos_w={float(pos_w.item()):.2f} mem={mem_gb:.2f}GB"
                )

        dt = time.time() - t0
        logger.info(
            f"[VQ] epoch_end {epoch+1}/{cfg.epochs_vae} "
            f"bce_mean={bce_sum/max(n_batches,1):.4f} dice_mean={dice_sum/max(n_batches,1):.4f} "
            f"vq_mean={vq_sum/max(n_batches,1):.4f} ppl_mean={ppl_sum/max(n_batches,1):.1f} "
            f"poro_gt_mean={poro_gt_sum/max(n_batches,1):.4f} poro_pr_mean={poro_pr_sum/max(n_batches,1):.4f} "
            f"time={dt/60:.2f}min"
        )

        if (epoch + 1) % cfg.save_interval == 0:
            extra = {"cfg": asdict(cfg)}
            save_ckpt(latest_path, vqvae, opt, epoch, extra=extra, logger=logger)

        if device == "cuda":
            torch.cuda.empty_cache()

    extra = {"cfg": asdict(cfg)}
    save_ckpt(final_path, vqvae, opt, cfg.epochs_vae - 1, extra=extra, logger=logger)
    logger.info("VQ-VAE training done.")

    compute_latent_stats(cfg, vqvae, dl, logger)


# =========================
# Train DDPM on normalized quantized embeddings
# =========================
def train_ddpm(cfg: CFG, logger: logging.Logger):
    device = cfg.device

    vqvae = VQVAE256Down4Light(
        embedding_dim=cfg.embedding_dim,
        num_embeddings=cfg.num_embeddings,
        commitment_cost=cfg.commitment_cost,
        base_ch=cfg.vae_base_ch,
        max_ch=cfg.vae_max_ch
    ).to(device)

    vqvae_final = os.path.join(cfg.save_dir, "vqvae_final.pth")
    if not os.path.exists(vqvae_final):
        raise FileNotFoundError(f"Need VQ-VAE first. Not found: {vqvae_final}")
    ckpt = torch.load(vqvae_final, map_location=device, weights_only=False)
    vqvae.load_state_dict(ckpt["model_state_dict"])
    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad = False

    lat_mean, lat_std = load_latent_stats(cfg, logger)

    ds = Rock256ProbDataset(cfg.raw_path, cfg.raw_shape, cfg.patch_size, cfg.n_samples, cfg.min_porosity, cfg.max_tries)
    dl = DataLoader(ds, batch_size=cfg.batch_size_ddpm, shuffle=True,
                    num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)

    unet = UNetLatentCond(cfg.embedding_dim, base=cfg.unet_base, time_dim=cfg.unet_time_dim).to(device)
    opt = torch.optim.Adam(unet.parameters(), lr=cfg.lr_ddpm)

    try:
        scaler = torch.amp.GradScaler("cuda", enabled=(cfg.amp and device == "cuda"))
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device == "cuda"))

    latest_path = os.path.join(cfg.save_dir, "unet_latest.pth")
    final_path = os.path.join(cfg.save_dir, "unet_final.pth")
    start_epoch, _ = load_ckpt(latest_path, unet, opt, device, logger)

    n_steps = cfg.n_steps
    betas = torch.linspace(cfg.beta_start, cfg.beta_end, n_steps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    latent_L = cfg.patch_size // cfg.downsample_factor
    logger.info("=== Stage: DDPM (latent 64^3, normalized, aligned) ===")
    logger.info(
        f"latent={latent_L}^3 | batch={cfg.batch_size_ddpm} | dim={cfg.embedding_dim} | "
        f"steps={cfg.n_steps} | amp={cfg.amp} | unet_base={cfg.unet_base} poro_clip={cfg.poro_clip} grad_clip={cfg.grad_clip}"
    )

    step = 0
    for epoch in range(start_epoch, cfg.epochs_ddpm):
        unet.train()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()

        loss_sum = 0.0
        poro_sum = 0.0
        n_batches = 0

        for x, poro in dl:
            x = x.to(device, non_blocking=True)
            poro = poro.to(device, non_blocking=True)

            with torch.no_grad():
                z = vqvae.encoder(x)
                z_q, _, _ = vqvae.vq(z)
                z_qn = (z_q - lat_mean[None, :, None, None, None]) / torch.clamp(
                    lat_std[None, :, None, None, None], min=1e-6
                )

            B = z_qn.size(0)
            t = torch.randint(0, n_steps, (B,), device=device, dtype=torch.long)
            noise = torch.randn_like(z_qn)

            sqrt_a = torch.sqrt(alphas_cumprod[t])[:, None, None, None, None]
            sqrt_om = torch.sqrt(1 - alphas_cumprod[t])[:, None, None, None, None]
            x_noisy = sqrt_a * z_qn + sqrt_om * noise

            poro_scaled = (poro - cfg.poro_center) / max(cfg.poro_scale, 1e-6)
            poro_scaled = poro_scaled.clamp(-cfg.poro_clip, cfg.poro_clip)

            opt.zero_grad(set_to_none=True)
            with cuda_autocast(cfg.amp and device == "cuda"):
                pred = unet(x_noisy, t, poro_scaled)
                loss = F.mse_loss(pred, noise)

            scaler.scale(loss).backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(unet.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()

            loss_sum += float(loss.detach().cpu())
            poro_sum += float(poro.mean().detach().cpu())
            n_batches += 1
            step += 1

            if step % cfg.log_interval == 0:
                mem_gb = torch.cuda.max_memory_allocated() / (1024**3) if device == "cuda" else 0.0
                logger.info(
                    f"[DDPM] epoch={epoch+1}/{cfg.epochs_ddpm} step={step} "
                    f"loss={loss.item():.4f} poro={float(poro.mean())*100:.2f}% mem={mem_gb:.2f}GB"
                )

        dt = time.time() - t0
        logger.info(
            f"[DDPM] epoch_end {epoch+1}/{cfg.epochs_ddpm} "
            f"loss_mean={loss_sum/max(n_batches,1):.4f} "
            f"poro_mean={poro_sum/max(n_batches,1):.4f} "
            f"time={dt/60:.2f}min"
        )

        # ===== Overwrite the latest checkpoint =====
        if (epoch + 1) % cfg.save_interval == 0:
            extra = {"cfg": asdict(cfg)}
            save_ckpt(latest_path, unet, opt, epoch, extra=extra, logger=logger)

        # ===== Save a persistent checkpoint every 50 epochs =====
        ep = epoch + 1
        if ep % 50 == 0:
            extra = {"cfg": asdict(cfg)}
            save50_path = os.path.join(cfg.save_dir, f"unet_ep{ep:03d}.pth")
            save_ckpt(save50_path, unet, opt, epoch, extra=extra, logger=logger)
            logger.info(f"[Save50] saved {save50_path}")

        if device == "cuda":
            torch.cuda.empty_cache()

    # ===== Save the final checkpoint after training =====
    extra = {"cfg": asdict(cfg)}
    save_ckpt(final_path, unet, opt, cfg.epochs_ddpm - 1, extra=extra, logger=logger)
    logger.info("DDPM training done.")


# =========================
# Sample ONE 256^3 (DDPM -> denorm -> VQ-VAE decoder)
# =========================
@torch.no_grad()
def sample_one(cfg: CFG, logger: logging.Logger):
    device = cfg.device
    torch.manual_seed(cfg.sample_seed)
    np.random.seed(cfg.sample_seed)

    # load VQ-VAE
    vqvae = VQVAE256Down4Light(
        embedding_dim=cfg.embedding_dim,
        num_embeddings=cfg.num_embeddings,
        commitment_cost=cfg.commitment_cost,
        base_ch=cfg.vae_base_ch,
        max_ch=cfg.vae_max_ch
    ).to(device)
    vqvae_final = os.path.join(cfg.save_dir, "vqvae_final.pth")
    if not os.path.exists(vqvae_final):
        raise FileNotFoundError(f"vqvae_final not found: {vqvae_final}")
    ckpt = torch.load(vqvae_final, map_location=device, weights_only=False)
    vqvae.load_state_dict(ckpt["model_state_dict"])
    vqvae.eval()

    # latent stats
    lat_mean, lat_std = load_latent_stats(cfg, logger)

    # load UNet
    unet = UNetLatentCond(cfg.embedding_dim, base=cfg.unet_base, time_dim=cfg.unet_time_dim).to(device)
    unet_final = os.path.join(cfg.save_dir, "unet_final.pth")
    if not os.path.exists(unet_final):
        raise FileNotFoundError(f"unet_final not found: {unet_final} (train ddpm first)")
    logger.info(f"[SAMPLE] loading UNet from: {unet_final}")
    ckpt = torch.load(unet_final, map_location=device, weights_only=False)
    unet.load_state_dict(ckpt["model_state_dict"])
    unet.eval()

    n_steps = cfg.n_steps
    betas = torch.linspace(cfg.beta_start, cfg.beta_end, n_steps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    L = cfg.patch_size // cfg.downsample_factor  # 64
    x = torch.randn(1, cfg.embedding_dim, L, L, L, device=device)  # normalized latent space
    poro = torch.tensor([float(cfg.target_porosity)], device=device, dtype=torch.float32)
    poro_scaled = (poro - cfg.poro_center) / max(cfg.poro_scale, 1e-6)
    poro_scaled = poro_scaled.clamp(-cfg.poro_clip, cfg.poro_clip)

    logger.info(f"=== Stage: SAMPLE ONE === target_porosity={cfg.target_porosity:.4f} latent={L}^3")

    for i in range(n_steps - 1, -1, -1):
        t = torch.full((1,), i, device=device, dtype=torch.long)
        eps = unet(x, t, poro_scaled)

        if cfg.eps_clip > 0:
            eps = eps.clamp(-cfg.eps_clip, cfg.eps_clip)

        alpha = alphas[i]
        alpha_hat = alphas_cumprod[i]
        beta = betas[i]
        z = torch.randn_like(x) if i > 0 else torch.zeros_like(x)

        x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * eps) + torch.sqrt(beta) * z
        if cfg.clamp_x > 0:
            x = x.clamp(-cfg.clamp_x, cfg.clamp_x)

        if i % 100 == 0:
            logger.info(
                f"[SAMPLE] step={i}/{n_steps} | x_abs_max={float(x.abs().max()):.3f} "
                f"| eps_abs_max={float(eps.abs().max()):.3f} eps_std={float(eps.std()):.3f}"
            )

    # denormalize to VQ embedding space
    x_denorm = x * torch.clamp(lat_std[None, :, None, None, None], min=1e-6) + lat_mean[None, :, None, None, None]

    # decode
       # --- Project back to the codebook ---
    z_q2, _, _ = vqvae.vq(x_denorm)
    logits = vqvae.decoder(z_q2)
    prob = torch.sigmoid(logits)
    prob_np = prob[0, 0].float().cpu().numpy()

    # --- Stable rank-based threshold ---
    flat = prob_np.reshape(-1)
    N = flat.size

    target_p = float(cfg.target_porosity)
    target_cnt = int(round(target_p * N))

    k = N - target_cnt
    k = np.clip(k, 0, N - 1)

    thr = np.partition(flat, k)[k]

    seg = (prob_np > thr).astype(np.uint8)

    cur_cnt = int(seg.sum())
    need = target_cnt - cur_cnt
    if need > 0:
        eq = (prob_np == thr).reshape(-1)
        idx = np.flatnonzero(eq)
        if idx.size > 0:
            np.random.shuffle(idx)
            seg.reshape(-1)[idx[:need]] = 1
    seg_poro = float(seg.mean())

    out_npz = os.path.join(cfg.save_dir, f"gen256_tp{cfg.target_porosity:.3f}.npz".replace(".", "p"))
    out_raw = os.path.join(cfg.save_dir, f"gen256_tp{cfg.target_porosity:.3f}.raw".replace(".", "p"))
    np.savez_compressed(out_npz, prob=prob_np, seg=seg, target_porosity=float(cfg.target_porosity), seg_porosity=seg_poro)
    seg.tofile(out_raw)

    # visualize slices
    vis_dir = os.path.join(cfg.save_dir, "sample_vis")
    os.makedirs(vis_dir, exist_ok=True)
    mid = cfg.patch_size // 2
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12,4))
    ax=plt.subplot(1,3,1); ax.set_title("prob (XY mid)"); ax.imshow(prob_np[:,:,mid], cmap="gray", vmin=0, vmax=1); ax.axis("off")
    ax=plt.subplot(1,3,2); ax.set_title("seg (XY mid)"); ax.imshow(seg[:,:,mid], cmap="gray", vmin=0, vmax=1); ax.axis("off")
    ax=plt.subplot(1,3,3); ax.set_title("seg (XZ mid)"); ax.imshow(seg[:,mid,:], cmap="gray", vmin=0, vmax=1); ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(vis_dir, "slices.png"), dpi=150)
    plt.close(fig)

    logger.info(f"[VIS] saved slices to: {vis_dir}")
    logger.info("SAMPLE DONE (one 256^3).")
    logger.info(f"target_porosity={cfg.target_porosity:.4f} | seg_porosity={seg_poro:.4f}")
    logger.info(f"saved: {out_npz}")
    logger.info(f"saved: {out_raw} (uint8 0=solid/1=pore, 256^3)")


# =========================
# Main
# =========================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", type=str, default="all", choices=["vqvae", "ddpm", "sample", "all"])

    p.add_argument("--raw_path", type=str, default="S1.raw")
    p.add_argument("--raw_shape", type=int, nargs=3, default=[800, 800, 800])
    p.add_argument("--save_dir", type=str, default="./ckpt_lat64_vq80_ddpm150_light96")
    p.add_argument("--device", type=str, default="cuda")

    # fixed down=4
    p.add_argument("--down", type=int, default=4, choices=[4])

    # train knobs
    p.add_argument("--target_porosity", type=float, default=0.13)
    p.add_argument("--batch_vae", type=int, default=1)
    p.add_argument("--batch_ddpm", type=int, default=1)
    p.add_argument("--epochs_vae", type=int, default=80)
    p.add_argument("--epochs_ddpm", type=int, default=150)
    p.add_argument("--min_porosity", type=float, default=0.0)
    p.add_argument("--n_samples", type=int, default=1000)

    p.add_argument("--log_interval", type=int, default=20)
    p.add_argument("--save_interval", type=int, default=5)

    p.add_argument("--amp", action="store_true")
    p.add_argument("--no_amp", action="store_true")

    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--sample_seed", type=int, default=0)

    p.add_argument("--eps_clip", type=float, default=5.0)
    p.add_argument("--clamp_x", type=float, default=15.0)

    p.add_argument("--poro_center", type=float, default=0.13)
    p.add_argument("--poro_scale", type=float, default=0.02)
    p.add_argument("--poro_clip", type=float, default=5.0)

    p.add_argument("--dice_weight", type=float, default=0.2)
    p.add_argument("--pos_weight_mode", type=str, default="batch", choices=["fixed", "batch"])
    p.add_argument("--use_quantile_threshold", action="store_true")

    # widths
    p.add_argument("--vae_base_ch", type=int, default=32)
    p.add_argument("--vae_max_ch", type=int, default=96)

    p.add_argument("--unet_base", type=int, default=64)
    p.add_argument("--unet_time_dim", type=int, default=128)

    p.add_argument("--grad_clip", type=float, default=1.0)

    return p.parse_args()

def main():
    args = parse_args()
    cfg = CFG()

    cfg.raw_path = args.raw_path
    cfg.raw_shape = tuple(args.raw_shape)
    cfg.save_dir = args.save_dir
    cfg.device = args.device
    cfg.downsample_factor = args.down

    cfg.target_porosity = args.target_porosity
    cfg.batch_size_vae = args.batch_vae
    cfg.batch_size_ddpm = args.batch_ddpm
    cfg.epochs_vae = args.epochs_vae
    cfg.epochs_ddpm = args.epochs_ddpm
    cfg.min_porosity = args.min_porosity
    cfg.n_samples = args.n_samples
    cfg.log_interval = args.log_interval
    cfg.save_interval = args.save_interval
    cfg.seed = args.seed
    cfg.sample_seed = args.sample_seed
    cfg.eps_clip = args.eps_clip
    cfg.clamp_x = args.clamp_x
    cfg.poro_center = args.poro_center
    cfg.poro_scale = args.poro_scale
    cfg.poro_clip = args.poro_clip
    cfg.dice_weight = args.dice_weight
    cfg.pos_weight_mode = args.pos_weight_mode
    cfg.use_quantile_threshold = bool(args.use_quantile_threshold)

    cfg.vae_base_ch = args.vae_base_ch
    cfg.vae_max_ch = args.vae_max_ch
    cfg.unet_base = args.unet_base
    cfg.unet_time_dim = args.unet_time_dim
    cfg.grad_clip = args.grad_clip

    if args.no_amp:
        cfg.amp = False
    elif args.amp:
        cfg.amp = True

    os.makedirs(cfg.save_dir, exist_ok=True)
    logger = setup_logger(cfg.save_dir)

    logger.info("==== RUN CONFIG ====")
    logger.info(str(cfg))

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if cfg.device == "cuda" and torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    else:
        logger.info("Running on CPU (slow).")

    try:
        if args.stage == "vqvae":
            train_vqvae(cfg, logger)
        elif args.stage == "ddpm":
            train_ddpm(cfg, logger)
        elif args.stage == "sample":
            sample_one(cfg, logger)
        else:
            train_vqvae(cfg, logger)
            train_ddpm(cfg, logger)
            sample_one(cfg, logger)

        logger.info("ALL DONE.")
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except Exception as e:
        logger.error(f"CRASH: {e}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()
