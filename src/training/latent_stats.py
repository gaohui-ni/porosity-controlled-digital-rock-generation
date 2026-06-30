import os
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import CFG
from src.models.vqvae3d import VQVAE256Down4Light

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
