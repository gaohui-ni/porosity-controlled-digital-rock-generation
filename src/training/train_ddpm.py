import os
import time
import logging
from contextlib import nullcontext
from dataclasses import asdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.config import CFG
from src.data.dataset import Rock256ProbDataset
from src.models.vqvae3d import VQVAE256Down4Light
from src.models.unet3d_film import UNetLatentCond
from src.training.latent_stats import load_latent_stats
from src.utils.checkpoint import load_ckpt, save_ckpt

def cuda_autocast(enabled: bool):
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp"):
        return torch.amp.autocast(device_type="cuda", enabled=True)
    return torch.cuda.amp.autocast(enabled=True)

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
