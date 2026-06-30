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
from src.training.latent_stats import compute_latent_stats
from src.utils.checkpoint import load_ckpt, save_ckpt, save_cfg_json
from src.utils.losses import make_pos_weight, soft_dice_loss

def cuda_autocast(enabled: bool):
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp"):
        return torch.amp.autocast(device_type="cuda", enabled=True)
    return torch.cuda.amp.autocast(enabled=True)

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
