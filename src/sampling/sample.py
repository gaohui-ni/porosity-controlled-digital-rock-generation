import os
import logging

import numpy as np
import torch

from src.config import CFG
from src.models.vqvae3d import VQVAE256Down4Light
from src.models.unet3d_film import UNetLatentCond
from src.training.latent_stats import load_latent_stats

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
