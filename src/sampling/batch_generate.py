import csv
import json
import os
from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional

import numpy as np
import torch

from src.config import CFG
from src.models.vqvae3d import VQVAE256Down4Light
from src.models.unet3d_film import UNetLatentCond
from src.training.latent_stats import load_latent_stats
from src.sampling.quantile_binarization import quantile_binarize


def phi_tag(phi: float) -> str:
    return f"{float(phi):.4f}".rstrip("0").rstrip(".").replace(".", "p")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_sampling_models(cfg: CFG, logger):
    """
    Load trained VQ-VAE, latent statistics, and FiLM-conditioned latent DDPM.

    Expected files under cfg.save_dir:
      - vqvae_final.pth
      - unet_final.pth
      - latent_stats.npz
    """
    device = cfg.device

    vqvae = VQVAE256Down4Light(
        embedding_dim=cfg.embedding_dim,
        num_embeddings=cfg.num_embeddings,
        commitment_cost=cfg.commitment_cost,
        base_ch=cfg.vae_base_ch,
        max_ch=cfg.vae_max_ch,
    ).to(device)

    vqvae_final = os.path.join(cfg.save_dir, "vqvae_final.pth")
    if not os.path.exists(vqvae_final):
        raise FileNotFoundError(f"vqvae_final not found: {vqvae_final}")
    ckpt = torch.load(vqvae_final, map_location=device, weights_only=False)
    vqvae.load_state_dict(ckpt["model_state_dict"])
    vqvae.eval()

    lat_mean, lat_std = load_latent_stats(cfg, logger)

    unet = UNetLatentCond(cfg.embedding_dim, base=cfg.unet_base, time_dim=cfg.unet_time_dim).to(device)
    unet_final = os.path.join(cfg.save_dir, "unet_final.pth")
    if not os.path.exists(unet_final):
        raise FileNotFoundError(f"unet_final not found: {unet_final}")
    logger.info(f"[BATCH] loading UNet from: {unet_final}")
    ckpt = torch.load(unet_final, map_location=device, weights_only=False)
    unet.load_state_dict(ckpt["model_state_dict"])
    unet.eval()

    return vqvae, unet, lat_mean, lat_std


def make_ddpm_schedule(cfg: CFG):
    betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.n_steps, device=cfg.device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_cumprod


@torch.no_grad()
def sample_probability_volume(
    cfg: CFG,
    vqvae: VQVAE256Down4Light,
    unet: UNetLatentCond,
    lat_mean: torch.Tensor,
    lat_std: torch.Tensor,
    target_porosity: float,
    seed: int,
    logger=None,
) -> np.ndarray:
    """Generate one 3D probability volume by latent DDPM sampling."""
    device = cfg.device
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    betas, alphas, alphas_cumprod = make_ddpm_schedule(cfg)

    L = int(cfg.patch_size // cfg.downsample_factor)
    x = torch.randn(1, cfg.embedding_dim, L, L, L, device=device)

    poro = torch.tensor([float(target_porosity)], device=device, dtype=torch.float32)
    poro_scaled = (poro - cfg.poro_center) / max(cfg.poro_scale, 1e-6)
    poro_scaled = poro_scaled.clamp(-cfg.poro_clip, cfg.poro_clip)

    if logger is not None:
        logger.info(f"[BATCH] sampling seed={seed} target={target_porosity:.4f} latent={L}^3")

    for i in range(cfg.n_steps - 1, -1, -1):
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

    x_denorm = x * torch.clamp(lat_std[None, :, None, None, None], min=1e-6) + lat_mean[None, :, None, None, None]
    z_q2, _, _ = vqvae.vq(x_denorm)
    logits = vqvae.decoder(z_q2)
    prob = torch.sigmoid(logits)
    return prob[0, 0].float().cpu().numpy()


def save_sample_outputs(
    prob_np: np.ndarray,
    seg: np.ndarray,
    out_dir: Path,
    target_porosity: float,
    seed: int,
    threshold: float,
    seg_porosity: float,
    save_prob: bool = True,
    save_raw: bool = True,
) -> Dict[str, Any]:
    tag = phi_tag(target_porosity)
    sample_id = f"gen256_phi{tag}_seed{int(seed):04d}"
    npz_path = out_dir / f"{sample_id}.npz"
    raw_path = out_dir / f"{sample_id}.raw"

    if save_prob:
        np.savez_compressed(
            npz_path,
            prob=prob_np.astype(np.float32),
            seg=seg.astype(np.uint8),
            target_porosity=float(target_porosity),
            seg_porosity=float(seg_porosity),
            threshold=float(threshold),
            seed=int(seed),
        )
    else:
        np.savez_compressed(
            npz_path,
            seg=seg.astype(np.uint8),
            target_porosity=float(target_porosity),
            seg_porosity=float(seg_porosity),
            threshold=float(threshold),
            seed=int(seed),
        )

    if save_raw:
        seg.astype(np.uint8).tofile(raw_path)

    return {
        "sample_id": sample_id,
        "target_phi": float(target_porosity),
        "phi_tag": f"phi{tag}",
        "seed": int(seed),
        "threshold": float(threshold),
        "seg_porosity": float(seg_porosity),
        "porosity_abs_error": float(abs(seg_porosity - target_porosity)),
        "npz_path": str(npz_path),
        "raw_path": str(raw_path) if save_raw else "",
    }


@torch.no_grad()
def batch_generate(
    cfg: CFG,
    targets: Iterable[float],
    seeds: Iterable[int],
    out_root: str | Path,
    logger,
    save_prob: bool = True,
    save_raw: bool = True,
) -> List[Dict[str, Any]]:
    """
    Batch generation for paper experiments.

    Folder layout:
      out_root/phi0p11/gen256_phi0p11_seed0000.npz
      out_root/phi0p11/gen256_phi0p11_seed0000.raw
      out_root/phi0p11/metadata_phi0p11.csv/json
      out_root/all_metadata.csv/json
    """
    out_root = ensure_dir(out_root)
    targets = [float(t) for t in targets]
    seeds = [int(s) for s in seeds]

    vqvae, unet, lat_mean, lat_std = load_sampling_models(cfg, logger)

    all_rows: List[Dict[str, Any]] = []

    for target in targets:
        tag = phi_tag(target)
        target_dir = ensure_dir(out_root / f"phi{tag}")
        rows: List[Dict[str, Any]] = []

        logger.info("=" * 80)
        logger.info(f"[BATCH] target phi={target:.4f}, n={len(seeds)}, out={target_dir}")
        logger.info("=" * 80)

        for idx, seed in enumerate(seeds, start=1):
            logger.info(f"[BATCH] target={target:.4f} sample {idx}/{len(seeds)} seed={seed}")
            prob_np = sample_probability_volume(
                cfg=cfg,
                vqvae=vqvae,
                unet=unet,
                lat_mean=lat_mean,
                lat_std=lat_std,
                target_porosity=target,
                seed=seed,
                logger=logger,
            )
            seg, thr, seg_poro = quantile_binarize(prob_np, target, seed=seed)
            row = save_sample_outputs(
                prob_np=prob_np,
                seg=seg,
                out_dir=target_dir,
                target_porosity=target,
                seed=seed,
                threshold=thr,
                seg_porosity=seg_poro,
                save_prob=save_prob,
                save_raw=save_raw,
            )
            rows.append(row)
            all_rows.append(row)
            logger.info(
                f"[BATCH] saved {row['sample_id']} | target={target:.6f} "
                f"seg_phi={seg_poro:.6f} abs_err={row['porosity_abs_error']:.6e}"
            )

        # per-target metadata
        csv_path = target_dir / f"metadata_phi{tag}.csv"
        json_path = target_dir / f"metadata_phi{tag}.json"
        if rows:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(rows, f, indent=2, ensure_ascii=False)

            summary = {
                "target_phi": target,
                "n_samples": len(rows),
                "seg_porosity_mean": float(np.mean([r["seg_porosity"] for r in rows])),
                "seg_porosity_std": float(np.std([r["seg_porosity"] for r in rows])),
                "abs_error_mean": float(np.mean([r["porosity_abs_error"] for r in rows])),
                "abs_error_max": float(np.max([r["porosity_abs_error"] for r in rows])),
            }
            with open(target_dir / f"summary_phi{tag}.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

    # global metadata
    if all_rows:
        with open(out_root / "all_metadata.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        with open(out_root / "all_metadata.json", "w", encoding="utf-8") as f:
            json.dump(all_rows, f, indent=2, ensure_ascii=False)

    logger.info(f"[BATCH] done. all outputs saved under: {out_root.resolve()}")
    return all_rows
