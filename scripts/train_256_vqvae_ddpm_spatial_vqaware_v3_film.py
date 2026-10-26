# Public spatial EMA-VQ-VAE / FiLM-DDPM implementation.
#
# Verified user-facing differences that matter for released-weight sampling
# are the configurable checkpoint directory, raw-data path, and UNet loading
# behavior. Historical claims about an unavailable parent file, an exact diff
# hunk count, or all other sections being unchanged are intentionally omitted;
# they cannot be reproduced from files bundled in this repository.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_256_vqvae_ddpm_spatial_vqaware_v2.py

Purpose
-------
Revised training pipeline for:
1) strict spatial Train / Validation / Test separation;
2) structure-preserving VQ-VAE;
3) FiLM-conditioned latent DDPM;
4) optional codebook-manifold consistency during diffusion training;
5) AQB-OFF / AQB-ON paired sampling;
6) matched ablation interfaces:
      variant = vqaware / vq / continuous
      condition_mode = film / none

Raw data
--------
S1.raw
shape = (800, 800, 800)
dtype = uint8
0 = solid
1 = pore

Frozen spatial split
--------------------
Train:
    x=[0,512), y=[0,800), z=[0,512)

Validation:
    x=[0,512), y=[0,800), z=[544,800)

Test:
    x=[544,800), y=[0,800), z=[0,800)

Important:
- Test is NEVER instantiated by training functions.
- Validation is used only for checkpoint selection.
- Final Test evaluation should be done in a separate script after model freeze.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import csv
import json
import logging
import math
import random
import time
import traceback
from contextlib import nullcontext
from dataclasses import dataclass, asdict
from typing import Tuple, Optional, List

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1. Logging
# ============================================================

def setup_logger(save_dir: str, name: str = "train") -> logging.Logger:
    os.makedirs(save_dir, exist_ok=True)

    log_path = os.path.join(save_dir, "train.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)

    logger.addHandler(fh)
    logger.addHandler(sh)

    logger.info(f"Logger initialized: {log_path}")
    return logger


# ============================================================
# 2. Config
# ============================================================

@dataclass
class CFG:

    # ---------------- data ----------------
    raw_path: str = "data/raw/Bei_800x800x800.raw"
    raw_shape: Tuple[int, int, int] = (800, 800, 800)
    patch_size: int = 256

    n_train_samples: int = 1000
    n_val_samples: int = 64

    min_porosity: float = 0.0
    max_tries: int = 50

    # ---------------- spatial split ----------------
    train_region: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]] = (
        (0, 512),
        (0, 800),
        (0, 512),
    )

    val_region: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]] = (
        (0, 512),
        (0, 800),
        (544, 800),
    )

    test_region: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]] = (
        (544, 800),
        (0, 800),
        (0, 800),
    )

    # ---------------- experiment ----------------
    # vqaware   : VQ-VAE + DDPM + codebook consistency
    # vq        : VQ-VAE + DDPM, no codebook consistency
    # continuous: continuous AE + DDPM
    variant: str = "vqaware"

    # film / none
    condition_mode: str = "film"

    # ---------------- latent ----------------
    downsample_factor: int = 4

    embedding_dim: int = 32
    num_embeddings: int = 1024

    # ---------------- VQ ----------------
    commitment_cost: float = 0.25

    ema_decay: float = 0.99
    ema_eps: float = 1e-5

    vq_chunk_size: int = 8192

    dead_code_threshold: float = 0.10
    dead_code_warmup: int = 300
    dead_code_reset_interval: int = 200
    dead_code_reset_max: int = 64

    # ---------------- AE/VQ-VAE ----------------
    lr_vae: float = 2e-4
    epochs_vae: int = 80
    batch_size_vae: int = 1

    vae_base_ch: int = 32
    vae_max_ch: int = 96

    # reconstruction losses
    bce_weight: float = 1.0
    dice_weight: float = 0.20

    # NEW structure-preserving terms
    local_phi_weight: float = 0.10
    boundary_weight: float = 0.05
    local_phi_kernel: int = 16

    pos_weight_mode: str = "batch"
    poro_center: float = 0.13
    pos_weight_clip: float = 20.0

    # ---------------- DDPM ----------------
    lr_ddpm: float = 1e-4
    epochs_ddpm: int = 150
    batch_size_ddpm: int = 1

    n_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02

    unet_base: int = 64
    unet_time_dim: int = 128

    # porosity conditioning
    poro_scale: float = 0.02
    poro_clip: float = 5.0

    # NEW codebook-manifold consistency
    lambda_codebook: float = 0.10
    codebook_sample_voxels: int = 8192

    # ---------------- latent normalization ----------------
    stats_batches: int = 80

    # ---------------- runtime ----------------
    amp: bool = True

    save_interval: int = 5
    log_interval: int = 20

    save_dir: str = "./ckpt_spatial_vqaware_v2"

    device: str = "cuda"
    num_workers: int = 0
    pin_memory: bool = True

    seed: int = 20260928

    grad_clip: float = 1.0

    # abort a run after this many CONSECUTIVE non-finite batches
    nonfinite_abort: int = 20

    # stage tag, used only for provenance in the run log / manifest
    stage_tag: str = ""

    # ---------------- sampling ----------------
    sample_targets: str = "0.11,0.12,0.13,0.14"
    samples_per_target: int = 5

    # Cross-arm model selection uses validation noise loss because it is
    # defined identically for vq, vqaware, and continuous variants.
    sample_checkpoint: str = "unet_best_noise.pth"

    clamp_x: float = 15.0
    eps_clip: float = 5.0

    fixed_threshold: float = 0.5


# ============================================================
# 3. Split utilities
# ============================================================

def interval_overlap(a, b):
    return max(a[0], b[0]) < min(a[1], b[1])


def regions_overlap(r1, r2):
    return (
        interval_overlap(r1[0], r2[0])
        and interval_overlap(r1[1], r2[1])
        and interval_overlap(r1[2], r2[2])
    )


def validate_split(cfg: CFG, logger: logging.Logger):

    assert not regions_overlap(cfg.train_region, cfg.val_region), \
        "Train and Validation overlap!"

    assert not regions_overlap(cfg.train_region, cfg.test_region), \
        "Train and Test overlap!"

    assert not regions_overlap(cfg.val_region, cfg.test_region), \
        "Validation and Test overlap!"

    for name, region in [
        ("train", cfg.train_region),
        ("val", cfg.val_region),
        ("test", cfg.test_region),
    ]:
        for axis in range(3):
            lo, hi = region[axis]
            assert hi - lo >= cfg.patch_size, (
                f"{name} axis={axis} smaller than patch size"
            )

    logger.info("=== SPATIAL SPLIT VERIFIED ===")
    logger.info(f"Train = {cfg.train_region}")
    logger.info(f"Val   = {cfg.val_region}")
    logger.info(f"Test  = {cfg.test_region}")
    logger.info("Train / Val / Test are strictly disjoint.")


def save_split_manifest(cfg: CFG):

    out = os.path.join(cfg.save_dir, "spatial_split.json")

    payload = {
        "raw_shape": cfg.raw_shape,
        "patch_size": cfg.patch_size,
        "train_region": cfg.train_region,
        "validation_region": cfg.val_region,
        "test_region": cfg.test_region,
        "note": (
            "Test region is frozen and is not used in VQ-VAE training, "
            "latent statistics, DDPM training, or checkpoint selection."
        )
    }

    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


# ============================================================
# 4. Dataset
# ============================================================

class RockRegionDataset(Dataset):

    def __init__(
        self,
        raw_path: str,
        raw_shape,
        region,
        patch_size: int,
        n_samples: int,
        min_porosity: float = 0.0,
        max_tries: int = 50,
        fixed_seed: Optional[int] = None,
    ):

        if not os.path.exists(raw_path):
            raise FileNotFoundError(raw_path)

        self.raw_shape = tuple(raw_shape)
        self.region = region

        self.patch_size = int(patch_size)
        self.n_samples = int(n_samples)

        self.min_porosity = float(min_porosity)
        self.max_tries = int(max_tries)

        self.mm = np.memmap(
            raw_path,
            dtype=np.uint8,
            mode="r",
            shape=self.raw_shape
        )

        self.fixed_coords = None

        # Validation coordinates are frozen once.
        if fixed_seed is not None:
            rng = np.random.default_rng(fixed_seed)

            coords = []

            for _ in range(self.n_samples):
                coords.append(self._random_coord(rng))

            self.fixed_coords = coords

    def __len__(self):
        return self.n_samples

    def _random_coord(self, rng=None):

        if rng is None:
            rng = np.random

        S = self.patch_size

        coords = []

        for lo, hi in self.region:

            max_start = hi - S

            if max_start < lo:
                raise RuntimeError(
                    f"Region {self.region} cannot hold patch={S}"
                )

            if max_start == lo:
                s = lo
            else:
                if isinstance(rng, np.random.Generator):
                    s = int(rng.integers(lo, max_start + 1))
                else:
                    s = int(rng.randint(lo, max_start + 1))

            coords.append(s)

        return tuple(coords)

    def _read_patch(self, coord):

        x0, y0, z0 = coord
        S = self.patch_size

        patch = np.asarray(
            self.mm[
                x0:x0 + S,
                y0:y0 + S,
                z0:z0 + S
            ],
            dtype=np.uint8
        ).copy()

        return patch

    def __getitem__(self, idx):

        # Validation = fixed spatial windows.
        if self.fixed_coords is not None:

            coord = self.fixed_coords[idx]
            patch = self._read_patch(coord)

            poro = float(np.mean(patch == 1))

            x = torch.from_numpy(
                patch.astype(np.float32)
            ).unsqueeze(0)

            return (
                x,
                torch.tensor(poro, dtype=torch.float32),
                torch.tensor(coord, dtype=torch.int32)
            )

        # Training = dynamic random cropping from TRAIN REGION ONLY.
        for i in range(self.max_tries):

            coord = self._random_coord()
            patch = self._read_patch(coord)

            poro = float(np.mean(patch == 1))

            if poro >= self.min_porosity or i == self.max_tries - 1:

                x = torch.from_numpy(
                    patch.astype(np.float32)
                ).unsqueeze(0)

                return (
                    x,
                    torch.tensor(poro, dtype=torch.float32),
                    torch.tensor(coord, dtype=torch.int32)
                )


def build_loaders(cfg: CFG, stage: str):

    train_ds = RockRegionDataset(
        raw_path=cfg.raw_path,
        raw_shape=cfg.raw_shape,
        region=cfg.train_region,
        patch_size=cfg.patch_size,
        n_samples=cfg.n_train_samples,
        min_porosity=cfg.min_porosity,
        max_tries=cfg.max_tries,
        fixed_seed=None
    )

    # frozen validation crop positions
    val_ds = RockRegionDataset(
        raw_path=cfg.raw_path,
        raw_shape=cfg.raw_shape,
        region=cfg.val_region,
        patch_size=cfg.patch_size,
        n_samples=cfg.n_val_samples,
        min_porosity=0.0,
        max_tries=1,
        fixed_seed=cfg.seed + 1000
    )

    if stage == "vae":
        bs = cfg.batch_size_vae
    else:
        bs = cfg.batch_size_ddpm

    train_dl = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory
    )

    return train_dl, val_dl


# ============================================================
# 5. Losses
# ============================================================

def soft_dice_loss(prob, target, eps=1e-6):

    num = 2.0 * (prob * target).sum(dim=(1, 2, 3, 4))

    den = (
        prob.sum(dim=(1, 2, 3, 4))
        + target.sum(dim=(1, 2, 3, 4))
        + eps
    )

    return 1.0 - (num / den).mean()


def dice_score_binary(pred, target, eps=1e-6):

    pred = pred.float()
    target = target.float()

    inter = (pred * target).sum(dim=(1, 2, 3, 4))

    den = (
        pred.sum(dim=(1, 2, 3, 4))
        + target.sum(dim=(1, 2, 3, 4))
    )

    return ((2 * inter + eps) / (den + eps)).mean()


def local_porosity_loss(prob, target, kernel=16):

    p1 = F.avg_pool3d(
        prob,
        kernel_size=kernel,
        stride=kernel
    )

    p2 = F.avg_pool3d(
        target,
        kernel_size=kernel,
        stride=kernel
    )

    return F.l1_loss(p1, p2)


def boundary_gradient_loss(prob, target):

    losses = []

    # x
    gp = prob[:, :, 1:, :, :] - prob[:, :, :-1, :, :]
    gt = target[:, :, 1:, :, :] - target[:, :, :-1, :, :]
    losses.append(F.l1_loss(gp.abs(), gt.abs()))

    # y
    gp = prob[:, :, :, 1:, :] - prob[:, :, :, :-1, :]
    gt = target[:, :, :, 1:, :] - target[:, :, :, :-1, :]
    losses.append(F.l1_loss(gp.abs(), gt.abs()))

    # z
    gp = prob[:, :, :, :, 1:] - prob[:, :, :, :, :-1]
    gt = target[:, :, :, :, 1:] - target[:, :, :, :, :-1]
    losses.append(F.l1_loss(gp.abs(), gt.abs()))

    return sum(losses) / 3.0


def boundary_density_binary(x):

    x = x.float()

    dx = (x[:, :, 1:] != x[:, :, :-1]).float().mean()
    dy = (x[:, :, :, 1:] != x[:, :, :, :-1]).float().mean()
    dz = (x[:, :, :, :, 1:] != x[:, :, :, :, :-1]).float().mean()

    return (dx + dy + dz) / 3.0


def make_pos_weight(cfg, poro_batch, device):

    if cfg.pos_weight_mode == "batch":
        p = float(poro_batch.mean().detach().cpu())
    else:
        p = float(cfg.poro_center)

    pw = (1.0 - p) / max(p, 1e-6)

    pw = float(
        np.clip(
            pw,
            1.0,
            cfg.pos_weight_clip
        )
    )

    return torch.tensor(
        [pw],
        dtype=torch.float32,
        device=device
    )


# ============================================================
# 6. Base blocks
# ============================================================

class ResBlock3D(nn.Module):

    def __init__(self, c):

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


# ============================================================
# 7. EMA VQ
# ============================================================

def nearest_code_indices(
    flat_x,
    embeddings,
    chunk_size=8192
):

    # use fp32 for numerical stability
    flat_x = flat_x.float()
    embeddings = embeddings.float()

    emb_norm = (
        embeddings ** 2
    ).sum(dim=1)

    all_idx = []

    for s in range(
        0,
        flat_x.size(0),
        chunk_size
    ):

        cur = flat_x[s:s + chunk_size]

        distances = (
            (cur ** 2).sum(dim=1, keepdim=True)
            + emb_norm[None]
            - 2.0 * cur @ embeddings.t()
        )

        idx = torch.argmin(
            distances,
            dim=1
        )

        all_idx.append(idx)

    return torch.cat(all_idx, dim=0)


class VectorQuantizerEMA(nn.Module):

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        commitment_cost=0.25,
        decay=0.99,
        eps=1e-5,
        chunk_size=8192,
        dead_threshold=0.10,
        dead_warmup=300,
        dead_reset_interval=200,
        dead_reset_max=64
    ):

        super().__init__()

        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)

        self.commitment_cost = float(commitment_cost)

        self.decay = float(decay)
        self.eps = float(eps)

        self.chunk_size = int(chunk_size)

        self.dead_threshold = float(dead_threshold)
        self.dead_warmup = int(dead_warmup)
        self.dead_reset_interval = int(dead_reset_interval)
        self.dead_reset_max = int(dead_reset_max)

        self.embeddings = nn.Embedding(
            self.num_embeddings,
            self.embedding_dim
        )

        nn.init.normal_(
            self.embeddings.weight,
            mean=0.0,
            std=0.02
        )

        self.embeddings.weight.requires_grad = False

        self.register_buffer(
            "ema_cluster_size",
            torch.zeros(
                self.num_embeddings
            )
        )

        self.register_buffer(
            "ema_embed_sum",
            torch.zeros(
                self.num_embeddings,
                self.embedding_dim
            )
        )

        self.register_buffer(
            "vq_step",
            torch.zeros(
                (),
                dtype=torch.long
            )
        )

    @torch.no_grad()
    def _ema_update(
        self,
        flat_x,
        indices
    ):

        flat_x = flat_x.float()

        counts = torch.bincount(
            indices,
            minlength=self.num_embeddings
        ).float()

        sums = torch.zeros(
            self.num_embeddings,
            self.embedding_dim,
            device=flat_x.device,
            dtype=torch.float32
        )

        sums.index_add_(
            0,
            indices,
            flat_x
        )

        self.ema_cluster_size.mul_(
            self.decay
        ).add_(
            counts,
            alpha=1.0 - self.decay
        )

        self.ema_embed_sum.mul_(
            self.decay
        ).add_(
            sums,
            alpha=1.0 - self.decay
        )

        used = (
            self.ema_cluster_size
            > self.eps
        )

        if used.any():

            new_weight = (
                self.ema_embed_sum[used]
                /
                self.ema_cluster_size[used, None]
                    .clamp_min(self.eps)
            )

            self.embeddings.weight.data[used] = (
                new_weight.to(
                    self.embeddings.weight.dtype
                )
            )

    @torch.no_grad()
    def _reset_dead_codes(
        self,
        flat_x
    ):

        step = int(
            self.vq_step.item()
        )

        if step < self.dead_warmup:
            return 0

        if self.dead_reset_interval <= 0:
            return 0

        if step % self.dead_reset_interval != 0:
            return 0

        dead = torch.nonzero(
            self.ema_cluster_size
            < self.dead_threshold,
            as_tuple=False
        ).flatten()

        if dead.numel() == 0:
            return 0

        if dead.numel() > self.dead_reset_max:

            perm = torch.randperm(
                dead.numel(),
                device=dead.device
            )

            dead = dead[
                perm[:self.dead_reset_max]
            ]

        n = dead.numel()

        rand_idx = torch.randint(
            0,
            flat_x.size(0),
            (n,),
            device=flat_x.device
        )

        replacement = (
            flat_x[rand_idx].float()
        )

        replacement = (
            replacement
            + 0.001
            * torch.randn_like(replacement)
        )

        self.embeddings.weight.data[dead] = (
            replacement.to(
                self.embeddings.weight.dtype
            )
        )

        self.ema_cluster_size[dead] = 1.0
        self.ema_embed_sum[dead] = replacement

        return int(n)

    def forward(
        self,
        inputs,
        update_ema=True
    ):

        B, C, D, H, W = inputs.shape

        assert C == self.embedding_dim

        x = inputs.permute(
            0, 2, 3, 4, 1
        ).contiguous()

        flat_x = x.view(
            -1,
            self.embedding_dim
        )

        indices = nearest_code_indices(
            flat_x,
            self.embeddings.weight,
            chunk_size=self.chunk_size
        )

        quantized = self.embeddings(
            indices
        ).view(
            B,
            D,
            H,
            W,
            C
        )

        # EMA codebook:
        # encoder receives commitment loss,
        # codebook itself is updated via EMA.
        commitment = F.mse_loss(
            x,
            quantized.detach()
        )

        vq_loss = (
            self.commitment_cost
            * commitment
        )

        if (
            self.training
            and update_ema
        ):

            self.vq_step += 1

            self._ema_update(
                flat_x.detach(),
                indices.detach()
            )

            reset_count = self._reset_dead_codes(
                flat_x.detach()
            )

        else:
            reset_count = 0

        # straight-through
        quantized_st = (
            x
            + (
                quantized - x
            ).detach()
        )

        quantized_st = quantized_st.permute(
            0, 4, 1, 2, 3
        ).contiguous()

        counts = torch.bincount(
            indices,
            minlength=self.num_embeddings
        ).float()

        probs = counts / counts.sum().clamp_min(1.0)

        nz = probs > 0

        perplexity = torch.exp(
            -torch.sum(
                probs[nz]
                * torch.log(
                    probs[nz] + 1e-10
                )
            )
        )

        usage_ratio = (
            counts > 0
        ).float().mean()

        dead_ratio = (
            self.ema_cluster_size
            < self.dead_threshold
        ).float().mean()

        quant_error = F.mse_loss(
            x.detach(),
            quantized.detach()
        )

        stats = {
            "perplexity": perplexity,
            "usage_ratio": usage_ratio,
            "dead_ratio": dead_ratio,
            "quant_error": quant_error,
            "reset_count": reset_count
        }

        return (
            quantized_st,
            vq_loss,
            stats
        )


# ============================================================
# 8. Autoencoder / VQ-VAE
# ============================================================

class LatentAE256Down4(nn.Module):

    def __init__(
        self,
        cfg: CFG,
        use_vq: bool
    ):

        super().__init__()

        self.use_vq = bool(use_vq)

        C = cfg.embedding_dim
        b = cfg.vae_base_ch
        m = cfg.vae_max_ch

        c1 = min(b, m)
        c2 = min(2 * b, m)
        c3 = min(3 * b, m)

        self.encoder = nn.Sequential(

            nn.Conv3d(
                1,
                c1,
                4,
                stride=2,
                padding=1
            ),
            nn.SiLU(),
            ResBlock3D(c1),

            nn.Conv3d(
                c1,
                c2,
                4,
                stride=2,
                padding=1
            ),
            nn.SiLU(),
            ResBlock3D(c2),

            nn.Conv3d(
                c2,
                c3,
                3,
                padding=1
            ),
            nn.SiLU(),
            ResBlock3D(c3),

            nn.Conv3d(
                c3,
                C,
                3,
                padding=1
            )
        )

        if self.use_vq:

            self.vq = VectorQuantizerEMA(

                num_embeddings=cfg.num_embeddings,

                embedding_dim=cfg.embedding_dim,

                commitment_cost=cfg.commitment_cost,

                decay=cfg.ema_decay,

                eps=cfg.ema_eps,

                chunk_size=cfg.vq_chunk_size,

                dead_threshold=cfg.dead_code_threshold,

                dead_warmup=cfg.dead_code_warmup,

                dead_reset_interval=cfg.dead_code_reset_interval,

                dead_reset_max=cfg.dead_code_reset_max
            )

        else:
            self.vq = None

        self.decoder = nn.Sequential(

            nn.Conv3d(
                C,
                c3,
                3,
                padding=1
            ),
            nn.SiLU(),
            ResBlock3D(c3),

            nn.ConvTranspose3d(
                c3,
                c2,
                4,
                stride=2,
                padding=1
            ),
            nn.SiLU(),
            ResBlock3D(c2),

            nn.ConvTranspose3d(
                c2,
                c1,
                4,
                stride=2,
                padding=1
            ),
            nn.SiLU(),
            ResBlock3D(c1),

            nn.Conv3d(
                c1,
                max(16, c1 // 2),
                3,
                padding=1
            ),
            nn.SiLU(),

            nn.Conv3d(
                max(16, c1 // 2),
                1,
                3,
                padding=1
            )
        )

    def encode(
        self,
        x,
        update_ema=False
    ):

        z = self.encoder(x)

        if self.use_vq:

            # NOTE: this MUST stay on one logical line (or be wrapped in
            # parentheses).  Written as
            #
            #     z_lat,
            #     vq_loss,
            #     vq_stats = self.vq(z, update_ema=update_ema)
            #
            # a newline outside brackets TERMINATES the statement, so
            # Python parses the first two lines as discarded tuple
            # *expressions* that merely READ z_lat / vq_loss.  The result
            # is `UnboundLocalError: local variable 'z_lat' referenced
            # before assignment` on the very first forward pass, for
            # every run with use_vq=True (variant vq / vqaware).  The
            # use_vq=False branch was unaffected, which is why the crash
            # only ever appeared on the quantized arms.
            (
                z_lat,
                vq_loss,
                vq_stats
            ) = self.vq(
                z,
                update_ema=update_ema
            )

        else:

            z_lat = z

            vq_loss = torch.zeros(
                (),
                device=x.device
            )

            zero = torch.zeros(
                (),
                device=x.device
            )

            vq_stats = {
                "perplexity": zero,
                "usage_ratio": zero,
                "dead_ratio": zero,
                "quant_error": zero,
                "reset_count": 0
            }

        return (
            z_lat,
            vq_loss,
            vq_stats
        )

    def forward(
        self,
        x,
        update_ema=True
    ):

        z_lat, vq_loss, stats = self.encode(
            x,
            update_ema=update_ema
        )

        logits = self.decoder(
            z_lat
        )

        prob = torch.sigmoid(
            logits
        )

        return (
            prob,
            logits,
            z_lat,
            vq_loss,
            stats
        )


# ============================================================
# 9. DDPM network
# ============================================================

def sinusoidal_time_embedding(
    t,
    dim,
    max_period=10000
):

    half = dim // 2

    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(
            half,
            device=t.device
        ).float()
        / half
    )

    args = (
        t.float()[:, None]
        * freqs[None]
    )

    emb = torch.cat(
        [
            torch.cos(args),
            torch.sin(args)
        ],
        dim=1
    )

    if dim % 2 == 1:

        emb = torch.cat(
            [
                emb,
                torch.zeros_like(
                    emb[:, :1]
                )
            ],
            dim=1
        )

    return emb


class SelfAttention3D(nn.Module):

    def __init__(
        self,
        channels,
        heads=4
    ):

        super().__init__()

        self.mha = nn.MultiheadAttention(
            channels,
            heads,
            batch_first=True
        )

        self.ln1 = nn.LayerNorm(
            channels
        )

        self.ff = nn.Sequential(

            nn.LayerNorm(
                channels
            ),

            nn.Linear(
                channels,
                channels
            ),

            nn.GELU(),

            nn.Linear(
                channels,
                channels
            )
        )

    def forward(self, x):

        B, C, D, H, W = x.shape

        N = D * H * W

        x_seq = x.view(
            B,
            C,
            N
        ).transpose(
            1,
            2
        )

        x_ln = self.ln1(
            x_seq
        )

        attn, _ = self.mha(
            x_ln,
            x_ln,
            x_ln
        )

        x_seq = (
            x_seq + attn
        )

        x_seq = (
            x_seq
            + self.ff(x_seq)
        )

        return x_seq.transpose(
            1,
            2
        ).view(
            B,
            C,
            D,
            H,
            W
        )


class UNetLatentCond(nn.Module):

    def __init__(
        self,
        channels,
        base=64,
        time_dim=128
    ):

        super().__init__()

        C = int(channels)

        self.time_dim = int(
            time_dim
        )

        self.inc = nn.Sequential(

            nn.Conv3d(
                C,
                base,
                3,
                padding=1
            ),

            nn.SiLU(),

            ResBlock3D(base)
        )

        self.down1 = nn.Sequential(

            nn.Conv3d(
                base,
                base * 2,
                4,
                stride=2,
                padding=1
            ),

            nn.SiLU(),

            ResBlock3D(
                base * 2
            )
        )

        self.down2 = nn.Sequential(

            nn.Conv3d(
                base * 2,
                base * 4,
                4,
                stride=2,
                padding=1
            ),

            nn.SiLU(),

            ResBlock3D(
                base * 4
            )
        )

        self.attn = SelfAttention3D(
            base * 4,
            heads=4
        )

        self.cond_mlp = nn.Sequential(

            nn.Linear(
                time_dim + 1,
                base * 4
            ),

            nn.SiLU(),

            nn.Linear(
                base * 4,
                base * 8
            )
        )

        self.up1 = nn.Sequential(

            nn.ConvTranspose3d(
                base * 4,
                base * 2,
                4,
                stride=2,
                padding=1
            ),

            nn.SiLU(),

            ResBlock3D(
                base * 2
            )
        )

        self.up2 = nn.Sequential(

            nn.ConvTranspose3d(
                base * 2,
                base,
                4,
                stride=2,
                padding=1
            ),

            nn.SiLU(),

            ResBlock3D(base)
        )

        self.out = nn.Conv3d(
            base,
            C,
            3,
            padding=1
        )

    def forward(
        self,
        x,
        t,
        poro_scaled
    ):

        temb = sinusoidal_time_embedding(
            t,
            self.time_dim
        )

        cond = torch.cat(
            [
                temb,
                poro_scaled[:, None]
            ],
            dim=1
        )

        film = self.cond_mlp(
            cond
        )

        B = x.size(0)

        ch = film.size(1) // 2

        gamma = film[
            :, :ch
        ]

        beta = film[
            :, ch:
        ]

        gamma = gamma.view(
            B,
            ch,
            1,
            1,
            1
        )

        beta = beta.view(
            B,
            ch,
            1,
            1,
            1
        )

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        mid = (
            x3
            * (1.0 + gamma)
            + beta
        )

        mid = self.attn(
            mid
        )

        u1 = (
            self.up1(mid)
            + x2
        )

        u2 = (
            self.up2(u1)
            + x1
        )

        return self.out(
            u2
        )


# ============================================================
# 10. General utilities
# ============================================================

def amp_context(cfg):

    if (
        cfg.amp
        and cfg.device.startswith("cuda")
    ):
        return torch.autocast(
            device_type="cuda",
            dtype=torch.float16
        )

    return nullcontext()


def build_scaler(cfg):

    enabled = (
        cfg.amp
        and cfg.device.startswith("cuda")
    )

    try:
        return torch.amp.GradScaler(
            "cuda",
            enabled=enabled
        )

    except Exception:
        return torch.cuda.amp.GradScaler(
            enabled=enabled
        )


def save_cfg(cfg):

    path = os.path.join(
        cfg.save_dir,
        "run_cfg.json"
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            asdict(cfg),
            f,
            indent=2,
            ensure_ascii=False
        )


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    cfg,
    extra=None
):

    payload = {

        "epoch": int(epoch),

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict()
            if optimizer is not None
            else None,

        "cfg":
            asdict(cfg),

        "extra":
            extra or {}
    }

    torch.save(
        payload,
        path
    )


def _matplotlib_agg():
    """
    Import matplotlib with a headless backend.

    Returns None when unavailable so that a plotting failure can never
    kill a multi-hour training run.
    """

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt

    except Exception:
        return None


def save_recon_slice_image(
    cfg,
    x,
    prob,
    epoch,
    tag="ae",
    logger=None
):
    """
    Save real-slice vs reconstruction-slice for the CURRENT version only.

    The file name is fixed, so every save overwrites the previous one and
    no history gallery is accumulated.  Three orthogonal central slices
    of the first item in the batch are shown as real | recon.
    """

    plt = _matplotlib_agg()

    if plt is None:

        if logger is not None:
            logger.warning(
                "[PLOT] matplotlib unavailable; "
                "skipping reconstruction slice"
            )

        return None

    try:
        real = x[0].detach().float().cpu().numpy()
        rec = prob[0].detach().float().cpu().numpy()

        # x is (B, C, D, H, W) with C == 1; drop BOTH leading axes so
        # that `d, h, w` unpacks.  Dropping only the batch axis leaves a
        # (1, D, H, W) array and raises "too many values to unpack",
        # which the except-block below would swallow into a WARN --
        # i.e. no image would ever be written for the whole run.
        if real.ndim == 4 and real.shape[0] == 1:
            real = real[0]
        if rec.ndim == 4 and rec.shape[0] == 1:
            rec = rec[0]

        d, h, w = real.shape

        views = [
            ("z", real[d // 2, :, :], rec[d // 2, :, :]),
            ("y", real[:, h // 2, :], rec[:, h // 2, :]),
            ("x", real[:, :, w // 2], rec[:, :, w // 2]),
        ]

        fig, axes = plt.subplots(
            2, 3, figsize=(13, 8)
        )

        for j, (name, r_img, c_img) in enumerate(views):

            axes[0, j].imshow(
                r_img, cmap="gray", vmin=0.0, vmax=1.0
            )
            axes[0, j].set_title(
                f"REAL {name}-slice"
            )

            axes[1, j].imshow(
                c_img, cmap="gray", vmin=0.0, vmax=1.0
            )
            axes[1, j].set_title(
                f"RECON {name}-slice"
            )

            axes[0, j].axis("off")
            axes[1, j].axis("off")

        phi_real = float(real.mean())
        phi_rec = float(rec.mean())

        fig.suptitle(
            f"[{tag}] variant={cfg.variant} "
            f"epoch={epoch} "
            f"phi_real={phi_real:.4f} "
            f"phi_recon={phi_rec:.4f}"
        )

        fig.tight_layout()

        out_path = os.path.join(
            cfg.save_dir,
            f"{tag}_recon_latest.png"
        )

        fig.savefig(out_path, dpi=90)
        plt.close(fig)

        return out_path

    except Exception as exc:

        if logger is not None:
            logger.warning(
                f"[PLOT] failed: {exc}"
            )

        return None


def check_finite(
    tensors,
    logger,
    where
):
    """
    Return True when every tensor is finite, else log and return False.
    """

    bad = []

    for name, t in tensors:

        if t is None:
            continue

        try:
            if not bool(torch.isfinite(t).all()):
                bad.append(name)

        except Exception:
            continue

    if bad:

        logger.error(
            f"[NONFINITE] at {where}: {bad}"
        )

        return False

    return True


def load_checkpoint(
    path,
    model,
    optimizer,
    device
):

    if not os.path.exists(path):
        return 0

    ckpt = torch.load(
        path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        ckpt["model_state_dict"]
    )

    if (
        optimizer is not None
        and ckpt.get(
            "optimizer_state_dict"
        ) is not None
    ):

        optimizer.load_state_dict(
            ckpt[
                "optimizer_state_dict"
            ]
        )

    return int(
        ckpt["epoch"]
    ) + 1


def use_vq(cfg):
    return cfg.variant in (
        "vq",
        "vqaware"
    )


# ============================================================
# 11. VQ-VAE validation
# ============================================================

@torch.no_grad()
def validate_vae(
    cfg,
    model,
    val_dl
):

    model.eval()

    device = cfg.device

    total_loss = 0.0
    total_bce = 0.0
    total_dice_loss = 0.0
    total_local = 0.0
    total_boundary = 0.0
    total_vq = 0.0

    total_dice_score = 0.0
    total_phi_mae = 0.0
    total_bd_rel = 0.0

    total_quant = 0.0
    total_ppl = 0.0
    total_usage = 0.0
    total_dead = 0.0

    n = 0

    for x, poro, _ in val_dl:

        x = x.to(
            device,
            non_blocking=True
        )

        poro = poro.to(
            device,
            non_blocking=True
        )

        pos_w = make_pos_weight(
            cfg,
            poro,
            device
        )

        with amp_context(cfg):

            (
                prob,
                logits,
                _,
                vq_loss,
                stats
            ) = model(
                x,
                update_ema=False
            )

            bce = F.binary_cross_entropy_with_logits(
                logits,
                x,
                pos_weight=pos_w
            )

            dsc = soft_dice_loss(
                prob,
                x
            )

            lp = local_porosity_loss(
                prob,
                x,
                cfg.local_phi_kernel
            )

            bd = boundary_gradient_loss(
                prob,
                x
            )

            loss = (
                cfg.bce_weight * bce
                + cfg.dice_weight * dsc
                + cfg.local_phi_weight * lp
                + cfg.boundary_weight * bd
                + vq_loss
            )

        pred = (
            prob >= cfg.fixed_threshold
        ).float()

        dscore = dice_score_binary(
            pred,
            x
        )

        phi_pred = pred.mean(
            dim=(1, 2, 3, 4)
        )

        phi_mae = (
            phi_pred - poro
        ).abs().mean()

        bd_true = boundary_density_binary(
            x
        )

        bd_pred = boundary_density_binary(
            pred
        )

        bd_rel = (
            (bd_pred - bd_true).abs()
            /
            bd_true.clamp_min(1e-8)
        )

        total_loss += float(loss)
        total_bce += float(bce)
        total_dice_loss += float(dsc)
        total_local += float(lp)
        total_boundary += float(bd)
        total_vq += float(vq_loss)

        total_dice_score += float(dscore)
        total_phi_mae += float(phi_mae)
        total_bd_rel += float(bd_rel)

        total_quant += float(
            stats["quant_error"]
        )

        total_ppl += float(
            stats["perplexity"]
        )

        total_usage += float(
            stats["usage_ratio"]
        )

        total_dead += float(
            stats["dead_ratio"]
        )

        n += 1

    d = max(n, 1)

    return {

        "loss": total_loss / d,

        "bce": total_bce / d,

        "dice_loss":
            total_dice_loss / d,

        "local_phi":
            total_local / d,

        "boundary":
            total_boundary / d,

        "vq":
            total_vq / d,

        "dice_score":
            total_dice_score / d,

        "phi_mae":
            total_phi_mae / d,

        "boundary_rel":
            total_bd_rel / d,

        "quant_error":
            total_quant / d,

        "perplexity":
            total_ppl / d,

        "usage_ratio":
            total_usage / d,

        "dead_ratio":
            total_dead / d
    }


# ============================================================
# 12. Train AE/VQ-VAE
# ============================================================

def train_vae(
    cfg,
    logger
):

    device = cfg.device

    train_dl, val_dl = build_loaders(
        cfg,
        stage="vae"
    )

    model = LatentAE256Down4(
        cfg,
        use_vq=use_vq(cfg)
    ).to(device)

    opt = torch.optim.Adam(
        [
            p
            for p in model.parameters()
            if p.requires_grad
        ],
        lr=cfg.lr_vae
    )

    scaler = build_scaler(
        cfg
    )

    latest_path = os.path.join(
        cfg.save_dir,
        "ae_latest.pth"
    )

    best_path = os.path.join(
        cfg.save_dir,
        "ae_best.pth"
    )

    final_path = os.path.join(
        cfg.save_dir,
        "ae_final.pth"
    )

    start_epoch = load_checkpoint(
        latest_path,
        model,
        opt,
        device
    )

    best_val = float("inf")

    best_meta_path = os.path.join(
        cfg.save_dir,
        "ae_best_metrics.json"
    )

    if os.path.exists(
        best_meta_path
    ):

        try:
            with open(
                best_meta_path,
                "r",
                encoding="utf-8"
            ) as f:

                best_val = float(
                    json.load(f)["val_loss"]
                )

        except Exception:
            pass

    logger.info(
        "=== STAGE: AE / VQ-VAE ==="
    )

    logger.info(
        f"variant={cfg.variant} "
        f"use_vq={use_vq(cfg)}"
    )

    step = 0

    nonfinite_count = 0
    nonfinite_total = 0

    history_path = os.path.join(
        cfg.save_dir,
        "ae_history.csv"
    )

    for epoch in range(
        start_epoch,
        cfg.epochs_vae
    ):

        model.train()

        t0 = time.time()

        loss_sum = 0.0
        n_batches = 0

        for x, poro, coord in train_dl:

            x = x.to(
                device,
                non_blocking=True
            )

            poro = poro.to(
                device,
                non_blocking=True
            )

            pos_w = make_pos_weight(
                cfg,
                poro,
                device
            )

            opt.zero_grad(
                set_to_none=True
            )

            with amp_context(cfg):

                (
                    prob,
                    logits,
                    z_lat,
                    vq_loss,
                    stats
                ) = model(
                    x,
                    update_ema=True
                )

                bce = F.binary_cross_entropy_with_logits(
                    logits,
                    x,
                    pos_weight=pos_w
                )

                dsc = soft_dice_loss(
                    prob,
                    x
                )

                lp = local_porosity_loss(
                    prob,
                    x,
                    cfg.local_phi_kernel
                )

                bd = boundary_gradient_loss(
                    prob,
                    x
                )

                loss = (
                    cfg.bce_weight * bce
                    + cfg.dice_weight * dsc
                    + cfg.local_phi_weight * lp
                    + cfg.boundary_weight * bd
                    + vq_loss
                )

            if not check_finite(
                [
                    ("loss", loss),
                    ("prob", prob),
                    ("logits", logits),
                    ("z_lat", z_lat),
                ],
                logger,
                f"AE epoch={epoch+1} step={step+1}"
            ):

                nonfinite_count += 1
                nonfinite_total += 1

                opt.zero_grad(set_to_none=True)
                scaler.update()

                if nonfinite_count >= cfg.nonfinite_abort:

                    raise RuntimeError(
                        f"AE: {nonfinite_count} consecutive "
                        f"non-finite batches at "
                        f"epoch={epoch+1} step={step+1}"
                    )

                continue

            nonfinite_count = 0

            scaler.scale(
                loss
            ).backward()

            grad_norm = None

            if (
                cfg.grad_clip
                and cfg.grad_clip > 0
            ):

                scaler.unscale_(opt)

                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        cfg.grad_clip
                    )
                )

            scaler.step(opt)
            scaler.update()

            loss_sum += float(
                loss.detach().cpu()
            )

            n_batches += 1
            step += 1

            if (
                step
                % cfg.log_interval
                == 0
            ):

                phi_gt = float(
                    poro.mean()
                )

                phi_soft = float(
                    prob.mean()
                )

                msg = (
                    f"[AE] "
                    f"epoch={epoch+1}/{cfg.epochs_vae} "
                    f"step={step} "
                    f"loss={float(loss):.5f} "
                    f"bce={float(bce):.5f} "
                    f"dice={float(dsc):.5f} "
                    f"local={float(lp):.5f} "
                    f"boundary={float(bd):.5f} "
                    f"vq={float(vq_loss):.5f} "
                    f"phi_gt={phi_gt:.4f} "
                    f"phi_soft={phi_soft:.4f} "
                    f"|z|={float(z_lat.abs().mean()):.4f} "
                    f"gnorm="
                    + (
                        f"{grad_norm:.3f}"
                        if grad_norm is not None
                        else "NA"
                    )
                    + (
                        f" nonfinite={nonfinite_total}"
                        if nonfinite_total > 0
                        else ""
                    )
                )

                if use_vq(cfg):

                    msg += (
                        f" ppl={float(stats['perplexity']):.1f}"
                        f" usage={float(stats['usage_ratio']):.3f}"
                        f" dead={float(stats['dead_ratio']):.3f}"
                        f" qerr={float(stats['quant_error']):.6f}"
                        f" reset={stats['reset_count']}"
                    )

                logger.info(msg)

        val = validate_vae(
            cfg,
            model,
            val_dl
        )

        dt = time.time() - t0

        logger.info(
            f"[AE-VAL] "
            f"epoch={epoch+1} "
            f"train_loss={loss_sum/max(n_batches,1):.6f} "
            f"val_loss={val['loss']:.6f} "
            f"dice={val['dice_score']:.4f} "
            f"phi_mae={val['phi_mae']:.6f} "
            f"boundary_rel={val['boundary_rel']:.4f} "
            f"ppl={val['perplexity']:.1f} "
            f"usage={val['usage_ratio']:.3f} "
            f"dead={val['dead_ratio']:.3f} "
            f"time={dt/60:.2f}min"
        )

        new_file = not os.path.exists(
            history_path
        )

        with open(
            history_path,
            "a",
            newline="",
            encoding="utf-8"
        ) as f:

            w = csv.DictWriter(
                f,
                fieldnames=[
                    "epoch",
                    "train_loss",
                    *[
                        f"val_{k}"
                        for k
                        in val.keys()
                    ]
                ]
            )

            if new_file:
                w.writeheader()

            row = {
                "epoch": epoch + 1,
                "train_loss":
                    loss_sum
                    / max(n_batches, 1)
            }

            for k, v in val.items():
                row[
                    f"val_{k}"
                ] = v

            w.writerow(row)

        if (
            epoch + 1
        ) % cfg.save_interval == 0:

            save_checkpoint(
                latest_path,
                model,
                opt,
                epoch,
                cfg
            )

            # Real vs reconstruction slice, CURRENT version only
            # (fixed filename, overwritten each time).
            try:

                vb = next(iter(val_dl))
                vx = vb[0].to(
                    device,
                    non_blocking=True
                )

                model.eval()

                with torch.no_grad():

                    vprob = model(
                        vx,
                        update_ema=False
                    )[0]

                save_recon_slice_image(
                    cfg,
                    vx,
                    vprob,
                    epoch + 1,
                    tag="ae",
                    logger=logger
                )

                model.train()

            except Exception as exc:

                logger.warning(
                    f"[PLOT] recon slice skipped: {exc}"
                )

        if val["loss"] < best_val:

            best_val = val["loss"]

            save_checkpoint(
                best_path,
                model,
                None,
                epoch,
                cfg,
                extra={
                    "validation_metrics":
                        val
                }
            )

            with open(
                best_meta_path,
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    {
                        "epoch":
                            epoch + 1,

                        "val_loss":
                            best_val,

                        "metrics":
                            val
                    },
                    f,
                    indent=2
                )

            logger.info(
                f"[AE-BEST] "
                f"epoch={epoch+1} "
                f"val_loss={best_val:.6f}"
            )

        if device.startswith(
            "cuda"
        ):
            torch.cuda.empty_cache()

    save_checkpoint(
        final_path,
        model,
        None,
        cfg.epochs_vae - 1,
        cfg
    )

    logger.info(
        "AE/VQ-VAE training done."
    )


# ============================================================
# 13. Load best AE
# ============================================================

def load_best_ae(
    cfg,
    logger
):

    model = LatentAE256Down4(
        cfg,
        use_vq=use_vq(cfg)
    ).to(cfg.device)

    path = os.path.join(
        cfg.save_dir,
        "ae_best.pth"
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing: {path}"
        )

    ckpt = torch.load(
        path,
        map_location=cfg.device,
        weights_only=False
    )

    model.load_state_dict(
        ckpt[
            "model_state_dict"
        ]
    )

    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    logger.info(
        f"[LOAD] best AE: {path}"
    )

    return model


# ============================================================
# 14. Train-only latent statistics
# ============================================================

@torch.no_grad()
def compute_latent_stats(
    cfg,
    logger
):

    model = load_best_ae(
        cfg,
        logger
    )

    train_dl, _ = build_loaders(
        cfg,
        stage="ddpm"
    )

    C = cfg.embedding_dim

    sum_c = torch.zeros(
        C,
        device=cfg.device,
        dtype=torch.float64
    )

    sumsq_c = torch.zeros(
        C,
        device=cfg.device,
        dtype=torch.float64
    )

    count = 0
    batches = 0

    for x, _, _ in train_dl:

        x = x.to(
            cfg.device,
            non_blocking=True
        )

        z, _, _ = model.encode(
            x,
            update_ema=False
        )

        z = z.double()

        sum_c += z.sum(
            dim=(0, 2, 3, 4)
        )

        sumsq_c += (
            z ** 2
        ).sum(
            dim=(0, 2, 3, 4)
        )

        count += (
            z.shape[0]
            * z.shape[2]
            * z.shape[3]
            * z.shape[4]
        )

        batches += 1

        if batches >= cfg.stats_batches:
            break

    mean = sum_c / count

    var = (
        sumsq_c / count
        - mean ** 2
    )

    var = torch.clamp(
        var,
        min=1e-8
    )

    std = torch.sqrt(
        var
    )

    out = os.path.join(
        cfg.save_dir,
        "latent_stats.npz"
    )

    np.savez_compressed(

        out,

        mean=mean.float()
            .cpu()
            .numpy(),

        std=std.float()
            .cpu()
            .numpy(),

        count=int(count),

        stats_batches=batches,

        source_region=np.asarray(
            cfg.train_region
        ),

        variant=cfg.variant,

        embedding_dim=
            cfg.embedding_dim
    )

    logger.info(
        f"[LATENT STATS] "
        f"TRAIN REGION ONLY "
        f"batches={batches} "
        f"count={count}"
    )

    logger.info(
        f"mean=[{float(mean.min()):.5f},"
        f"{float(mean.max()):.5f}] "
        f"std=[{float(std.min()):.5f},"
        f"{float(std.max()):.5f}]"
    )


def load_latent_stats(
    cfg,
    logger
):

    path = os.path.join(
        cfg.save_dir,
        "latent_stats.npz"
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            path
        )

    d = np.load(
        path,
        allow_pickle=True
    )

    mean = torch.tensor(
        d["mean"],
        dtype=torch.float32,
        device=cfg.device
    )

    std = torch.tensor(
        d["std"],
        dtype=torch.float32,
        device=cfg.device
    )

    logger.info(
        f"[LOAD] latent stats: {path}"
    )

    return mean, std


# ============================================================
# 15. Codebook consistency
# ============================================================

def codebook_consistency_loss(
    z_pred,
    embedding_weight,
    n_samples=8192,
    chunk_size=8192
):

    # z_pred:
    # [B,C,D,H,W]

    B, C, D, H, W = (
        z_pred.shape
    )

    flat = z_pred.permute(
        0, 2, 3, 4, 1
    ).reshape(
        -1,
        C
    )

    n = min(
        int(n_samples),
        flat.size(0)
    )

    idx_sample = torch.randint(
        0,
        flat.size(0),
        (n,),
        device=flat.device
    )

    selected = flat[
        idx_sample
    ]

    with torch.no_grad():

        code_idx = nearest_code_indices(
            selected.detach(),
            embedding_weight.detach(),
            chunk_size=chunk_size
        )

        target = embedding_weight[
            code_idx
        ].detach()

    loss = F.mse_loss(
        selected,
        target
    )

    return loss


# ============================================================
# 16. DDPM validation
# ============================================================

@torch.no_grad()
def validate_ddpm(
    cfg,
    ae,
    unet,
    val_dl,
    lat_mean,
    lat_std,
    alphas_cumprod
):

    unet.eval()

    noise_sum = 0.0
    cb_sum = 0.0
    total_sum = 0.0

    n = 0

    for batch_idx, (
        x,
        poro,
        _
    ) in enumerate(val_dl):

        x = x.to(
            cfg.device,
            non_blocking=True
        )

        poro = poro.to(
            cfg.device,
            non_blocking=True
        )

        z, _, _ = ae.encode(
            x,
            update_ema=False
        )

        z_n = (
            z
            - lat_mean[
                None, :, None, None, None
            ]
        ) / lat_std[
            None, :, None, None, None
        ].clamp_min(
            1e-6
        )

        B = z_n.size(0)

        # deterministic validation randomness
        g = torch.Generator(
            device=cfg.device
        )

        g.manual_seed(
            cfg.seed
            + 100000
            + batch_idx
        )

        t = torch.randint(
            0,
            cfg.n_steps,
            (B,),
            device=cfg.device,
            generator=g
        )

        noise = torch.randn(
            z_n.shape,
            device=cfg.device,
            generator=g,
            dtype=z_n.dtype
        )

        a_hat = alphas_cumprod[
            t
        ]

        sqrt_a = torch.sqrt(
            a_hat
        )[
            :, None, None, None, None
        ]

        sqrt_om = torch.sqrt(
            1.0 - a_hat
        )[
            :, None, None, None, None
        ]

        x_t = (
            sqrt_a * z_n
            + sqrt_om * noise
        )

        poro_scaled = (
            poro
            - cfg.poro_center
        ) / max(
            cfg.poro_scale,
            1e-6
        )

        poro_scaled = poro_scaled.clamp(
            -cfg.poro_clip,
            cfg.poro_clip
        )

        if cfg.condition_mode == "none":

            poro_scaled = torch.zeros_like(
                poro_scaled
            )

        pred = unet(
            x_t,
            t,
            poro_scaled
        )

        noise_loss = F.mse_loss(
            pred,
            noise
        )

        cb_loss = torch.zeros(
            (),
            device=cfg.device
        )

        if (
            cfg.variant == "vqaware"
            and cfg.lambda_codebook > 0
        ):

            x0_pred_n = (
                x_t
                - sqrt_om * pred
            ) / sqrt_a.clamp_min(
                1e-4
            )

            x0_pred = (
                x0_pred_n
                * lat_std[
                    None, :, None, None, None
                ]
                + lat_mean[
                    None, :, None, None, None
                ]
            )

            cb_loss = codebook_consistency_loss(

                x0_pred,

                ae.vq.embeddings.weight,

                n_samples=
                    cfg.codebook_sample_voxels,

                chunk_size=
                    cfg.vq_chunk_size
            )

        # low-noise states receive stronger manifold constraint
        cb_weight = a_hat.mean()

        total = (
            noise_loss
            + cfg.lambda_codebook
            * cb_weight
            * cb_loss
            if cfg.variant == "vqaware"
            else noise_loss
        )

        noise_sum += float(
            noise_loss
        )

        cb_sum += float(
            cb_loss
        )

        total_sum += float(
            total
        )

        n += 1

    d = max(n, 1)

    return {

        "noise_loss":
            noise_sum / d,

        "codebook_loss":
            cb_sum / d,

        "total_loss":
            total_sum / d
    }


# ============================================================
# 17. Train DDPM
# ============================================================

def train_ddpm(
    cfg,
    logger
):

    device = cfg.device

    ae = load_best_ae(
        cfg,
        logger
    )

    lat_mean, lat_std = load_latent_stats(
        cfg,
        logger
    )

    train_dl, val_dl = build_loaders(
        cfg,
        stage="ddpm"
    )

    unet = UNetLatentCond(

        channels=cfg.embedding_dim,

        base=cfg.unet_base,

        time_dim=cfg.unet_time_dim

    ).to(device)

    # Warm start, added by the FiLM iteration.  Weights ONLY -- the optimiser
    # state is deliberately not restored, because every round of this study
    # changes the condition scaling, which rescales the very input the first
    # cond_mlp layer reads; carrying Adam's moment estimates for that weight
    # across a rescaled input would import a stale curvature estimate.
    init_p = getattr(cfg, "init_unet", "")
    if init_p:
        _ck = torch.load(init_p, map_location=device, weights_only=False)
        _sd = _ck["model_state_dict"]
        # Shape check before load: silently accepting a mismatched warm start
        # would look exactly like "the knob did nothing".
        _cur = unet.state_dict()
        if set(_sd) != set(_cur):
            raise SystemExit(
                f"--init_unet {init_p}: state_dict keys differ from a freshly "
                f"built UNet. Missing {sorted(set(_cur)-set(_sd))[:4]}, "
                f"unexpected {sorted(set(_sd)-set(_cur))[:4]}")
        _bad = [k for k in _sd if tuple(_sd[k].shape) != tuple(_cur[k].shape)]
        if _bad:
            raise SystemExit(
                f"--init_unet {init_p}: shape mismatch on {_bad[:4]}")
        unet.load_state_dict(_sd)
        logger.info(
            f"[INIT] warm start from {init_p} (epoch {_ck.get('epoch')}); "
            f"optimiser state NOT restored")

    opt = torch.optim.Adam(
        unet.parameters(),
        lr=cfg.lr_ddpm
    )

    scaler = build_scaler(
        cfg
    )

    betas = torch.linspace(
        cfg.beta_start,
        cfg.beta_end,
        cfg.n_steps,
        device=device
    )

    alphas = 1.0 - betas

    alphas_cumprod = torch.cumprod(
        alphas,
        dim=0
    )

    latest_path = os.path.join(
        cfg.save_dir,
        "unet_latest.pth"
    )

    best_path = os.path.join(
        cfg.save_dir,
        "unet_best.pth"
    )

    final_path = os.path.join(
        cfg.save_dir,
        "unet_final.pth"
    )

    start_epoch = load_checkpoint(
        latest_path,
        unet,
        opt,
        device
    )

    best_val = float("inf")

    history_path = os.path.join(
        cfg.save_dir,
        "ddpm_history.csv"
    )

    logger.info(
        "=== STAGE: LATENT DDPM ==="
    )

    logger.info(
        f"variant={cfg.variant} "
        f"condition={cfg.condition_mode} "
        f"lambda_codebook="
        f"{cfg.lambda_codebook}"
    )

    step = 0

    nonfinite_count = 0
    nonfinite_total = 0
    best_val_noise = float("inf")

    for epoch in range(
        start_epoch,
        cfg.epochs_ddpm
    ):

        unet.train()

        t0 = time.time()

        noise_sum = 0.0
        cb_sum = 0.0
        total_sum = 0.0

        n_batches = 0

        for x, poro, coord in train_dl:

            x = x.to(
                device,
                non_blocking=True
            )

            poro = poro.to(
                device,
                non_blocking=True
            )

            with torch.no_grad():

                z, _, _ = ae.encode(
                    x,
                    update_ema=False
                )

                z_n = (
                    z
                    - lat_mean[
                        None,
                        :,
                        None,
                        None,
                        None
                    ]
                ) / lat_std[
                    None,
                    :,
                    None,
                    None,
                    None
                ].clamp_min(
                    1e-6
                )

            B = z_n.size(0)

            t = torch.randint(
                0,
                cfg.n_steps,
                (B,),
                device=device,
                dtype=torch.long
            )

            noise = torch.randn_like(
                z_n
            )

            a_hat = alphas_cumprod[
                t
            ]

            sqrt_a = torch.sqrt(
                a_hat
            )[
                :,
                None,
                None,
                None,
                None
            ]

            sqrt_om = torch.sqrt(
                1.0 - a_hat
            )[
                :,
                None,
                None,
                None,
                None
            ]

            x_noisy = (
                sqrt_a * z_n
                + sqrt_om * noise
            )

            poro_scaled = (
                poro
                - cfg.poro_center
            ) / max(
                cfg.poro_scale,
                1e-6
            )

            poro_scaled = poro_scaled.clamp(
                -cfg.poro_clip,
                cfg.poro_clip
            )

            if cfg.condition_mode == "none":

                poro_scaled = torch.zeros_like(
                    poro_scaled
                )

            opt.zero_grad(
                set_to_none=True
            )

            with amp_context(cfg):

                pred = unet(
                    x_noisy,
                    t,
                    poro_scaled
                )

                noise_loss = F.mse_loss(
                    pred,
                    noise
                )

                cb_loss = torch.zeros(
                    (),
                    device=device
                )

                if (
                    cfg.variant == "vqaware"
                    and cfg.lambda_codebook > 0
                ):

                    x0_pred_n = (
                        x_noisy
                        - sqrt_om * pred
                    ) / sqrt_a.clamp_min(
                        1e-4
                    )

                    x0_pred = (
                        x0_pred_n
                        * lat_std[
                            None,
                            :,
                            None,
                            None,
                            None
                        ]
                        + lat_mean[
                            None,
                            :,
                            None,
                            None,
                            None
                        ]
                    )

                    cb_loss = (
                        codebook_consistency_loss(

                            x0_pred,

                            ae.vq.embeddings.weight,

                            n_samples=
                                cfg.codebook_sample_voxels,

                            chunk_size=
                                cfg.vq_chunk_size
                        )
                    )

                # stronger codebook regularization
                # at lower diffusion noise.
                cb_weight = a_hat.mean()

                if cfg.variant == "vqaware":

                    loss = (
                        noise_loss
                        + cfg.lambda_codebook
                        * cb_weight
                        * cb_loss
                    )

                else:

                    loss = noise_loss

            if not check_finite(
                [
                    ("loss", loss),
                    ("noise_loss", noise_loss),
                    ("pred", pred),
                    ("z_n", z_n),
                ],
                logger,
                f"DDPM epoch={epoch+1} step={step+1}"
            ):

                nonfinite_count += 1
                nonfinite_total += 1

                opt.zero_grad(set_to_none=True)
                scaler.update()

                if nonfinite_count >= cfg.nonfinite_abort:

                    raise RuntimeError(
                        f"DDPM: {nonfinite_count} consecutive "
                        f"non-finite batches at "
                        f"epoch={epoch+1} step={step+1}"
                    )

                continue

            nonfinite_count = 0

            scaler.scale(
                loss
            ).backward()

            grad_norm = None

            if (
                cfg.grad_clip
                and cfg.grad_clip > 0
            ):

                scaler.unscale_(opt)

                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        unet.parameters(),
                        cfg.grad_clip
                    )
                )

            scaler.step(opt)
            scaler.update()

            noise_sum += float(
                noise_loss.detach().cpu()
            )

            cb_sum += float(
                cb_loss.detach().cpu()
            )

            total_sum += float(
                loss.detach().cpu()
            )

            n_batches += 1
            step += 1

            if (
                step
                % cfg.log_interval
                == 0
            ):

                logger.info(
                    f"[DDPM] "
                    f"epoch={epoch+1}/{cfg.epochs_ddpm} "
                    f"step={step} "
                    f"loss={float(loss):.5f} "
                    f"noise={float(noise_loss):.5f} "
                    f"cb={float(cb_loss):.6f} "
                    f"cb_w={float(cb_weight):.4f} "
                    f"phi={float(poro.mean()):.4f} "
                    f"|zn|={float(z_n.abs().mean()):.4f} "
                    f"gnorm="
                    + (
                        f"{grad_norm:.3f}"
                        if grad_norm is not None
                        else "NA"
                    )
                    + (
                        f" nonfinite={nonfinite_total}"
                        if nonfinite_total > 0
                        else ""
                    )
                )

        val = validate_ddpm(
            cfg,
            ae,
            unet,
            val_dl,
            lat_mean,
            lat_std,
            alphas_cumprod
        )

        dt = time.time() - t0

        logger.info(
            f"[DDPM-VAL] "
            f"epoch={epoch+1} "
            f"train_total="
            f"{total_sum/max(n_batches,1):.6f} "
            f"val_noise="
            f"{val['noise_loss']:.6f} "
            f"val_cb="
            f"{val['codebook_loss']:.6f} "
            f"val_total="
            f"{val['total_loss']:.6f} "
            f"time={dt/60:.2f}min"
        )

        new_file = not os.path.exists(
            history_path
        )

        with open(
            history_path,
            "a",
            newline="",
            encoding="utf-8"
        ) as f:

            w = csv.writer(f)

            if new_file:

                w.writerow([
                    "epoch",
                    "train_noise",
                    "train_codebook",
                    "train_total",
                    "val_noise",
                    "val_codebook",
                    "val_total"
                ])

            w.writerow([
                epoch + 1,

                noise_sum
                / max(n_batches, 1),

                cb_sum
                / max(n_batches, 1),

                total_sum
                / max(n_batches, 1),

                val["noise_loss"],

                val["codebook_loss"],

                val["total_loss"]
            ])

        if (
            epoch + 1
        ) % cfg.save_interval == 0:

            save_checkpoint(
                latest_path,
                unet,
                opt,
                epoch,
                cfg
            )

        if (
            epoch + 1
        ) % 50 == 0:

            save_checkpoint(
                os.path.join(
                    cfg.save_dir,
                    f"unet_ep{epoch+1:03d}.pth"
                ),
                unet,
                None,
                epoch,
                cfg
            )

        if (
            val["total_loss"]
            < best_val
        ):

            best_val = val[
                "total_loss"
            ]

            save_checkpoint(
                best_path,
                unet,
                None,
                epoch,
                cfg,
                extra={
                    "validation_metrics":
                        val
                }
            )

            logger.info(
                f"[DDPM-BEST] "
                f"epoch={epoch+1} "
                f"val_total={best_val:.6f}"
            )

        # Secondary selection keyed on val NOISE loss only.
        #
        # val_total includes the codebook-consistency term, whose weight
        # (lambda_codebook) differs BY DESIGN across the Stage-7 ablation
        # arms.  Selecting on val_total would therefore reward the arm
        # with the smallest lambda rather than the best noise predictor.
        # val_noise is defined identically in every arm, so the
        # checkpoint picked here is the one that is comparable.
        if val["noise_loss"] < best_val_noise:

            best_val_noise = val[
                "noise_loss"
            ]

            save_checkpoint(
                os.path.join(
                    cfg.save_dir,
                    "unet_best_noise.pth"
                ),
                unet,
                None,
                epoch,
                cfg,
                extra={
                    "validation_metrics":
                        val
                }
            )

        if device.startswith(
            "cuda"
        ):
            torch.cuda.empty_cache()

    save_checkpoint(
        final_path,
        unet,
        None,
        cfg.epochs_ddpm - 1,
        cfg
    )

    logger.info(
        "DDPM training done."
    )


# ============================================================
# 18. AQB exact top-N
# ============================================================

def aqb_exact(
    prob_np,
    target_porosity
):

    flat = prob_np.reshape(-1)

    N = flat.size

    target_cnt = int(
        round(
            target_porosity
            * N
        )
    )

    target_cnt = int(
        np.clip(
            target_cnt,
            0,
            N
        )
    )

    seg = np.zeros(
        N,
        dtype=np.uint8
    )

    if target_cnt == 0:
        return seg.reshape(
            prob_np.shape
        )

    if target_cnt == N:

        seg[:] = 1

        return seg.reshape(
            prob_np.shape
        )

    idx = np.argpartition(
        flat,
        N - target_cnt
    )[N - target_cnt:]

    seg[idx] = 1

    return seg.reshape(
        prob_np.shape
    )


# ============================================================
# 19. Sample one
# ============================================================

@torch.no_grad()
def sample_single(
    cfg,
    ae,
    unet,
    lat_mean,
    lat_std,
    target_porosity,
    seed,
    logger
):

    torch.manual_seed(
        seed
    )

    np.random.seed(
        seed
    )

    device = cfg.device

    betas = torch.linspace(
        cfg.beta_start,
        cfg.beta_end,
        cfg.n_steps,
        device=device
    )

    alphas = 1.0 - betas

    alphas_cumprod = torch.cumprod(
        alphas,
        dim=0
    )

    L = (
        cfg.patch_size
        // cfg.downsample_factor
    )

    x = torch.randn(
        1,
        cfg.embedding_dim,
        L,
        L,
        L,
        device=device
    )

    poro = torch.tensor(
        [float(target_porosity)],
        dtype=torch.float32,
        device=device
    )

    poro_scaled = (
        poro
        - cfg.poro_center
    ) / max(
        cfg.poro_scale,
        1e-6
    )

    poro_scaled = poro_scaled.clamp(
        -cfg.poro_clip,
        cfg.poro_clip
    )

    if cfg.condition_mode == "none":

        poro_scaled = torch.zeros_like(
            poro_scaled
        )

    for i in range(
        cfg.n_steps - 1,
        -1,
        -1
    ):

        t = torch.full(
            (1,),
            i,
            device=device,
            dtype=torch.long
        )

        eps = unet(
            x,
            t,
            poro_scaled
        )

        if cfg.eps_clip > 0:

            eps = eps.clamp(
                -cfg.eps_clip,
                cfg.eps_clip
            )

        alpha = alphas[i]
        alpha_hat = alphas_cumprod[i]
        beta = betas[i]

        if i > 0:

            z = torch.randn_like(
                x
            )

        else:

            z = torch.zeros_like(
                x
            )

        x = (
            1.0
            / torch.sqrt(alpha)
        ) * (
            x
            - (
                (1.0 - alpha)
                /
                torch.sqrt(
                    1.0 - alpha_hat
                )
            ) * eps
        ) + torch.sqrt(
            beta
        ) * z

        if cfg.clamp_x > 0:

            x = x.clamp(
                -cfg.clamp_x,
                cfg.clamp_x
            )

    # normalized latent -> AE latent
    x_denorm = (
        x
        * lat_std[
            None, :, None, None, None
        ]
        + lat_mean[
            None, :, None, None, None
        ]
    )

    cb_distance = np.nan

    if use_vq(cfg):

        # distance BEFORE final codebook projection
        flat = x_denorm.permute(
            0, 2, 3, 4, 1
        ).reshape(
            -1,
            cfg.embedding_dim
        )

        n = min(
            8192,
            flat.size(0)
        )

        ridx = torch.randint(
            0,
            flat.size(0),
            (n,),
            device=device
        )

        selected = flat[
            ridx
        ]

        code_idx = nearest_code_indices(
            selected,
            ae.vq.embeddings.weight,
            chunk_size=cfg.vq_chunk_size
        )

        nearest = ae.vq.embeddings.weight[
            code_idx
        ]

        cb_distance = float(
            torch.sqrt(
                F.mse_loss(
                    selected,
                    nearest
                )
            )
        )

        # final projection to learned codebook
        x_decode, _, _ = ae.vq(
            x_denorm,
            update_ema=False
        )

    else:

        x_decode = x_denorm

    logits = ae.decoder(
        x_decode
    )

    prob = torch.sigmoid(
        logits
    )

    prob_np = (
        prob[0, 0]
        .float()
        .cpu()
        .numpy()
    )

    soft_phi = float(
        prob_np.mean()
    )

    # AQB OFF
    seg_fixed = (
        prob_np
        >= cfg.fixed_threshold
    ).astype(
        np.uint8
    )

    fixed_phi = float(
        seg_fixed.mean()
    )

    # AQB ON
    seg_aqb = aqb_exact(
        prob_np,
        target_porosity
    )

    aqb_phi = float(
        seg_aqb.mean()
    )

    return {

        "prob":
            prob_np,

        "seg_fixed":
            seg_fixed,

        "seg_aqb":
            seg_aqb,

        "soft_phi":
            soft_phi,

        "fixed_phi":
            fixed_phi,

        "aqb_phi":
            aqb_phi,

        "codebook_distance":
            cb_distance
    }


# ============================================================
# 20. Sample grid: 0.12 / 0.13 / 0.14
# ============================================================

@torch.no_grad()
def sample_grid(
    cfg,
    logger
):

    ae = load_best_ae(
        cfg,
        logger
    )

    lat_mean, lat_std = load_latent_stats(
        cfg,
        logger
    )

    unet = UNetLatentCond(

        channels=cfg.embedding_dim,

        base=cfg.unet_base,

        time_dim=cfg.unet_time_dim

    ).to(cfg.device)

    path = os.path.join(
        cfg.save_dir,
        cfg.sample_checkpoint
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            path
        )

    ckpt = torch.load(
        path,
        map_location=cfg.device,
        weights_only=False
    )

    unet.load_state_dict(
        ckpt[
            "model_state_dict"
        ]
    )

    unet.eval()

    targets = [
        float(x)
        for x in
        cfg.sample_targets.split(",")
    ]

    out_dir = os.path.join(
        cfg.save_dir,
        "condition_sweep"
    )

    os.makedirs(
        out_dir,
        exist_ok=True
    )

    summary = []

    # SAME seed across targets gives paired condition test.
    for target in targets:

        for i in range(
            cfg.samples_per_target
        ):

            seed = (
                cfg.seed
                + 50000
                + i
            )

            logger.info(
                f"[SAMPLE] "
                f"target={target:.3f} "
                f"seed={seed}"
            )

            result = sample_single(

                cfg,

                ae,

                unet,

                lat_mean,

                lat_std,

                target,

                seed,

                logger
            )

            tag = (
                f"tp{target:.3f}"
                f"_seed{seed}"
            ).replace(
                ".",
                "p"
            )

            np.savez_compressed(

                os.path.join(
                    out_dir,
                    f"{tag}.npz"
                ),

                prob=result["prob"],

                seg_fixed=
                    result["seg_fixed"],

                seg_aqb=
                    result["seg_aqb"],

                target_porosity=
                    target,

                seed=
                    seed,

                soft_phi=
                    result["soft_phi"],

                fixed_phi=
                    result["fixed_phi"],

                aqb_phi=
                    result["aqb_phi"],

                codebook_distance=
                    result[
                        "codebook_distance"
                    ]
            )

            result[
                "seg_fixed"
            ].tofile(
                os.path.join(
                    out_dir,
                    f"{tag}_fixed.raw"
                )
            )

            result[
                "seg_aqb"
            ].tofile(
                os.path.join(
                    out_dir,
                    f"{tag}_aqb.raw"
                )
            )

            summary.append({

                "target":
                    target,

                "seed":
                    seed,

                "soft_phi":
                    result[
                        "soft_phi"
                    ],

                # MAIN FiLM evidence:
                # this is AQB-OFF
                "fixed_phi":
                    result[
                        "fixed_phi"
                    ],

                # exact calibrated result
                "aqb_phi":
                    result[
                        "aqb_phi"
                    ],

                "fixed_abs_error":
                    abs(
                        result[
                            "fixed_phi"
                        ]
                        - target
                    ),

                "aqb_abs_error":
                    abs(
                        result[
                            "aqb_phi"
                        ]
                        - target
                    ),

                "codebook_distance":
                    result[
                        "codebook_distance"
                    ]
            })

            logger.info(
                f"[RESULT] "
                f"target={target:.3f} "
                f"soft={result['soft_phi']:.4f} "
                f"fixed={result['fixed_phi']:.4f} "
                f"AQB={result['aqb_phi']:.4f} "
                f"cb_dist={result['codebook_distance']}"
            )

    summary_csv = os.path.join(
        out_dir,
        "condition_sweep_summary.csv"
    )

    with open(
        summary_csv,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        w = csv.DictWriter(
            f,
            fieldnames=list(
                summary[0].keys()
            )
        )

        w.writeheader()
        w.writerows(
            summary
        )

    logger.info(
        f"[SAMPLE] summary -> "
        f"{summary_csv}"
    )


# ============================================================
# 21. CLI
# ============================================================

def parse_args():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--stage",
        default="vqvae",
        choices=[
            "vqvae",
            "stats",
            "ddpm",
            "sample",
            "all"
        ]
    )

    p.add_argument(
        "--raw_path",
        default="data/raw/Bei_800x800x800.raw"
    )

    p.add_argument(
        "--save_dir",
        default="./ckpt_spatial_vqaware_v2"
    )

    p.add_argument(
        "--device",
        default="cuda"
    )

    p.add_argument(
        "--variant",
        default="vqaware",
        choices=[
            "vqaware",
            "vq",
            "continuous"
        ]
    )

    p.add_argument(
        "--condition_mode",
        default="film",
        choices=[
            "film",
            "none"
        ]
    )

    p.add_argument(
        "--epochs_vae",
        type=int,
        default=80
    )

    p.add_argument(
        "--epochs_ddpm",
        type=int,
        default=150
    )

    p.add_argument(
        "--n_train_samples",
        type=int,
        default=1000
    )

    p.add_argument(
        "--n_val_samples",
        type=int,
        default=64
    )

    p.add_argument(
        "--batch_vae",
        type=int,
        default=1
    )

    p.add_argument(
        "--batch_ddpm",
        type=int,
        default=1
    )

    p.add_argument(
        "--lambda_codebook",
        type=float,
        default=0.10
    )

    p.add_argument(
        "--codebook_sample_voxels",
        type=int,
        default=8192
    )

    p.add_argument(
        "--local_phi_weight",
        type=float,
        default=0.10
    )

    p.add_argument(
        "--boundary_weight",
        type=float,
        default=0.05
    )

    p.add_argument(
        "--dice_weight",
        type=float,
        default=0.20
    )

    p.add_argument(
        "--sample_targets",
        default="0.11,0.12,0.13,0.14"
    )

    p.add_argument(
        "--samples_per_target",
        type=int,
        default=5
    )

    p.add_argument(
        "--sample_checkpoint",
        default="unet_best_noise.pth",
        choices=[
            "unet_best_noise.pth",
            "unet_best.pth",
            "unet_final.pth"
        ]
    )

    p.add_argument(
        "--seed",
        type=int,
        default=20260928
    )

    p.add_argument(
        "--num_workers",
        type=int,
        default=0
    )

    p.add_argument(
        "--save_interval",
        type=int,
        default=5
    )

    p.add_argument(
        "--log_interval",
        type=int,
        default=20
    )

    p.add_argument(
        "--stage_tag",
        default=""
    )

    # ---- added by the FiLM iteration (see the header) ---------------------
    # In the parent these three are CFG DEFAULTS ONLY.  The arm B baseline
    # ran at (0.13, 0.02, 5.0) and there was no way to vary them, which is
    # why the first question of this study ("does the condition scaling
    # matter?") could not be asked of the frozen script.
    p.add_argument("--poro_center", type=float, default=0.13)
    p.add_argument("--poro_scale", type=float, default=0.02)
    p.add_argument("--poro_clip", type=float, default=5.0)

    # Warm-start the DDPM from a checkpoint.  The parent always builds a
    # fresh UNet + fresh Adam, so every round would cost a full from-scratch
    # training; exploration needs many short rounds instead.
    p.add_argument("--init_unet", default="")

    p.add_argument(
        "--amp",
        action="store_true"
    )

    p.add_argument(
        "--no_amp",
        action="store_true"
    )

    return p.parse_args()


# ============================================================
# 22. Main
# ============================================================

def main():

    args = parse_args()

    cfg = CFG()

    cfg.raw_path = args.raw_path
    cfg.save_dir = args.save_dir
    cfg.device = args.device

    cfg.variant = args.variant
    cfg.condition_mode = args.condition_mode

    cfg.epochs_vae = args.epochs_vae
    cfg.epochs_ddpm = args.epochs_ddpm

    cfg.n_train_samples = args.n_train_samples
    cfg.n_val_samples = args.n_val_samples

    cfg.batch_size_vae = args.batch_vae
    cfg.batch_size_ddpm = args.batch_ddpm

    cfg.lambda_codebook = args.lambda_codebook

    cfg.codebook_sample_voxels = (
        args.codebook_sample_voxels
    )

    cfg.local_phi_weight = (
        args.local_phi_weight
    )

    cfg.boundary_weight = (
        args.boundary_weight
    )

    cfg.dice_weight = (
        args.dice_weight
    )

    cfg.sample_targets = (
        args.sample_targets
    )

    cfg.samples_per_target = (
        args.samples_per_target
    )

    cfg.sample_checkpoint = (
        args.sample_checkpoint
    )

    cfg.seed = args.seed
    cfg.num_workers = args.num_workers
    cfg.save_interval = args.save_interval
    cfg.log_interval = args.log_interval
    cfg.stage_tag = args.stage_tag

    cfg.poro_center = args.poro_center
    cfg.poro_scale = args.poro_scale
    cfg.poro_clip = args.poro_clip
    cfg.init_unet = args.init_unet

    if args.no_amp:
        cfg.amp = False

    elif args.amp:
        cfg.amp = True

    os.makedirs(
        cfg.save_dir,
        exist_ok=True
    )

    logger = setup_logger(
        cfg.save_dir
    )

    validate_split(
        cfg,
        logger
    )

    save_cfg(cfg)
    save_split_manifest(cfg)

    torch.manual_seed(
        cfg.seed
    )

    np.random.seed(
        cfg.seed
    )

    if (
        cfg.device.startswith("cuda")
        and torch.cuda.is_available()
    ):

        logger.info(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        # benchmark=True makes cuDNN autotune per-shape algorithm
        # selection, which is run-to-run nondeterministic in wall time
        # and in kernel choice.  Fixed to False so that a re-run of the
        # same arm is comparable.
        torch.backends.cudnn.benchmark = False

    random.seed(cfg.seed)

    logger.info(
        "==== RUN CONFIG ===="
    )

    logger.info(
        str(cfg)
    )

    try:

        if args.stage == "vqvae":

            train_vae(
                cfg,
                logger
            )

        elif args.stage == "stats":

            compute_latent_stats(
                cfg,
                logger
            )

        elif args.stage == "ddpm":

            train_ddpm(
                cfg,
                logger
            )

        elif args.stage == "sample":

            sample_grid(
                cfg,
                logger
            )

        elif args.stage == "all":

            train_vae(
                cfg,
                logger
            )

            compute_latent_stats(
                cfg,
                logger
            )

            train_ddpm(
                cfg,
                logger
            )

            sample_grid(
                cfg,
                logger
            )

        logger.info(
            "ALL DONE."
        )

    except KeyboardInterrupt:

        logger.info(
            "Interrupted by user."
        )

    except Exception as e:

        logger.error(
            f"CRASH: {e}"
        )

        logger.error(
            traceback.format_exc()
        )

        raise


if __name__ == "__main__":
    main()
