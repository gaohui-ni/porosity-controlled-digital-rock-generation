"""Regenerate the frozen Test-95 probability volumes from released weights.

Each real crop is sampled once at its own measured porosity. The saved
probability volume is also converted to the fixed-threshold and conventional
exact top-N baselines. TP-AQB v2 is a separate downstream step implemented by
``scripts/apply_tp_aqb.py``.
"""

import argparse
import csv
import hashlib
import importlib.util
import logging
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "savedmodels" / "spatial_vqaware_v2"
GROUP_SIZES = {"0.11": 8, "0.12": 29, "0.13": 39, "0.14": 19}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Regenerate the released spatial Test-95 cohort."
    )
    parser.add_argument(
        "--raw-path",
        default="data/raw/Bei_800x800x800.raw",
        help="Published binary uint8 800^3 sandstone volume.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=str(RELEASE / "R03_calibrated"),
    )
    parser.add_argument("--checkpoint", default="calib_A_scale.pth")
    parser.add_argument(
        "--cohort-csv",
        default=str(RELEASE / "test_cohort_v1" / "test_crop_porosity_all_100.csv"),
    )
    parser.add_argument(
        "--sampler-script",
        default=str(ROOT / "scripts" / "train_256_vqvae_ddpm_spatial_vqaware_v3_film.py"),
    )
    parser.add_argument("--outdir", default="outputs/spatial_test95/gen")
    parser.add_argument("--manifest-dir", default="outputs/spatial_test95")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed-base", type=int, default=2026093000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_sampler(path):
    spec = importlib.util.spec_from_file_location("released_spatial_sampler", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import sampler script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_cohort(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    keep = sorted(
        (row for row in rows if row["target_group"] != "none"),
        key=lambda row: row["crop_id"],
    )
    groups = {
        group: sum(row["target_group"] == group for row in keep)
        for group in GROUP_SIZES
    }
    if len(rows) != 100 or len(keep) != 95 or groups != GROUP_SIZES:
        raise ValueError(
            f"Unexpected cohort: total={len(rows)}, used={len(keep)}, groups={groups}"
        )
    if len({row["crop_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate crop_id values in cohort manifest")
    return keep


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    raw_path = Path(args.raw_path).resolve()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    checkpoint_path = checkpoint_dir / args.checkpoint
    cohort_path = Path(args.cohort_csv).resolve()
    sampler_path = Path(args.sampler_script).resolve()
    outdir = Path(args.outdir).resolve()
    manifest_dir = Path(args.manifest_dir).resolve()

    required = [
        raw_path,
        checkpoint_path,
        checkpoint_dir / "ae_best.pth",
        checkpoint_dir / "latent_stats.npz",
        cohort_path,
        sampler_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

    outdir.mkdir(parents=True, exist_ok=True)
    if not args.overwrite and any(outdir.glob("*_prob.npz")):
        raise FileExistsError(
            f"Existing probability files found in {outdir}; use --overwrite to replace"
        )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(outdir / "test95.log", mode="a", encoding="utf-8"),
        ],
    )
    logger = logging.getLogger("test95")
    sampler = load_sampler(sampler_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = sampler.CFG(**dict(checkpoint["cfg"]))
    cfg.save_dir = str(checkpoint_dir)
    cfg.device = args.device
    if cfg.condition_mode != "film":
        raise ValueError(f"Expected condition_mode=film, got {cfg.condition_mode}")
    if abs(cfg.poro_scale - 0.011804) >= 1e-12:
        raise ValueError(f"Expected calibrated poro_scale=0.011804, got {cfg.poro_scale}")
    if cfg.fixed_threshold != 0.5:
        raise ValueError(f"Expected fixed_threshold=0.5, got {cfg.fixed_threshold}")

    ae_hash = hashlib.sha256((checkpoint_dir / "ae_best.pth").read_bytes()).hexdigest()
    logger.info("released checkpoint loaded; ae_sha256=%s", ae_hash)

    cohort = read_cohort(cohort_path)
    if args.limit:
        cohort = cohort[: args.limit]

    ae = sampler.load_best_ae(cfg, logger)
    latent_mean, latent_std = sampler.load_latent_stats(cfg, logger)
    unet = sampler.UNetLatentCond(
        channels=cfg.embedding_dim,
        base=cfg.unet_base,
        time_dim=cfg.unet_time_dim,
    ).to(cfg.device)
    unet.load_state_dict(checkpoint["model_state_dict"])
    unet.eval()

    raw = np.memmap(raw_path, dtype=np.uint8, mode="r").reshape(tuple(cfg.raw_shape))
    patch = int(cfg.patch_size)
    seed_rows = []
    pair_rows = []

    for index, row in enumerate(cohort):
        crop_id = row["crop_id"]
        x0, y0, z0 = (int(row[key]) for key in ("x0", "y0", "z0"))
        real_porosity = float(row["phi"])
        seed = args.seed_base + index

        with torch.no_grad():
            result = sampler.sample_single(
                cfg,
                ae,
                unet,
                latent_mean,
                latent_std,
                real_porosity,
                seed,
                logger,
            )

        probability = result["prob"].astype(np.float32)
        fixed = result["seg_fixed"]
        top_n = result["seg_aqb"]
        real = np.asarray(
            raw[x0 : x0 + patch, y0 : y0 + patch, z0 : z0 + patch]
        ).astype(np.uint8)
        if not np.all(np.isin(np.unique(real), (0, 1))):
            raise ValueError(f"{crop_id}: real crop is not binary")
        if not np.array_equal(
            (probability >= cfg.fixed_threshold).astype(np.uint8), fixed
        ):
            raise AssertionError(f"{crop_id}: fixed branch is not derived from P")
        if not np.array_equal(sampler.aqb_exact(probability, real_porosity), top_n):
            raise AssertionError(f"{crop_id}: top-N branch is not derived from P")

        flat = probability.reshape(-1)
        count = int(np.clip(round(real_porosity * flat.size), 0, flat.size))
        threshold = (
            float(np.partition(flat, flat.size - count)[flat.size - count])
            if 0 < count < flat.size
            else float("nan")
        )
        np.savez_compressed(
            outdir / f"{crop_id}_prob.npz",
            prob=probability,
            target_porosity=np.float64(real_porosity),
            seed=np.int64(seed),
            soft_phi=np.float64(result["soft_phi"]),
            fixed_phi=np.float64(result["fixed_phi"]),
            aqb_phi=np.float64(result["aqb_phi"]),
            codebook_distance=np.float64(result["codebook_distance"]),
            fixed_threshold=np.float64(cfg.fixed_threshold),
            aqb_threshold=np.float64(threshold),
        )
        real.tofile(outdir / f"{crop_id}_real.raw")
        fixed.tofile(outdir / f"{crop_id}_off.raw")
        top_n.tofile(outdir / f"{crop_id}_on.raw")

        seed_rows.append(
            {
                "index": index,
                "crop_id": crop_id,
                "group": row["target_group"],
                "phi_real": repr(real_porosity),
                "target_phi": repr(real_porosity),
                "seed": seed,
            }
        )
        pair_rows.append(
            {
                "crop_id": crop_id,
                "group": row["target_group"],
                "x0": x0,
                "y0": y0,
                "z0": z0,
                "phi_real": repr(real_porosity),
                "target_phi": repr(real_porosity),
                "seed": seed,
                "phi_pre_aqb": repr(float(fixed.mean())),
                "phi_post_aqb": repr(float(top_n.mean())),
                "soft_phi": repr(float(result["soft_phi"])),
                "fixed_threshold": cfg.fixed_threshold,
                "aqb_threshold": repr(threshold),
                "codebook_distance": repr(float(result["codebook_distance"])),
                "real_path": f"{crop_id}_real.raw",
                "prob_path": f"{crop_id}_prob.npz",
                "pre_aqb_path": f"{crop_id}_off.raw",
                "post_aqb_path": f"{crop_id}_on.raw",
            }
        )
        logger.info(
            "[%d/%d] %s phi=%.8f seed=%d fixed=%.6f top_n=%.6f",
            index + 1,
            len(cohort),
            crop_id,
            real_porosity,
            seed,
            fixed.mean(),
            top_n.mean(),
        )

    write_csv(manifest_dir / "final_test_seed_manifest.csv", seed_rows)
    write_csv(manifest_dir / "FINAL_TEST_PAIR_MANIFEST.csv", pair_rows)
    logger.info("Generated %d crops under %s", len(pair_rows), outdir)


if __name__ == "__main__":
    main()
