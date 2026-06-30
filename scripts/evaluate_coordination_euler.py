import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile
import imageio.v2 as imageio

from scipy import ndimage as ndi
from skimage import measure

import porespy as ps


# =========================================================
# Basic utils
# =========================================================
def safe_mkdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def discover_sample_items(phi_dir: Path):
    """
    Scan all samples under one phi folder.
    Supported inputs:
      - Single-file samples: .npy .npz .tif .tiff .raw .bin
      - A directory can represent one sample if it contains slice images.
    """
    valid_suffixes = {".npy", ".npz", ".tif", ".tiff", ".raw", ".bin"}

    if not phi_dir.exists():
        return []

    items = []
    for p in sorted(phi_dir.iterdir()):
        if p.name.startswith("."):
            continue
        if p.is_dir():
            items.append(p)
        elif p.is_file() and p.suffix.lower() in valid_suffixes:
            items.append(p)
    return items


# =========================================================
# Volume IO
# =========================================================
def load_volume_from_dir(dir_path: Path):
    """
    If the sample is a directory, read and stack its slice images.
    """
    valid_suffixes = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}
    files = sorted([p for p in dir_path.iterdir() if p.suffix.lower() in valid_suffixes])

    if len(files) == 0:
        raise ValueError(f"No recognizable slice images were found in directory sample: {dir_path}")

    stack = []
    for f in files:
        arr = imageio.imread(f)
        if arr.ndim == 3:
            arr = arr[..., 0]
        stack.append(arr)
    vol = np.stack(stack, axis=0)
    return vol


def choose_npz_key(data, npz_key="auto"):
    """
    Choose the correct 3D volume key from an .npz file.
    Generated samples should usually prefer the 'seg' key.
    """
    if npz_key != "auto":
        if npz_key not in data.files:
            raise ValueError(f"Specified npz_key={npz_key}  does not exist; available keys={data.files}")
        return npz_key

    preferred_keys = [
        "seg",            # Preferred key for generated binary samples.
        "seg_quantile",
        "binary",
        "volume",
        "image",
        "arr_0",
        "prob",           # Last fallback; avoid using probability fields for final topology metrics.
    ]

    for k in preferred_keys:
        if k in data.files and np.asarray(data[k]).ndim == 3:
            return k

    # Final fallback: use the first 3D array.
    for k in data.files:
        if np.asarray(data[k]).ndim == 3:
            return k

    raise ValueError(f"No usable 3D volume found in npz; keys={data.files}")


def load_volume(path: Path, raw_shape=(256, 256, 256), raw_dtype="uint8", npz_key="auto", verbose=False):
    suffix = path.suffix.lower()
    meta = {}

    if path.is_dir():
        vol = load_volume_from_dir(path)

    elif suffix == ".npy":
        vol = np.load(path)

    elif suffix == ".npz":
        data = np.load(path)
        if len(data.files) == 0:
            raise ValueError(f"No arrays found in npz file: {path}")

        chosen_key = choose_npz_key(data, npz_key=npz_key)
        vol = data[chosen_key]

        meta["npz_keys"] = list(data.files)
        meta["chosen_key"] = chosen_key

        if "target_porosity" in data.files:
            meta["target_porosity"] = float(data["target_porosity"])
        if "seg_porosity" in data.files:
            meta["seg_porosity"] = float(data["seg_porosity"])

        if verbose:
            print(f"[INFO] npz chosen key for {path.name}: {chosen_key}")
            if "target_porosity" in meta:
                print(f"[INFO]   target_porosity: {meta['target_porosity']}")
            if "seg_porosity" in meta:
                print(f"[INFO]   seg_porosity: {meta['seg_porosity']}")

    elif suffix in [".tif", ".tiff"]:
        vol = tifffile.imread(path)

    elif suffix in [".raw", ".bin"]:
        arr = np.fromfile(path, dtype=np.dtype(raw_dtype))
        expected = int(np.prod(raw_shape))
        if arr.size != expected:
            raise ValueError(
                f"RAW size mismatch: {path}\n"
                f"actual elements={arr.size}, expected={expected}, shape={raw_shape}, dtype={raw_dtype}"
            )
        vol = arr.reshape(raw_shape)

    else:
        raise ValueError(f"Unsupported sample format: {path}")

    vol = np.asarray(vol)

    if vol.ndim != 3:
        raise ValueError(f"Sample is not a 3D volume: {path}, shape={vol.shape}")

    return vol, meta


def binarize_volume(vol, invert=False, threshold=None):
    """
    Data convention:
      0 = solid matrix
      1 = pore space

    Therefore, by default True = pore = (vol == 1).
    """
    vol = np.asarray(vol)
    unique_vals = np.unique(vol)

    # Standard binary 0/1 values.
    if set(unique_vals.tolist()).issubset({0, 1}):
        bw = (vol == 1)

    # Compatible with 0/255 binary values.
    elif set(unique_vals.tolist()).issubset({0, 255}):
        bw = (vol == 255)

    # Other numeric ranges use thresholding.
    else:
        if threshold is None:
            threshold = (float(vol.min()) + float(vol.max())) / 2.0
        bw = vol > threshold

    if invert:
        bw = ~bw

    return bw.astype(bool)


# =========================================================
# Topology features
# =========================================================
def coordination_distribution(volume_bool, max_coord=8):
    """
    Extract the network with PoreSpy SNOW2, then compute each pore coordination number.
    """
    snow = ps.networks.snow2(phases=volume_bool)
    net = snow.network if hasattr(snow, "network") else snow

    if "throat.conns" not in net:
        raise ValueError("Network extraction failed: throat.conns was not found.")

    conns = np.asarray(net["throat.conns"])
    n_pores = int(conns.max()) + 1 if conns.size > 0 else 0

    if n_pores == 0:
        hist = np.zeros(max_coord + 1, dtype=float)
        return np.arange(max_coord + 1), hist

    coord_num = np.bincount(conns.ravel(), minlength=n_pores)
    hist = np.bincount(np.clip(coord_num, 0, max_coord), minlength=max_coord + 1).astype(float)
    prob = hist / hist.sum() if hist.sum() > 0 else hist

    return np.arange(max_coord + 1), prob


def specific_euler_curve(
    volume_bool,
    voxel_size_um=3.5,
    radius_step_um=3.5,
    radius_max_um=None,
    normalize_by="total_voxels",
):
    """
    Compute the specific Euler characteristic versus pore-radius curve.
    Radius is reported in micrometers.
    """
    edt = ndi.distance_transform_edt(volume_bool) * voxel_size_um
    pore_vals = edt[edt > 0]

    if pore_vals.size == 0:
        raise ValueError("The sample has no pore voxels, so the Euler curve cannot be computed.")

    if radius_max_um is None:
        radius_max_um = float(np.percentile(pore_vals, 99))

    radii = np.arange(0, radius_max_um + 1e-9, radius_step_um)

    if normalize_by == "total_voxels":
        denom = volume_bool.size
    elif normalize_by == "pore_voxels":
        denom = int(volume_bool.sum())
    else:
        raise ValueError("normalize_by must be either total_voxels or pore_voxels.")

    denom = max(denom, 1)

    curve = []
    for r in radii:
        mask = edt >= r
        chi = measure.euler_number(mask.astype(np.uint8), connectivity=3)
        curve.append(chi / denom * 1e3)

    return radii, np.asarray(curve, dtype=float)


def process_one_sample(
    sample_path,
    invert=False,
    threshold=None,
    raw_shape=(256, 256, 256),
    raw_dtype="uint8",
    voxel_size_um=3.5,
    radius_step_um=3.5,
    radius_max_um=None,
    max_coord=8,
    normalize_by="total_voxels",
    npz_key="auto",
    verbose=False,
):
    vol, meta = load_volume(
        sample_path,
        raw_shape=raw_shape,
        raw_dtype=raw_dtype,
        npz_key=npz_key,
        verbose=verbose,
    )
    bw = binarize_volume(vol, invert=invert, threshold=threshold)

    coord_x, coord_y = coordination_distribution(bw, max_coord=max_coord)

    radii, euler_curve = specific_euler_curve(
        bw,
        voxel_size_um=voxel_size_um,
        radius_step_um=radius_step_um,
        radius_max_um=radius_max_um,
        normalize_by=normalize_by,
    )

    result = {
        "coord_x": coord_x,
        "coord_y": coord_y,
        "radii": radii,
        "euler": euler_curve,
        "porosity": float(bw.mean()),
    }
    result.update(meta)
    return result


# =========================================================
# Aggregation
# =========================================================
def interpolate_curve(x_src, y_src, x_common):
    y_common = np.full_like(x_common, np.nan, dtype=float)
    valid = (x_common >= x_src.min()) & (x_common <= x_src.max())
    if np.any(valid):
        y_common[valid] = np.interp(x_common[valid], x_src, y_src)
    return y_common


def aggregate_group(sample_results, radius_step_um=3.5, max_coord=8):
    if len(sample_results) == 0:
        raise ValueError("aggregate_group received an empty sample list.")

    coord_mat = np.stack([r["coord_y"] for r in sample_results], axis=0)
    coord_mean = np.mean(coord_mat, axis=0)
    coord_std = np.std(coord_mat, axis=0)

    global_r_max = max(r["radii"].max() for r in sample_results)
    common_radii = np.arange(0, global_r_max + 1e-9, radius_step_um)

    euler_mat = []
    for r in sample_results:
        euler_interp = interpolate_curve(r["radii"], r["euler"], common_radii)
        euler_mat.append(euler_interp)
    euler_mat = np.stack(euler_mat, axis=0)

    euler_mean = np.nanmean(euler_mat, axis=0)
    euler_std = np.nanstd(euler_mat, axis=0)

    porosity_vals = np.array([r["porosity"] for r in sample_results], dtype=float)

    return {
        "coord_x": np.arange(max_coord + 1),
        "coord_mean": coord_mean,
        "coord_std": coord_std,
        "radii": common_radii,
        "euler_mean": euler_mean,
        "euler_std": euler_std,
        "porosity_mean": float(np.mean(porosity_vals)),
        "porosity_std": float(np.std(porosity_vals)),
        "n_samples": len(sample_results),
    }


# =========================================================
# Plot
# =========================================================
def plot_phi_average_figure(phi_name, real_stats, gen_stats, save_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax0, ax1 = axes

    # Left panel: mean coordination-number distribution.
    x = real_stats["coord_x"]
    width = 0.35

    ax0.bar(
        x - width / 2,
        real_stats["coord_mean"],
        width=width,
        yerr=real_stats["coord_std"],
        capsize=3,
        label=f"Real (n={real_stats['n_samples']})",
    )
    ax0.bar(
        x + width / 2,
        gen_stats["coord_mean"],
        width=width,
        yerr=gen_stats["coord_std"],
        capsize=3,
        label=f"Generated (n={gen_stats['n_samples']})",
    )

    ax0.set_xticks(x)
    ax0.set_xticklabels([str(i) for i in x])
    ax0.set_xlabel("Coordination number")
    ax0.set_ylabel("Mean frequency / probability")
    ax0.set_title(f"{phi_name} - Mean coordination distribution")
    ax0.legend(frameon=False, fontsize=9)
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)

    # Right panel: mean Euler-characteristic curve.
    rr = real_stats["radii"]
    rg = gen_stats["radii"]

    ax1.plot(rr, real_stats["euler_mean"], linewidth=2, label=f"Real (n={real_stats['n_samples']})")
    ax1.fill_between(
        rr,
        real_stats["euler_mean"] - real_stats["euler_std"],
        real_stats["euler_mean"] + real_stats["euler_std"],
        alpha=0.2,
    )

    ax1.plot(rg, gen_stats["euler_mean"], linewidth=2, label=f"Generated (n={gen_stats['n_samples']})")
    ax1.fill_between(
        rg,
        gen_stats["euler_mean"] - gen_stats["euler_std"],
        gen_stats["euler_mean"] + gen_stats["euler_std"],
        alpha=0.2,
    )

    ax1.set_xlabel("Pore radius (um)")
    ax1.set_ylabel("Specific Euler characteristic (x1e-3)")
    ax1.set_title(f"{phi_name} - Mean specific Euler characteristic")
    ax1.legend(frameon=False, fontsize=9)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    poro_text = (
        f"Real porosity = {real_stats['porosity_mean']:.4f} +/- {real_stats['porosity_std']:.4f}\n"
        f"Gen porosity  = {gen_stats['porosity_mean']:.4f} +/- {gen_stats['porosity_std']:.4f}"
    )
    fig.suptitle(f"Topology average comparison for {phi_name}", fontsize=13, y=1.02)
    fig.text(0.5, -0.02, poro_text, ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# Main
# =========================================================
def run_pipeline(args):
    real_root = Path(args.real_root)
    gen_root = Path(args.gen_root)
    out_root = Path(args.out_root)

    safe_mkdir(out_root)
    safe_mkdir(out_root / "figures")
    safe_mkdir(out_root / "per_sample_curves")
    safe_mkdir(out_root / "group_summary")

    summary_rows = []

    for phi in args.phis:
        print(f"\n========== Processing {phi} ==========")

        real_phi_dir = real_root / phi
        gen_phi_dir = gen_root / phi

        real_items = discover_sample_items(real_phi_dir)
        gen_items = discover_sample_items(gen_phi_dir)

        if len(real_items) == 0:
            print(f"[WARN] {phi}  has no real samples: {real_phi_dir}")
            continue
        if len(gen_items) == 0:
            print(f"[WARN] {phi}  has no generated samples: {gen_phi_dir}")
            continue

        print(f"[INFO] {phi} real samples: {len(real_items)}")
        print(f"[INFO] {phi} gen  samples: {len(gen_items)}")

        real_results = []
        gen_results = []

        # -------- real --------
        for idx, sample_path in enumerate(real_items, start=1):
            print(f"[{phi}] REAL {idx}/{len(real_items)}: {sample_path}")
            try:
                res = process_one_sample(
                    sample_path=sample_path,
                    invert=args.invert,
                    threshold=args.threshold,
                    raw_shape=tuple(args.raw_shape),
                    raw_dtype=args.raw_dtype,
                    voxel_size_um=args.voxel_size_um,
                    radius_step_um=args.radius_step_um,
                    radius_max_um=args.radius_max_um,
                    max_coord=args.max_coord,
                    normalize_by=args.normalize_by,
                    npz_key=args.npz_key,
                    verbose=args.verbose_npz,
                )
                res["label"] = f"real_{idx}"
                res["sample_path"] = str(sample_path)
                real_results.append(res)

                pd.DataFrame({
                    "coordination_number": res["coord_x"],
                    "probability": res["coord_y"],
                }).to_csv(out_root / "per_sample_curves" / f"{phi}_real_{idx}_coord.csv", index=False)

                pd.DataFrame({
                    "radius_um": res["radii"],
                    "specific_euler_x1e3": res["euler"],
                }).to_csv(out_root / "per_sample_curves" / f"{phi}_real_{idx}_euler.csv", index=False)

            except Exception as e:
                print(f"[ERROR] real sample failed: {sample_path}\n{e}")

        # -------- generated --------
        for idx, sample_path in enumerate(gen_items, start=1):
            print(f"[{phi}] GEN  {idx}/{len(gen_items)}: {sample_path}")
            try:
                res = process_one_sample(
                    sample_path=sample_path,
                    invert=args.invert,
                    threshold=args.threshold,
                    raw_shape=tuple(args.raw_shape),
                    raw_dtype=args.raw_dtype,
                    voxel_size_um=args.voxel_size_um,
                    radius_step_um=args.radius_step_um,
                    radius_max_um=args.radius_max_um,
                    max_coord=args.max_coord,
                    normalize_by=args.normalize_by,
                    npz_key=args.npz_key,
                    verbose=args.verbose_npz,
                )
                res["label"] = f"gen_{idx}"
                res["sample_path"] = str(sample_path)
                gen_results.append(res)

                pd.DataFrame({
                    "coordination_number": res["coord_x"],
                    "probability": res["coord_y"],
                }).to_csv(out_root / "per_sample_curves" / f"{phi}_gen_{idx}_coord.csv", index=False)

                pd.DataFrame({
                    "radius_um": res["radii"],
                    "specific_euler_x1e3": res["euler"],
                }).to_csv(out_root / "per_sample_curves" / f"{phi}_gen_{idx}_euler.csv", index=False)

            except Exception as e:
                print(f"[ERROR] gen sample failed: {sample_path}\n{e}")

        if len(real_results) == 0 or len(gen_results) == 0:
            print(f"[WARN] {phi}  has one group without successful results; skipping plot.")
            continue

        real_stats = aggregate_group(
            real_results,
            radius_step_um=args.radius_step_um,
            max_coord=args.max_coord,
        )
        gen_stats = aggregate_group(
            gen_results,
            radius_step_um=args.radius_step_um,
            max_coord=args.max_coord,
        )

        # Save mean coordination-number distribution.
        pd.DataFrame({
            "coordination_number": real_stats["coord_x"],
            "real_mean": real_stats["coord_mean"],
            "real_std": real_stats["coord_std"],
            "gen_mean": gen_stats["coord_mean"],
            "gen_std": gen_stats["coord_std"],
        }).to_csv(out_root / "group_summary" / f"{phi}_coordination_mean_std.csv", index=False)

        # Save mean Euler curve.
        df_real_euler = pd.DataFrame({
            "radius_um": real_stats["radii"],
            "real_mean": real_stats["euler_mean"],
            "real_std": real_stats["euler_std"],
        })
        df_gen_euler = pd.DataFrame({
            "radius_um": gen_stats["radii"],
            "gen_mean": gen_stats["euler_mean"],
            "gen_std": gen_stats["euler_std"],
        })
        df_euler = pd.merge(df_real_euler, df_gen_euler, on="radius_um", how="outer")
        df_euler.to_csv(out_root / "group_summary" / f"{phi}_euler_mean_std.csv", index=False)

        fig_path = out_root / "figures" / f"{phi}_mean_compare.png"
        plot_phi_average_figure(phi, real_stats, gen_stats, fig_path)

        summary_rows.append({
            "phi": phi,
            "real_n": real_stats["n_samples"],
            "gen_n": gen_stats["n_samples"],
            "real_porosity_mean": real_stats["porosity_mean"],
            "real_porosity_std": real_stats["porosity_std"],
            "gen_porosity_mean": gen_stats["porosity_mean"],
            "gen_porosity_std": gen_stats["porosity_std"],
            "figure_path": str(fig_path),
        })

    pd.DataFrame(summary_rows).to_csv(out_root / "summary_all_phi.csv", index=False)
    print(f"\nDone. Results saved to: {out_root}")


# =========================================================
# CLI
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--real-root", type=str, required=True, help="Root directory for real samples.")
    parser.add_argument("--gen-root", type=str, required=True, help="Root directory for generated samples.")
    parser.add_argument("--out-root", type=str, required=True, help="Output directory.")

    parser.add_argument(
        "--phis",
        nargs="+",
        default=["phi0p11", "phi0p12", "phi0p13", "phi0p14", "phi0p15"],
        help="Porosity folder names to process.",
    )

    parser.add_argument("--voxel-size-um", type=float, default=3.5, help="Voxel resolution; default is 3.5 um.")
    parser.add_argument("--radius-step-um", type=float, default=3.5, help="Radius step for the right panel; default is 3.5 um.")
    parser.add_argument("--radius-max-um", type=float, default=None, help="Maximum radius for the right panel; automatic by default.")
    parser.add_argument("--max-coord", type=int, default=8)
    parser.add_argument("--normalize-by", type=str, default="total_voxels", choices=["total_voxels", "pore_voxels"])

    parser.add_argument("--invert", action="store_true", help="Use only if the pore/solid semantics are inverted.")
    parser.add_argument("--threshold", type=float, default=None)

    parser.add_argument("--raw-shape", nargs=3, type=int, default=[256, 256, 256])
    parser.add_argument("--raw-dtype", type=str, default="uint8")

    parser.add_argument("--npz-key", type=str, default="auto", help="NPZ key to use; default is auto. Use auto or seg for generated samples.")
    parser.add_argument("--verbose-npz", action="store_true", help="Print the selected npz key and metadata.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)