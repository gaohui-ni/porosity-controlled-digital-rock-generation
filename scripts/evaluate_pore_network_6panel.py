#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import json
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

import porespy as ps
import openpnm as op


# =========================================================
# 1) Global configuration
# =========================================================

REAL_ROOT = "data/real256_sets_from_S1_strict"
GEN_ROOT  = "data/generated_phi_sets"
OUT_ROOT  = "results/pore_network_6panel"

TARGETS = [
    ("phi0p11", 0.11),
    ("phi0p12", 0.12),
    ("phi0p13", 0.13),
    ("phi0p14", 0.14),
    ("phi0p15", 0.15),
]

# -------- real: raw --------
REAL_FILE_EXT = ".raw"
RAW_SHAPE = (256, 256, 256)
RAW_DTYPE = np.uint8
REAL_PORE_VALUE = 1
REAL_INVERT = False

# -------- gen: npz --------
GEN_FILE_EXT = ".npz"
GEN_NPZ_KEY = None          # Keep None if the key is unknown; the first array is used by default.
GEN_PORE_VALUE = 1
GEN_USE_THRESHOLD = False   # Set True if generated samples are probability fields.
GEN_THRESHOLD = 0.5
GEN_INVERT = False

# -------- Sample-count limits --------
# None = all samples
MAX_REAL_FILES = None
MAX_GEN_FILES = None

# -------- EDT parameters --------
EDT_MAX_RADIUS = 12.0
EDT_NBINS = 60

# -------- PNM parameters --------
DO_PNM = True
VOXEL_SIZE = 3.0e-6
PNM_ACCURACY = "standard"   # "high" more accurate but slower
SHAPE_FACTOR_MAX = 0.12
PNM_MAX_PORE_RADIUS = 30.0
PNM_MAX_THROAT_RADIUS = 15.0
PNM_MAX_THROAT_LENGTH = 80.0
PNM_NBINS = 60

# -------- Tortuosity parameters --------
DO_TORTUOSITY = True

# =========================================================
# 2) Basic functions
# =========================================================

def ensure_3d(arr, path=""):
    arr = np.asarray(arr)
    arr = np.squeeze(arr)
    if arr.ndim != 3:
        raise ValueError(f"{path} is not 3D data，shape={arr.shape}")
    return arr


def load_raw(path: Path):
    arr = np.fromfile(path, dtype=RAW_DTYPE)
    expected = int(np.prod(RAW_SHAPE))
    if arr.size != expected:
        raise ValueError(
            f"{path} element count mismatch: got {arr.size}, RAW_SHAPE={RAW_SHAPE} requires {expected}"
        )
    return arr.reshape(RAW_SHAPE)


def load_npz(path: Path, npz_key=None):
    data = np.load(path)
    keys = list(data.keys())
    if len(keys) == 0:
        raise ValueError(f"{path} contains no arrays")
    if npz_key is not None:
        if npz_key not in data:
            raise KeyError(f"{path} does not contain key={npz_key}，available keys={keys}")
        arr = data[npz_key]
    else:
        arr = data[keys[0]]
    return arr


def to_pore_mask(arr, pore_value=1, invert=False, use_threshold=False, threshold=0.5):
    arr = np.asarray(arr)
    if arr.dtype == np.bool_:
        pore = arr.copy()
    else:
        if use_threshold:
            pore = arr.astype(np.float32) > float(threshold)
        else:
            pore = (arr == pore_value)
    if invert:
        pore = ~pore
    return pore.astype(bool)


def collect_files(folder, ext, max_files=None):
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Directory does not exist: {folder}")
    files = sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ext.lower()])
    if not files:
        raise RuntimeError(f"{folder} has no {ext} files")
    if max_files is not None:
        files = files[:max_files]
    return files


def hist_curve(values, bin_edges):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.zeros(len(bin_edges) - 1, dtype=np.float64)
    hist, _ = np.histogram(values, bins=bin_edges, density=True)
    return hist.astype(np.float64)


def aggregate_curves(curves):
    arr = np.asarray(curves, dtype=np.float64)
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0, ddof=0)
    return mean, std


def aggregate_xyz_dicts(items):
    # items = [{"tau_x":..., "tau_y":..., "tau_z":...}, ...]
    arr = np.array([[d.get("tau_x", np.nan),
                     d.get("tau_y", np.nan),
                     d.get("tau_z", np.nan)] for d in items], dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0, ddof=0)
    return mean, std, arr


def edt_radius_curve(pore_mask, bin_edges):
    d = distance_transform_edt(pore_mask)
    radii = d[pore_mask]   # voxel
    curve = hist_curve(radii, bin_edges)
    return curve, radii


# =========================================================
# 3) PNM extraction
# =========================================================

def build_pnm_network(pore_mask):
    """
    Extract the network with snow2.
    No artificial boundary padding is added here to avoid boundary artifacts in distributions.
    Then label_boundaries is used to mark boundary pores for later filtering.
    """
    snow = ps.networks.snow2(
        phases=pore_mask.astype(np.uint8),
        voxel_size=VOXEL_SIZE,
        boundary_width=0,
        accuracy=PNM_ACCURACY,
    )

    net = snow.network
    try:
        net = ps.networks.label_boundaries(network=net)
    except Exception:
        pass

    pn = op.io.network_from_porespy(net)

    # Repair network health.
    try:
        health = op.utils.check_network_health(pn)
        if isinstance(health, dict):
            bad = health.get("disconnected_pores", [])
            if len(bad) > 0:
                op.topotools.trim(network=pn, pores=bad)
    except Exception:
        pass

    return pn


def get_interior_masks(pn):
    """
    Remove boundary pores/throats where possible to reduce boundary artifacts.
    """
    if "pore.boundary" in pn.keys():
        pore_interior = ~np.asarray(pn["pore.boundary"], dtype=bool)
    else:
        pore_interior = np.ones(pn.Np, dtype=bool)

    conns = np.asarray(pn["throat.conns"], dtype=int)
    throat_interior = pore_interior[conns].all(axis=1)

    return pore_interior, throat_interior


def safe_shape_factor_A_over_P2(area, perimeter):
    """
    Geometric shape factor = A / P^2.
    """
    area = np.asarray(area, dtype=float)
    perimeter = np.asarray(perimeter, dtype=float)
    out = np.full(area.shape, np.nan, dtype=float)
    good = np.isfinite(area) & np.isfinite(perimeter) & (area > 0) & (perimeter > 0)
    out[good] = area[good] / (perimeter[good] ** 2)
    return out


def pnm_distributions(pore_mask):
    pn = build_pnm_network(pore_mask)
    pore_interior, throat_interior = get_interior_masks(pn)

    pore_rad = np.asarray(pn["pore.equivalent_diameter"], dtype=float) / 2.0
    throat_rad = np.asarray(pn["throat.inscribed_diameter"], dtype=float) / 2.0
    throat_len = np.asarray(pn["throat.total_length"], dtype=float)
    throat_area = np.asarray(pn["throat.cross_sectional_area"], dtype=float)
    throat_perim = np.asarray(pn["throat.perimeter"], dtype=float)

    pore_rad_vox = pore_rad / VOXEL_SIZE
    throat_rad_vox = throat_rad / VOXEL_SIZE
    throat_len_vox = throat_len / VOXEL_SIZE
    throat_shape = safe_shape_factor_A_over_P2(throat_area, throat_perim)

    return {
        "pore_radius_voxel": pore_rad_vox[pore_interior & np.isfinite(pore_rad_vox)],
        "throat_radius_voxel": throat_rad_vox[throat_interior & np.isfinite(throat_rad_vox)],
        "throat_length_voxel": throat_len_vox[throat_interior & np.isfinite(throat_len_vox)],
        "throat_shape_factor": throat_shape[throat_interior & np.isfinite(throat_shape)],
        "n_pores": int(pn.Np),
        "n_throats": int(pn.Nt),
    }


# =========================================================
# 4) Tortuosity computed from image volumes
# =========================================================

def tortuosity_xyz_from_image(pore_mask):
    """
    Use PoreSpy tortuosity_fd directly on binary volumes to compute x/y/z tortuosity.
    """
    out = {"tau_x": np.nan, "tau_y": np.nan, "tau_z": np.nan}

    # For 3D images, axis=0/1/2 correspond to the three directions.
    for ax, name in zip([0, 1, 2], ["tau_x", "tau_y", "tau_z"]):
        try:
            res = ps.simulations.tortuosity_fd(im=pore_mask, axis=ax)
            out[name] = float(res.tortuosity)
        except Exception:
            out[name] = np.nan

    return out


# =========================================================
# 5) Plotting
# =========================================================

def plot_mean_std(ax, x, mean1, std1, mean2, std2,
                  label1="real", label2="gen",
                  xlabel="", ylabel="pdf", title=""):
    low1 = np.clip(mean1 - std1, 0, None)
    high1 = mean1 + std1
    low2 = np.clip(mean2 - std2, 0, None)
    high2 = mean2 + std2

    ax.plot(x, mean1, label=label1, linewidth=2)
    ax.fill_between(x, low1, high1, alpha=0.2)

    ax.plot(x, mean2, label=label2, linewidth=2)
    ax.fill_between(x, low2, high2, alpha=0.2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()


def plot_tortuosity_panel(ax, real_tau_mean, real_tau_std, gen_tau_mean, gen_tau_std):
    labels = ["x", "y", "z"]
    x = np.arange(3)
    width = 0.36

    ax.bar(x - width / 2, real_tau_mean, width, yerr=real_tau_std, capsize=4, label="real")
    ax.bar(x + width / 2, gen_tau_mean, width, yerr=gen_tau_std, capsize=4, label="gen")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("direction")
    ax.set_ylabel("tortuosity")
    ax.set_title("Tortuosity (x/y/z)")
    ax.legend()


# =========================================================
# 6) Process one sample group
# =========================================================

def process_real_group(files, edt_bins, pnm_bins):
    edt_curves = []
    edt_meta = []
    pnm_curves = {
        "pore_radius": [],
        "throat_radius": [],
        "throat_length": [],
        "shape_factor": [],
    }
    pnm_meta = []
    tau_list = []
    skipped = []

    for i, fp in enumerate(files, 1):
        print(f"[REAL] {i}/{len(files)} -> {fp.name}")
        try:
            arr = load_raw(fp)
            arr = ensure_3d(arr, fp)
            pore = to_pore_mask(
                arr,
                pore_value=REAL_PORE_VALUE,
                invert=REAL_INVERT,
                use_threshold=False,
                threshold=0.5,
            )

            curve_edt, radii_edt = edt_radius_curve(pore, edt_bins)
            edt_curves.append(curve_edt)
            edt_meta.append({
                "file": str(fp),
                "porosity": float(pore.mean()),
                "n_pore_vox": int(pore.sum()),
                "edt_radius_mean_vox": float(np.mean(radii_edt)) if radii_edt.size else np.nan,
            })

            if DO_PNM:
                pnm = pnm_distributions(pore)
                pnm_curves["pore_radius"].append(
                    hist_curve(pnm["pore_radius_voxel"], pnm_bins["pore_radius"])
                )
                pnm_curves["throat_radius"].append(
                    hist_curve(pnm["throat_radius_voxel"], pnm_bins["throat_radius"])
                )
                pnm_curves["throat_length"].append(
                    hist_curve(pnm["throat_length_voxel"], pnm_bins["throat_length"])
                )
                pnm_curves["shape_factor"].append(
                    hist_curve(pnm["throat_shape_factor"], pnm_bins["shape_factor"])
                )
                pnm_meta.append({
                    "file": str(fp),
                    "n_pores": pnm["n_pores"],
                    "n_throats": pnm["n_throats"],
                })

            if DO_TORTUOSITY:
                tau = tortuosity_xyz_from_image(pore)
                tau["file"] = str(fp)
                tau_list.append(tau)

        except Exception as e:
            skipped.append({"file": str(fp), "error": repr(e)})

    return {
        "edt_curves": edt_curves,
        "edt_meta": edt_meta,
        "pnm_curves": pnm_curves,
        "pnm_meta": pnm_meta,
        "tau_list": tau_list,
        "skipped": skipped,
    }


def process_gen_group(files, edt_bins, pnm_bins):
    edt_curves = []
    edt_meta = []
    pnm_curves = {
        "pore_radius": [],
        "throat_radius": [],
        "throat_length": [],
        "shape_factor": [],
    }
    pnm_meta = []
    tau_list = []
    skipped = []

    for i, fp in enumerate(files, 1):
        print(f"[GEN ] {i}/{len(files)} -> {fp.name}")
        try:
            arr = load_npz(fp, npz_key=GEN_NPZ_KEY)
            arr = ensure_3d(arr, fp)
            pore = to_pore_mask(
                arr,
                pore_value=GEN_PORE_VALUE,
                invert=GEN_INVERT,
                use_threshold=GEN_USE_THRESHOLD,
                threshold=GEN_THRESHOLD,
            )

            curve_edt, radii_edt = edt_radius_curve(pore, edt_bins)
            edt_curves.append(curve_edt)
            edt_meta.append({
                "file": str(fp),
                "porosity": float(pore.mean()),
                "n_pore_vox": int(pore.sum()),
                "edt_radius_mean_vox": float(np.mean(radii_edt)) if radii_edt.size else np.nan,
            })

            if DO_PNM:
                pnm = pnm_distributions(pore)
                pnm_curves["pore_radius"].append(
                    hist_curve(pnm["pore_radius_voxel"], pnm_bins["pore_radius"])
                )
                pnm_curves["throat_radius"].append(
                    hist_curve(pnm["throat_radius_voxel"], pnm_bins["throat_radius"])
                )
                pnm_curves["throat_length"].append(
                    hist_curve(pnm["throat_length_voxel"], pnm_bins["throat_length"])
                )
                pnm_curves["shape_factor"].append(
                    hist_curve(pnm["throat_shape_factor"], pnm_bins["shape_factor"])
                )
                pnm_meta.append({
                    "file": str(fp),
                    "n_pores": pnm["n_pores"],
                    "n_throats": pnm["n_throats"],
                })

            if DO_TORTUOSITY:
                tau = tortuosity_xyz_from_image(pore)
                tau["file"] = str(fp)
                tau_list.append(tau)

        except Exception as e:
            skipped.append({"file": str(fp), "error": repr(e)})

    return {
        "edt_curves": edt_curves,
        "edt_meta": edt_meta,
        "pnm_curves": pnm_curves,
        "pnm_meta": pnm_meta,
        "tau_list": tau_list,
        "skipped": skipped,
    }


# =========================================================
# 7) Single target
# =========================================================

def run_one_target(folder_name, target_value):
    real_dir = Path(REAL_ROOT) / folder_name
    gen_dir = Path(GEN_ROOT) / folder_name
    out_dir = Path(OUT_ROOT) / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 84)
    print(f"Processing {folder_name}  (target={target_value:.2f})")
    print("=" * 84)

    real_files = collect_files(real_dir, REAL_FILE_EXT, MAX_REAL_FILES)
    gen_files = collect_files(gen_dir, GEN_FILE_EXT, MAX_GEN_FILES)

    print(f"real files: {len(real_files)}")
    print(f"gen  files: {len(gen_files)}")
    print(f"DO_PNM: {DO_PNM}")
    print(f"DO_TORTUOSITY: {DO_TORTUOSITY}")

    # bins
    edt_bins = np.linspace(0.0, EDT_MAX_RADIUS, EDT_NBINS + 1)
    edt_centers = 0.5 * (edt_bins[:-1] + edt_bins[1:])

    pnm_bins = {
        "pore_radius": np.linspace(0.0, PNM_MAX_PORE_RADIUS, PNM_NBINS + 1),
        "throat_radius": np.linspace(0.0, PNM_MAX_THROAT_RADIUS, PNM_NBINS + 1),
        "throat_length": np.linspace(0.0, PNM_MAX_THROAT_LENGTH, PNM_NBINS + 1),
        "shape_factor": np.linspace(0.0, SHAPE_FACTOR_MAX, PNM_NBINS + 1),
    }
    pnm_centers = {k: 0.5 * (v[:-1] + v[1:]) for k, v in pnm_bins.items()}

    real = process_real_group(real_files, edt_bins, pnm_bins)
    gen = process_gen_group(gen_files, edt_bins, pnm_bins)

    if len(real["edt_curves"]) == 0 or len(gen["edt_curves"]) == 0:
        raise RuntimeError(f"{folder_name}: EDT  curve is empty; check the input.")

    # ---------- EDT ----------
    real_edt_mean, real_edt_std = aggregate_curves(real["edt_curves"])
    gen_edt_mean, gen_edt_std = aggregate_curves(gen["edt_curves"])

    # ---------- PNM ----------
    pnm_saved = False
    if DO_PNM:
        needed = ["pore_radius", "throat_radius", "throat_length", "shape_factor"]
        enough_real = all(len(real["pnm_curves"][k]) > 0 for k in needed)
        enough_gen = all(len(gen["pnm_curves"][k]) > 0 for k in needed)

        if not (enough_real and enough_gen):
            raise RuntimeError(f"{folder_name}: PNM  has at least one empty metric and cannot be plotted as six panels.")

        real_pr_mean, real_pr_std = aggregate_curves(real["pnm_curves"]["pore_radius"])
        gen_pr_mean, gen_pr_std = aggregate_curves(gen["pnm_curves"]["pore_radius"])

        real_tr_mean, real_tr_std = aggregate_curves(real["pnm_curves"]["throat_radius"])
        gen_tr_mean, gen_tr_std = aggregate_curves(gen["pnm_curves"]["throat_radius"])

        real_tl_mean, real_tl_std = aggregate_curves(real["pnm_curves"]["throat_length"])
        gen_tl_mean, gen_tl_std = aggregate_curves(gen["pnm_curves"]["throat_length"])

        real_sf_mean, real_sf_std = aggregate_curves(real["pnm_curves"]["shape_factor"])
        gen_sf_mean, gen_sf_std = aggregate_curves(gen["pnm_curves"]["shape_factor"])

        pnm_saved = True

    # ---------- Tortuosity ----------
    tau_saved = False
    if DO_TORTUOSITY:
        if len(real["tau_list"]) == 0 or len(gen["tau_list"]) == 0:
            raise RuntimeError(f"{folder_name}:  tortuosity result is empty and cannot be plotted as six panels.")

        real_tau_mean, real_tau_std, real_tau_arr = aggregate_xyz_dicts(real["tau_list"])
        gen_tau_mean, gen_tau_std, gen_tau_arr = aggregate_xyz_dicts(gen["tau_list"])
        tau_saved = True

    # ---------- Draw combined figure ----------
    if not (pnm_saved and tau_saved):
        raise RuntimeError(f"{folder_name}: results required by the six-panel plot are incomplete.")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    plot_mean_std(
        axes[0, 0],
        pnm_centers["pore_radius"],
        real_pr_mean, real_pr_std,
        gen_pr_mean, gen_pr_std,
        xlabel="pore radius (voxel)",
        ylabel="pdf",
        title="Pore-radius distribution",
    )

    plot_mean_std(
        axes[0, 1],
        pnm_centers["throat_radius"],
        real_tr_mean, real_tr_std,
        gen_tr_mean, gen_tr_std,
        xlabel="throat radius (voxel)",
        ylabel="pdf",
        title="Throat-radius distribution",
    )

    plot_mean_std(
        axes[0, 2],
        pnm_centers["throat_length"],
        real_tl_mean, real_tl_std,
        gen_tl_mean, gen_tl_std,
        xlabel="throat length (voxel)",
        ylabel="pdf",
        title="Throat-length distribution",
    )

    plot_mean_std(
        axes[1, 0],
        pnm_centers["shape_factor"],
        real_sf_mean, real_sf_std,
        gen_sf_mean, gen_sf_std,
        xlabel="shape factor (A/P^2)",
        ylabel="pdf",
        title="Throat shape-factor distribution",
    )

    plot_tortuosity_panel(
        axes[1, 1],
        real_tau_mean, real_tau_std,
        gen_tau_mean, gen_tau_std,
    )

    plot_mean_std(
        axes[1, 2],
        edt_centers,
        real_edt_mean, real_edt_std,
        gen_edt_mean, gen_edt_std,
        xlabel="radius (voxel)",
        ylabel="pdf",
        title="EDT pore-radius distribution",
    )

    fig.suptitle(f"Six-panel comparison at target={target_value:.2f}", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / f"six_panel_phi{target_value:.2f}.png", dpi=220)
    plt.close(fig)

    # ---------- Save data ----------
    save_dict = {
        "edt_centers": edt_centers,
        "real_edt_mean": real_edt_mean,
        "real_edt_std": real_edt_std,
        "gen_edt_mean": gen_edt_mean,
        "gen_edt_std": gen_edt_std,

        "pnm_pore_radius_centers": pnm_centers["pore_radius"],
        "pnm_throat_radius_centers": pnm_centers["throat_radius"],
        "pnm_throat_length_centers": pnm_centers["throat_length"],
        "pnm_shape_factor_centers": pnm_centers["shape_factor"],

        "real_pore_radius_mean": real_pr_mean,
        "real_pore_radius_std": real_pr_std,
        "gen_pore_radius_mean": gen_pr_mean,
        "gen_pore_radius_std": gen_pr_std,

        "real_throat_radius_mean": real_tr_mean,
        "real_throat_radius_std": real_tr_std,
        "gen_throat_radius_mean": gen_tr_mean,
        "gen_throat_radius_std": gen_tr_std,

        "real_throat_length_mean": real_tl_mean,
        "real_throat_length_std": real_tl_std,
        "gen_throat_length_mean": gen_tl_mean,
        "gen_throat_length_std": gen_tl_std,

        "real_shape_factor_mean": real_sf_mean,
        "real_shape_factor_std": real_sf_std,
        "gen_shape_factor_mean": gen_sf_mean,
        "gen_shape_factor_std": gen_sf_std,

        "real_tau_mean_xyz": real_tau_mean,
        "real_tau_std_xyz": real_tau_std,
        "gen_tau_mean_xyz": gen_tau_mean,
        "gen_tau_std_xyz": gen_tau_std,
        "real_tau_all_xyz": real_tau_arr,
        "gen_tau_all_xyz": gen_tau_arr,
    }

    np.savez(out_dir / f"curves_phi{target_value:.2f}.npz", **save_dict)

    # ---------- summary ----------
    summary = {
        "target": target_value,
        "folder_name": folder_name,
        "real_dir": str(real_dir),
        "gen_dir": str(gen_dir),
        "out_dir": str(out_dir),
        "n_real_files": len(real_files),
        "n_gen_files": len(gen_files),
        "n_real_edt_ok": len(real["edt_curves"]),
        "n_gen_edt_ok": len(gen["edt_curves"]),
        "n_real_tau_ok": len(real["tau_list"]),
        "n_gen_tau_ok": len(gen["tau_list"]),
        "n_real_skipped": len(real["skipped"]),
        "n_gen_skipped": len(gen["skipped"]),
        "real_skipped": real["skipped"],
        "gen_skipped": gen["skipped"],
        "do_pnm": DO_PNM,
        "do_tortuosity": DO_TORTUOSITY,
        "pnm_saved": pnm_saved,
        "tau_saved": tau_saved,
        "figure_saved": True,
    }

    with open(out_dir / f"summary_phi{target_value:.2f}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Finished {folder_name} | figure saved: {out_dir / f'six_panel_phi{target_value:.2f}.png'}")
    return summary


# =========================================================
# 8) Main program
# =========================================================

def main():
    out_root = Path(OUT_ROOT)
    out_root.mkdir(parents=True, exist_ok=True)

    all_summaries = []

    print("=" * 84)
    print("Batch processing starts.")
    print("=" * 84)
    print(f"REAL_ROOT       : {REAL_ROOT}")
    print(f"GEN_ROOT        : {GEN_ROOT}")
    print(f"OUT_ROOT        : {OUT_ROOT}")
    print(f"DO_PNM          : {DO_PNM}")
    print(f"DO_TORTUOSITY   : {DO_TORTUOSITY}")
    print(f"MAX_REAL_FILES  : {MAX_REAL_FILES}")
    print(f"MAX_GEN_FILES   : {MAX_GEN_FILES}")
    print("=" * 84)

    for folder_name, target_value in TARGETS:
        try:
            summary = run_one_target(folder_name, target_value)
            all_summaries.append(summary)
        except Exception as e:
            err = {
                "folder_name": folder_name,
                "target": target_value,
                "error": repr(e),
            }
            all_summaries.append(err)
            print(f"[ERROR] {folder_name}: {e}")

    with open(out_root / "all_targets_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 84)
    print("All processing completed.")
    print("=" * 84)
    print(f"Output root: {out_root}")


if __name__ == "__main__":
    main()