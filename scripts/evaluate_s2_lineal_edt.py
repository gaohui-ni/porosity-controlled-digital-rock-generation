import os
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi


# =========================
# I/O
# =========================
def read_raw_uint8(path, shape):
    arr = np.fromfile(path, dtype=np.uint8)
    expected = int(np.prod(shape))
    if arr.size != expected:
        raise ValueError(f"{path}: size mismatch, got {arr.size}, expected {expected}")
    return arr.reshape(shape)


def load_binary_volume(path, shape=None, npz_key="seg", pore_value=1, threshold=0.5, invert=False):
    path = Path(path)
    suf = path.suffix.lower()

    if suf == ".raw" or suf == "":
        if shape is None:
            raise ValueError(f"RAW file requires shape: {path}")
        vol = read_raw_uint8(path, shape)

    elif suf == ".npz":
        z = np.load(path)
        if npz_key not in z.files:
            raise ValueError(f"{path}: npz key '{npz_key}' not found. available={z.files}")
        vol = z[npz_key]

    elif suf == ".npy":
        vol = np.load(path)

    else:
        raise ValueError(f"Unsupported file type: {path}")

    if vol.dtype == np.bool_:
        vol01 = vol.astype(np.uint8)
    else:
        if pore_value is not None:
            vol01 = (vol == pore_value).astype(np.uint8)
        else:
            vol01 = (vol.astype(np.float32) > threshold).astype(np.uint8)

    if invert:
        vol01 = 1 - vol01

    if vol01.ndim != 3:
        raise ValueError(f"{path}: expected 3D volume, got ndim={vol01.ndim}")

    return vol01


# =========================
# Curves
# =========================
def porosity(vol01):
    return float(vol01.mean())


def s2_direction(vol01, axis=0, r_max=128, normalize=False):
    """
    S2(r) = < I(x) I(x+r) >
    Optional normalization:
      S2n(r) = (S2(r)-phi^2)/(phi-phi^2)
    """
    v = vol01.astype(np.float32)
    phi = float(v.mean())
    r_max = int(min(r_max, v.shape[axis] - 1))

    s2 = np.zeros(r_max + 1, dtype=np.float64)
    s2[0] = float((v * v).mean())

    for r in range(1, r_max + 1):
        if axis == 0:
            a = v[:-r, :, :]
            b = v[r:, :, :]
        elif axis == 1:
            a = v[:, :-r, :]
            b = v[:, r:, :]
        else:
            a = v[:, :, :-r]
            b = v[:, :, r:]
        s2[r] = float((a * b).mean())

    if normalize:
        denom = (phi - phi * phi)
        if abs(denom) > 1e-12:
            s2 = (s2 - phi * phi) / denom

    return s2


def lineal_path_direction(vol01, axis=0, r_max=128, normalize=False):
    """
    L(r): Probability that a line segment of length r lies entirely in pore space.
    Optional normalization:
      Ln(r) = L(r)/phi
    """
    vol = vol01.astype(np.uint8)
    phi = float(vol.mean())
    r_max = int(min(r_max, vol.shape[axis] - 1))

    L = np.zeros(r_max + 1, dtype=np.float64)
    L[0] = phi

    def one_runs(arr1d):
        if arr1d.ndim != 1:
            arr1d = arr1d.ravel()
        d = np.diff(np.concatenate(([0], arr1d, [0])))
        starts = np.where(d == 1)[0]
        ends = np.where(d == -1)[0]
        return (ends - starts).astype(np.int32)

    X, Y, Z = vol.shape

    if axis == 0:
        len_line = X
        n_lines = Y * Z
        for r in range(1, r_max + 1):
            total = n_lines * (len_line - r + 1)
            ok = 0
            for y in range(Y):
                for z in range(Z):
                    runs = one_runs(vol[:, y, z])
                    ok += int(np.maximum(runs - r + 1, 0).sum())
            L[r] = ok / max(total, 1)

    elif axis == 1:
        len_line = Y
        n_lines = X * Z
        for r in range(1, r_max + 1):
            total = n_lines * (len_line - r + 1)
            ok = 0
            for x in range(X):
                for z in range(Z):
                    runs = one_runs(vol[x, :, z])
                    ok += int(np.maximum(runs - r + 1, 0).sum())
            L[r] = ok / max(total, 1)

    else:
        len_line = Z
        n_lines = X * Y
        for r in range(1, r_max + 1):
            total = n_lines * (len_line - r + 1)
            ok = 0
            for x in range(X):
                for y in range(Y):
                    runs = one_runs(vol[x, y, :])
                    ok += int(np.maximum(runs - r + 1, 0).sum())
            L[r] = ok / max(total, 1)

    if normalize and phi > 1e-12:
        L = L / phi

    return L


def edt_percentile(vol01, q=99.5):
    pore = vol01.astype(bool)
    d = ndi.distance_transform_edt(pore)
    dp = d[pore]
    if dp.size == 0:
        return 0.0
    return float(np.percentile(dp, q=q))


def pore_size_hist(vol01, bins=80, r_max=10.0):
    pore = vol01.astype(bool)
    d = ndi.distance_transform_edt(pore)
    dp = d[pore]

    if dp.size == 0:
        hist = np.zeros(bins, dtype=np.float64)
        edges = np.linspace(0.0, r_max, bins + 1)
        return hist, edges

    hist, edges = np.histogram(dp, bins=bins, range=(0.0, r_max), density=True)
    return hist.astype(np.float64), edges.astype(np.float64)


# =========================
# Helpers
# =========================
def phi_tag(phi):
    return f"{phi:.2f}".replace(".", "p")


def collect_files_for_target(root, target_tag, use_mode="raw", recursive=False):
    """
    use_mode:
      - raw
      - npz
      - both
    """
    root = Path(root)
    target_dir = root / f"phi{target_tag}"
    if not target_dir.exists():
        raise FileNotFoundError(f"Target folder not found: {target_dir}")

    if recursive:
        all_files = [p for p in target_dir.rglob("*") if p.is_file()]
    else:
        all_files = [p for p in target_dir.iterdir() if p.is_file()]

    if use_mode == "raw":
        files = [p for p in all_files if p.suffix.lower() == ".raw"]
    elif use_mode == "npz":
        files = [p for p in all_files if p.suffix.lower() == ".npz"]
    elif use_mode == "both":
        files = [p for p in all_files if p.suffix.lower() in [".raw", ".npz", ".npy"]]
    else:
        raise ValueError(f"Unknown use_mode: {use_mode}")

    files.sort()
    return files


def compute_group_curves(
    files,
    shape,
    npz_key,
    pore_value,
    threshold,
    invert,
    r_max,
    psd_bins,
    psd_r_max,
    normalize_s2,
    normalize_l,
):
    s2 = {"X": [], "Y": [], "Z": []}
    L = {"X": [], "Y": [], "Z": []}
    psd = []

    for i, f in enumerate(files, start=1):
        print(f"    [{i}/{len(files)}] {f}")
        vol01 = load_binary_volume(
            f,
            shape=shape,
            npz_key=npz_key,
            pore_value=pore_value,
            threshold=threshold,
            invert=invert,
        )

        s2["X"].append(s2_direction(vol01, axis=0, r_max=r_max, normalize=normalize_s2))
        s2["Y"].append(s2_direction(vol01, axis=1, r_max=r_max, normalize=normalize_s2))
        s2["Z"].append(s2_direction(vol01, axis=2, r_max=r_max, normalize=normalize_s2))

        L["X"].append(lineal_path_direction(vol01, axis=0, r_max=r_max, normalize=normalize_l))
        L["Y"].append(lineal_path_direction(vol01, axis=1, r_max=r_max, normalize=normalize_l))
        L["Z"].append(lineal_path_direction(vol01, axis=2, r_max=r_max, normalize=normalize_l))

        hist, edges = pore_size_hist(vol01, bins=psd_bins, r_max=psd_r_max)
        psd.append(hist)

    for k in s2:
        s2[k] = np.stack(s2[k], axis=0) if len(s2[k]) > 0 else np.zeros((0, r_max + 1))
    for k in L:
        L[k] = np.stack(L[k], axis=0) if len(L[k]) > 0 else np.zeros((0, r_max + 1))
    psd = np.stack(psd, axis=0) if len(psd) > 0 else np.zeros((0, psd_bins))

    centers = 0.5 * (edges[:-1] + edges[1:])
    return s2, L, psd, centers


def add_r_direction(curve_dict):
    """
    Add R direction: R = (X + Y + Z) / 3.
    shape: [n_samples, n_points]
    """
    curve_dict["R"] = (curve_dict["X"] + curve_dict["Y"] + curve_dict["Z"]) / 3.0
    return curve_dict


def mean_std(arr2d):
    if arr2d.shape[0] == 0:
        n_points = arr2d.shape[1]
        return np.zeros(n_points), np.zeros(n_points)
    return arr2d.mean(axis=0), arr2d.std(axis=0, ddof=0)


def plot_mean_std_curve(
    x,
    y_real_mean,
    y_real_std,
    y_gen_mean,
    y_gen_std,
    title,
    xlabel,
    ylabel,
    out_path,
    real_label="Real",
    gen_label="Gen",
):
    plt.figure(figsize=(6.5, 4.5))

    plt.plot(x, y_real_mean, label=real_label)
    plt.fill_between(x, y_real_mean - y_real_std, y_real_mean + y_real_std, alpha=0.25)

    plt.plot(x, y_gen_mean, label=gen_label)
    plt.fill_between(x, y_gen_mean - y_gen_std, y_gen_mean + y_gen_std, alpha=0.25)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_curve_summary(out_path, payload):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_root", required=True, help="Root directory of real samples, e.g. real256_sets_from_S1_scheme1.")
    ap.add_argument("--gen_root", required=True, help="Root directory of generated samples, e.g. generated_phi_sets.")
    ap.add_argument("--targets", type=float, nargs="+", default=[0.11, 0.12, 0.13, 0.14, 0.15])
    ap.add_argument("--out_root", default="curve_band_out")

    ap.add_argument("--shape", type=int, nargs=3, default=[256, 256, 256], help="RAW volume shape.")
    ap.add_argument("--real_use", choices=["raw", "npz", "both"], default="raw")
    ap.add_argument("--gen_use", choices=["raw", "npz", "both"], default="npz")
    ap.add_argument("--real_npz_key", default="seg")
    ap.add_argument("--gen_npz_key", default="seg")

    ap.add_argument("--pore_value", type=float, default=1)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--invert", action="store_true")
    ap.add_argument("--recursive", action="store_true")

    ap.add_argument("--r_max", type=int, default=128)
    ap.add_argument("--psd_bins", type=int, default=80)
    ap.add_argument("--psd_r_max", type=float, default=-1.0, help="If <=0, choose automatically from the maximum p99.5 EDT value over real+generated samples for the target.")
    ap.add_argument("--normalize_s2", action="store_true", help="Recommended: normalize S2 as (S2 - phi^2) / (phi - phi^2).")
    ap.add_argument("--normalize_l", action="store_true", help="Recommended: normalize L as L / phi.")

    args = ap.parse_args()

    shape = tuple(args.shape)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for target in args.targets:
        tag = phi_tag(target)
        print(f"\n===== target = {target:.2f} (phi{tag}) =====")

        target_out = out_root / f"phi{tag}"
        target_out.mkdir(parents=True, exist_ok=True)

        real_files = collect_files_for_target(args.real_root, tag, use_mode=args.real_use, recursive=args.recursive)
        gen_files = collect_files_for_target(args.gen_root, tag, use_mode=args.gen_use, recursive=args.recursive)

        print(f"  real files: {len(real_files)}")
        print(f"  gen  files: {len(gen_files)}")

        if len(real_files) == 0:
            raise ValueError(f"No real files found for phi{tag}")
        if len(gen_files) == 0:
            raise ValueError(f"No gen files found for phi{tag}")

        # Automatically determine a common PSD radius upper bound.
        if args.psd_r_max <= 0:
            print("  estimating psd_r_max automatically ...")
            all_files = real_files + gen_files
            qvals = []
            for i, f in enumerate(all_files, start=1):
                print(f"    [auto-rmax {i}/{len(all_files)}] {f}")
                npz_key = args.real_npz_key if f in real_files else args.gen_npz_key
                vol01 = load_binary_volume(
                    f,
                    shape=shape,
                    npz_key=npz_key,
                    pore_value=args.pore_value,
                    threshold=args.threshold,
                    invert=args.invert,
                )
                qvals.append(edt_percentile(vol01, q=99.5))
            psd_r_max = max(qvals) if len(qvals) > 0 else 10.0
        else:
            psd_r_max = args.psd_r_max

        print(f"  psd_r_max = {psd_r_max:.4f}")

        # Real-sample curves.
        print("  computing REAL curves ...")
        s2_real, L_real, psd_real, centers = compute_group_curves(
            files=real_files,
            shape=shape,
            npz_key=args.real_npz_key,
            pore_value=args.pore_value,
            threshold=args.threshold,
            invert=args.invert,
            r_max=args.r_max,
            psd_bins=args.psd_bins,
            psd_r_max=psd_r_max,
            normalize_s2=args.normalize_s2,
            normalize_l=args.normalize_l,
        )
        s2_real = add_r_direction(s2_real)
        L_real = add_r_direction(L_real)

        # Generated-sample curves.
        print("  computing GEN curves ...")
        s2_gen, L_gen, psd_gen, _ = compute_group_curves(
            files=gen_files,
            shape=shape,
            npz_key=args.gen_npz_key,
            pore_value=args.pore_value,
            threshold=args.threshold,
            invert=args.invert,
            r_max=args.r_max,
            psd_bins=args.psd_bins,
            psd_r_max=psd_r_max,
            normalize_s2=args.normalize_s2,
            normalize_l=args.normalize_l,
        )
        s2_gen = add_r_direction(s2_gen)
        L_gen = add_r_direction(L_gen)

        r = np.arange(args.r_max + 1)

        # S2
        for name in ["X", "Y", "Z", "R"]:
            m_real, s_real = mean_std(s2_real[name])
            m_gen, s_gen = mean_std(s2_gen[name])

            ylabel = "Normalized S2(r)" if args.normalize_s2 else "S2(r)"
            title_name = "R (mean of X,Y,Z)" if name == "R" else name

            plot_mean_std_curve(
                x=r,
                y_real_mean=m_real,
                y_real_std=s_real,
                y_gen_mean=m_gen,
                y_gen_std=s_gen,
                title=f"S2(r) - {title_name} - phi={target:.2f}",
                xlabel="r (voxel)",
                ylabel=ylabel,
                out_path=target_out / f"s2_{name}_phi{tag}.png",
            )

        # L
        for name in ["X", "Y", "Z", "R"]:
            m_real, s_real = mean_std(L_real[name])
            m_gen, s_gen = mean_std(L_gen[name])

            ylabel = "Normalized L(r)" if args.normalize_l else "L(r)"
            title_name = "R (mean of X,Y,Z)" if name == "R" else name

            plot_mean_std_curve(
                x=r,
                y_real_mean=m_real,
                y_real_std=s_real,
                y_gen_mean=m_gen,
                y_gen_std=s_gen,
                title=f"Lineal-path L(r) - {title_name} - phi={target:.2f}",
                xlabel="r (voxel)",
                ylabel=ylabel,
                out_path=target_out / f"lineal_{name}_phi{tag}.png",
            )

        # PSD
        m_real_psd, s_real_psd = mean_std(psd_real)
        m_gen_psd, s_gen_psd = mean_std(psd_gen)

        plot_mean_std_curve(
            x=centers,
            y_real_mean=m_real_psd,
            y_real_std=s_real_psd,
            y_gen_mean=m_gen_psd,
            y_gen_std=s_gen_psd,
            title=f"Pore size distribution - phi={target:.2f}",
            xlabel="EDT radius (voxel)",
            ylabel="PDF",
            out_path=target_out / f"pore_size_dist_phi{tag}.png",
        )

        # Save curve summary.
        summary = {
            "target_phi": float(target),
            "phi_tag": f"phi{tag}",
            "n_real": int(len(real_files)),
            "n_gen": int(len(gen_files)),
            "r_max": int(args.r_max),
            "psd_bins": int(args.psd_bins),
            "psd_r_max": float(psd_r_max),
            "normalize_s2": bool(args.normalize_s2),
            "normalize_l": bool(args.normalize_l),
            "s2": {},
            "lineal": {},
            "pore_size_distribution": {
                "centers": centers.tolist(),
                "real_mean": m_real_psd.tolist(),
                "real_std": s_real_psd.tolist(),
                "gen_mean": m_gen_psd.tolist(),
                "gen_std": s_gen_psd.tolist(),
            },
        }

        for name in ["X", "Y", "Z", "R"]:
            m_real_s2, s_real_s2 = mean_std(s2_real[name])
            m_gen_s2, s_gen_s2 = mean_std(s2_gen[name])

            summary["s2"][name] = {
                "r": r.tolist(),
                "real_mean": m_real_s2.tolist(),
                "real_std": s_real_s2.tolist(),
                "gen_mean": m_gen_s2.tolist(),
                "gen_std": s_gen_s2.tolist(),
            }

            m_real_l, s_real_l = mean_std(L_real[name])
            m_gen_l, s_gen_l = mean_std(L_gen[name])

            summary["lineal"][name] = {
                "r": r.tolist(),
                "real_mean": m_real_l.tolist(),
                "real_std": s_real_l.tolist(),
                "gen_mean": m_gen_l.tolist(),
                "gen_std": s_gen_l.tolist(),
            }

        save_curve_summary(target_out / f"curve_summary_phi{tag}.json", summary)

        print(f"  [OK] saved plots to {target_out}")

    print(f"\nDone. All outputs saved under: {out_root.resolve()}")


if __name__ == "__main__":
    main()