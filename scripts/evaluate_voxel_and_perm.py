import os
import re
import csv
import json
import argparse
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

import openpnm as op
import porespy as ps


DARCY_IN_M2 = 9.869233e-13


# =========================
# I/O
# =========================
def load_volume(path, shape=None, npz_key="seg", dtype="uint8", pore_value=1, threshold=0.5, invert=False):
    path = Path(path)
    suf = path.suffix.lower()

    if suf == ".raw" or suf == "":
        if shape is None:
            raise ValueError(f"RAW file requires --shape: {path}")
        arr = np.fromfile(path, dtype=np.dtype(dtype))
        expected = int(np.prod(shape))
        if arr.size != expected:
            raise ValueError(f"{path}: size mismatch, got {arr.size}, expected {expected}")
        arr = arr.reshape(shape)

    elif suf == ".npy":
        arr = np.load(path)

    elif suf == ".npz":
        z = np.load(path)
        if npz_key not in z.files:
            raise ValueError(f"{path}: npz key '{npz_key}' not found, available: {z.files}")
        arr = z[npz_key]

    else:
        raise ValueError(f"Unsupported file type: {path}")

    if arr.dtype == np.bool_:
        vol01 = arr.astype(np.uint8)
    else:
        if pore_value is not None:
            vol01 = (arr == pore_value).astype(np.uint8)
        else:
            vol01 = (arr.astype(np.float32) > threshold).astype(np.uint8)

    if invert:
        vol01 = 1 - vol01

    if vol01.ndim != 3:
        raise ValueError(f"{path}: volume ndim must be 3, got {vol01.ndim}")

    return vol01


# =========================
# Basic voxel metrics
# =========================
def porosity(vol01):
    return float(vol01.mean())


def total_pore_vox(vol01):
    return int(vol01.sum())


def surface_area_per_volume(vol01):
    v = vol01.astype(np.uint8)
    X, Y, Z = v.shape
    interfaces = 0
    interfaces += np.sum(v[:-1, :, :] != v[1:, :, :])
    interfaces += np.sum(v[:, :-1, :] != v[:, 1:, :])
    interfaces += np.sum(v[:, :, :-1] != v[:, :, 1:])
    V = X * Y * Z
    return float(interfaces / V)


def cluster_stats(vol01, connectivity=3):
    pore = vol01.astype(bool)
    structure = ndi.generate_binary_structure(3, connectivity)  # 26-connectivity
    lab, n = ndi.label(pore, structure=structure)
    if n == 0:
        return {
            "n_clusters": 0,
            "largest_size": 0,
            "largest_frac": 0.0,
            "total_pore_vox": 0,
        }, lab

    sizes = np.bincount(lab.ravel())[1:]  # exclude label 0
    largest = int(sizes.max())
    total = int(pore.sum())

    return {
        "n_clusters": int(n),
        "largest_size": largest,
        "largest_frac": float(largest / max(total, 1)),
        "total_pore_vox": total,
    }, lab


def percolation_along_axis(lab, axis=0):
    if lab.max() == 0:
        return False, 0.0

    if axis == 0:
        a = lab[0, :, :].ravel()
        b = lab[-1, :, :].ravel()
    elif axis == 1:
        a = lab[:, 0, :].ravel()
        b = lab[:, -1, :].ravel()
    else:
        a = lab[:, :, 0].ravel()
        b = lab[:, :, -1].ravel()

    A = set(int(x) for x in a if x != 0)
    B = set(int(x) for x in b if x != 0)
    inter = A.intersection(B)

    if not inter:
        return False, 0.0

    mask = np.isin(lab, list(inter))
    pore_total = np.count_nonzero(lab)
    return True, float(mask.sum() / max(pore_total, 1))


def edt_stats(vol01):
    pore = vol01.astype(bool)
    d = ndi.distance_transform_edt(pore)
    dp = d[pore]
    if dp.size == 0:
        return {
            "edt_mean": 0.0,
            "edt_p90": 0.0,
            "edt_p99": 0.0,
        }

    return {
        "edt_mean": float(dp.mean()),
        "edt_p90": float(np.percentile(dp, 90)),
        "edt_p99": float(np.percentile(dp, 99)),
    }


# =========================
# Permeability (SNOW + OpenPNM)
# =========================
def boundary_width_for_axis(axis):
    if axis == "x":
        return [[3, 3], 0, 0]
    if axis == "y":
        return [0, [3, 3], 0]
    if axis == "z":
        return [0, 0, [3, 3]]
    raise ValueError(axis)


def axis_labels(axis):
    if axis == "x":
        return "xmin", "xmax", 0, (1, 2)
    if axis == "y":
        return "ymin", "ymax", 1, (0, 2)
    if axis == "z":
        return "zmin", "zmax", 2, (0, 1)
    raise ValueError(axis)


def build_network_for_axis(vol01, voxel_size, axis, accuracy="standard"):
    snow = ps.networks.snow2(
        phases=vol01.astype(np.uint8),
        voxel_size=voxel_size,
        boundary_width=boundary_width_for_axis(axis),
        accuracy=accuracy,
    )

    pn = op.io.network_from_porespy(snow.network)

    health = op.utils.check_network_health(pn)
    bad = health.get("disconnected_pores", [])
    if len(bad) > 0:
        op.topotools.trim(network=pn, pores=bad)

    pn["pore.diameter"] = pn["pore.equivalent_diameter"]
    pn["throat.diameter"] = pn["throat.inscribed_diameter"]
    pn["throat.spacing"] = pn["throat.total_length"]

    pn.add_model(
        propname="throat.hydraulic_size_factors",
        model=op.models.geometry.hydraulic_size_factors.pyramids_and_cuboids,
    )
    pn.add_model(
        propname="throat.diffusive_size_factors",
        model=op.models.geometry.diffusive_size_factors.pyramids_and_cuboids,
    )
    pn.regenerate_models()
    return pn


def directional_perm(vol01, voxel_size, axis, viscosity=1.0, accuracy="standard"):
    pn = build_network_for_axis(vol01, voxel_size, axis, accuracy=accuracy)

    phase = op.phase.Phase(network=pn)
    phase["pore.viscosity"] = float(viscosity)
    phase.add_model_collection(op.models.collections.physics.basic)
    phase.regenerate_models()

    inlet_label, outlet_label, ax_id, area_axes = axis_labels(axis)
    inlet = pn.pores(inlet_label)
    outlet = pn.pores(outlet_label)

    if len(inlet) == 0 or len(outlet) == 0:
        raise RuntimeError(f"{axis} direction has no valid inlet/outlet pores")

    flow = op.algorithms.StokesFlow(network=pn, phase=phase)
    flow.set_value_BC(pores=inlet, values=1.0)
    flow.set_value_BC(pores=outlet, values=0.0)
    flow.run()

    Q = float(np.abs(flow.rate(pores=inlet, mode="group")[0]))
    shape = vol01.shape
    L = float((shape[ax_id] + 6) * voxel_size)
    A = float(shape[area_axes[0]] * shape[area_axes[1]] * voxel_size * voxel_size)

    K = Q * viscosity * L / A
    return K


def compute_perm_all(vol01, voxel_size, viscosity=1.0, accuracy="standard"):
    Kx = directional_perm(vol01, voxel_size, "x", viscosity, accuracy)
    Ky = directional_perm(vol01, voxel_size, "y", viscosity, accuracy)
    Kz = directional_perm(vol01, voxel_size, "z", viscosity, accuracy)
    Kgeom = float(np.exp(np.mean(np.log([Kx, Ky, Kz]))))

    return {
        "Kx_m2": Kx,
        "Ky_m2": Ky,
        "Kz_m2": Kz,
        "Kgeom_m2": Kgeom,
        "Kx_D": Kx / DARCY_IN_M2,
        "Ky_D": Ky / DARCY_IN_M2,
        "Kz_D": Kz / DARCY_IN_M2,
        "Kgeom_D": Kgeom / DARCY_IN_M2,
    }


# =========================
# Helpers
# =========================
def infer_target_info(path_str):
    """
    Infer target information from the path, for example:
    phi0p11 / tp0p11 / phi0.11 / tp0.11
    """
    s = path_str.replace("\\", "/")

    patterns = [
        r"(phi|tp)(\d+p\d+)",
        r"(phi|tp)(\d+\.\d+)",
    ]
    for pat in patterns:
        m = re.search(pat, s)
        if m:
            tag = m.group(2)
            val = float(tag.replace("p", "."))
            return val, tag

    return None, None


def find_input_files(root, exts=(".raw", ".npz"), recursive=True):
    root = Path(root)
    files = []
    if recursive:
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                files.append(p)
    else:
        for p in root.iterdir():
            if p.is_file() and p.suffix.lower() in exts:
                files.append(p)
    files.sort()
    return files


def process_one_file(
    path,
    shape,
    npz_key,
    dtype,
    pore_value,
    threshold,
    invert,
    voxel_size,
    perm_accuracy,
):
    vol01 = load_volume(
        path=path,
        shape=shape,
        npz_key=npz_key,
        dtype=dtype,
        pore_value=pore_value,
        threshold=threshold,
        invert=invert,
    )

    phi = porosity(vol01)
    cs, lab = cluster_stats(vol01, connectivity=3)

    px_is, px_frac = percolation_along_axis(lab, axis=0)
    py_is, py_frac = percolation_along_axis(lab, axis=1)
    pz_is, pz_frac = percolation_along_axis(lab, axis=2)

    sv = surface_area_per_volume(vol01)
    edt = edt_stats(vol01)
    perm = compute_perm_all(vol01, voxel_size=voxel_size, viscosity=1.0, accuracy=perm_accuracy)

    target_phi, target_tag = infer_target_info(str(path))

    row = {
        "path": str(path),
        "sample_id": Path(path).stem,
        "target_phi": "" if target_phi is None else target_phi,
        "target_tag": "" if target_tag is None else target_tag,
        "porosity": phi,
        "total_pore_vox": cs["total_pore_vox"],
        "n_clusters": cs["n_clusters"],
        "largest_size": cs["largest_size"],
        "largest_frac": cs["largest_frac"],
        "perc_x_is": int(px_is),
        "perc_y_is": int(py_is),
        "perc_z_is": int(pz_is),
        "perc_frac_x": px_frac,
        "perc_frac_y": py_frac,
        "perc_frac_z": pz_frac,
        "perc_frac_mean": float((px_frac + py_frac + pz_frac) / 3.0),
        "S_over_V": sv,
        "edt_mean": edt["edt_mean"],
        "edt_p90": edt["edt_p90"],
        "edt_p99": edt["edt_p99"],
        **perm,
    }
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_root", type=str, required=True, help="Input root directory; recursively search raw/npz files.")
    ap.add_argument("--output_csv", type=str, default="voxel_metrics.csv")
    ap.add_argument("--group_name", type=str, default="real", help="Label this sample batch, e.g. real / gen_raw / gen_seg.")
    ap.add_argument("--shape", type=int, nargs=3, default=[256, 256, 256], help="RAW volume shape.")
    ap.add_argument("--dtype", type=str, default="uint8")
    ap.add_argument("--npz_key", type=str, default="seg", help="If input is npz, choose which key to read.")
    ap.add_argument("--pore_value", type=float, default=1, help="Treat values equal to this value as pore voxels.")
    ap.add_argument("--threshold", type=float, default=0.5, help="Use this threshold when pore_value=None.")
    ap.add_argument("--invert", action="store_true")
    ap.add_argument("--voxel_size", type=float, default=3.0e-6, help="Voxel size in meters.")
    ap.add_argument("--perm_accuracy", choices=["standard", "high"], default="standard")
    ap.add_argument("--recursive", action="store_true", help="Recursively search subdirectories.")
    ap.add_argument("--limit", type=int, default=None, help="Process only the first N files for debugging.")
    args = ap.parse_args()

    shape = tuple(args.shape)
    files = find_input_files(
        root=args.input_root,
        exts=(".raw", ".npz"),
        recursive=args.recursive,
    )

    if args.limit is not None:
        files = files[: args.limit]

    if len(files) == 0:
        raise ValueError(f"No .raw or .npz files found under {args.input_root}")

    fieldnames = [
        "group_name",
        "path",
        "sample_id",
        "target_phi",
        "target_tag",
        "porosity",
        "total_pore_vox",
        "n_clusters",
        "largest_size",
        "largest_frac",
        "perc_x_is",
        "perc_y_is",
        "perc_z_is",
        "perc_frac_x",
        "perc_frac_y",
        "perc_frac_z",
        "perc_frac_mean",
        "S_over_V",
        "edt_mean",
        "edt_p90",
        "edt_p99",
        "Kx_m2",
        "Ky_m2",
        "Kz_m2",
        "Kgeom_m2",
        "Kx_D",
        "Ky_D",
        "Kz_D",
        "Kgeom_D",
        "status",
        "error",
    ]

    rows = []
    print(f"Found {len(files)} files")

    for i, path in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] {path}")
        try:
            row = process_one_file(
                path=path,
                shape=shape,
                npz_key=args.npz_key,
                dtype=args.dtype,
                pore_value=args.pore_value,
                threshold=args.threshold,
                invert=args.invert,
                voxel_size=args.voxel_size,
                perm_accuracy=args.perm_accuracy,
            )
            row["group_name"] = args.group_name
            row["status"] = "ok"
            row["error"] = ""
        except Exception as e:
            row = {
                "group_name": args.group_name,
                "path": str(path),
                "sample_id": Path(path).stem,
                "target_phi": "",
                "target_tag": "",
                "porosity": "",
                "total_pore_vox": "",
                "n_clusters": "",
                "largest_size": "",
                "largest_frac": "",
                "perc_x_is": "",
                "perc_y_is": "",
                "perc_z_is": "",
                "perc_frac_x": "",
                "perc_frac_y": "",
                "perc_frac_z": "",
                "perc_frac_mean": "",
                "S_over_V": "",
                "edt_mean": "",
                "edt_p90": "",
                "edt_p99": "",
                "Kx_m2": "",
                "Ky_m2": "",
                "Kz_m2": "",
                "Kgeom_m2": "",
                "Kx_D": "",
                "Ky_D": "",
                "Kz_D": "",
                "Kgeom_D": "",
                "status": "error",
                "error": str(e),
            }
            print(f"  [ERROR] {e}")

        rows.append(row)

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"\nSaved: {args.output_csv}")

    ok_count = sum(1 for r in rows if r["status"] == "ok")
    err_count = len(rows) - ok_count
    print(f"Done. ok={ok_count}, error={err_count}")


if __name__ == "__main__":
    main()