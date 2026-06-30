import os
import csv
import json
import argparse
import numpy as np


def read_raw_uint8(path, shape):
    arr = np.fromfile(path, dtype=np.uint8)
    expected = int(np.prod(shape))
    if arr.size != expected:
        raise ValueError(f"{path}: size mismatch, got {arr.size}, expected {expected}")
    return arr.reshape(shape)


def to_binary01(vol):
    return (vol > 0).astype(np.uint8)


def build_integral_volume(vol01):
    sat = vol01.astype(np.uint32, copy=False)
    sat = sat.cumsum(axis=0, dtype=np.uint32)
    sat = sat.cumsum(axis=1, dtype=np.uint32)
    sat = sat.cumsum(axis=2, dtype=np.uint32)
    return sat


def sat_get(sat, x, y, z):
    if x < 0 or y < 0 or z < 0:
        return 0
    return int(sat[x, y, z])


def box_sum(sat, x0, y0, z0, p):
    x1 = x0 + p - 1
    y1 = y0 + p - 1
    z1 = z0 + p - 1

    s = (
        sat_get(sat, x1, y1, z1)
        - sat_get(sat, x0 - 1, y1, z1)
        - sat_get(sat, x1, y0 - 1, z1)
        - sat_get(sat, x1, y1, z0 - 1)
        + sat_get(sat, x0 - 1, y0 - 1, z1)
        + sat_get(sat, x0 - 1, y1, z0 - 1)
        + sat_get(sat, x1, y0 - 1, z0 - 1)
        - sat_get(sat, x0 - 1, y0 - 1, z0 - 1)
    )
    return s


def make_starts(L, p, stride):
    starts = list(range(0, L - p + 1, stride))
    if starts[-1] != L - p:
        starts.append(L - p)
    return starts


def phi_tag(phi):
    return f"{phi:.2f}".replace(".", "p")


def chebyshev_dist(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]), abs(a[2] - b[2]))


def compute_phi_errors(phi, target_phi):
    phi_err = float(phi - target_phi)
    phi_abs_err = float(abs(phi_err))
    phi_rel_err = float(phi_err / target_phi) if target_phi != 0 else 0.0
    phi_rel_err_pct = float(phi_rel_err * 100.0)
    return phi_err, phi_abs_err, phi_rel_err, phi_rel_err_pct


def select_patches_strict(
    candidates,
    target_phi,
    n_keep=100,
    phi_tol=0.003,
    min_sep=64,
    used_positions=None,
):
    """
    Strict filtering:
    Keep only patches with abs(phi - target_phi) <= phi_tol.
    Sort by absolute error and keep selected patch origins sufficiently separated.
    """
    if used_positions is None:
        used_positions = set()

    filtered = []
    for c in candidates:
        pos = (c["x0"], c["y0"], c["z0"])
        phi_err, phi_abs_err, phi_rel_err, phi_rel_err_pct = compute_phi_errors(
            c["phi"], target_phi
        )

        if phi_abs_err <= phi_tol:
            filtered.append({
                "x0": c["x0"],
                "y0": c["y0"],
                "z0": c["z0"],
                "phi": c["phi"],
                "phi_err": phi_err,
                "phi_abs_err": phi_abs_err,
                "phi_rel_err": phi_rel_err,
                "phi_rel_err_pct": phi_rel_err_pct,
                "pos": pos,
            })

    filtered.sort(key=lambda r: r["phi_abs_err"])

    selected = []

    def can_use(row):
        if row["pos"] in used_positions:
            return False
        for s in selected:
            if chebyshev_dist(row["pos"], s["pos"]) < min_sep:
                return False
        return True

    for row in filtered:
        if can_use(row):
            selected.append(row)
            if len(selected) >= n_keep:
                break

    return selected, len(filtered)


def save_patch(patch, out_path):
    patch.astype(np.uint8).tofile(out_path)


def summarize_target_rows(rows, target_phi, n_candidates_in_tol):
    if len(rows) == 0:
        return {
            "target_phi": float(target_phi),
            "n_candidates_in_tol": int(n_candidates_in_tol),
            "n_selected": 0,
        }

    phis = np.array([r["phi"] for r in rows], dtype=np.float64)
    abs_errs = np.array([r["phi_abs_err"] for r in rows], dtype=np.float64)
    signed_errs = np.array([r["phi_err"] for r in rows], dtype=np.float64)
    rel_errs_pct = np.array([r["phi_rel_err_pct"] for r in rows], dtype=np.float64)

    return {
        "target_phi": float(target_phi),
        "n_candidates_in_tol": int(n_candidates_in_tol),
        "n_selected": int(len(rows)),
        "phi_mean": float(phis.mean()),
        "phi_std": float(phis.std()),
        "phi_min": float(phis.min()),
        "phi_max": float(phis.max()),
        "phi_abs_err_mean": float(abs_errs.mean()),
        "phi_abs_err_std": float(abs_errs.std()),
        "phi_abs_err_min": float(abs_errs.min()),
        "phi_abs_err_max": float(abs_errs.max()),
        "phi_err_mean": float(signed_errs.mean()),
        "phi_err_std": float(signed_errs.std()),
        "phi_rel_err_pct_mean": float(rel_errs_pct.mean()),
        "phi_rel_err_pct_std": float(rel_errs_pct.std()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_path", type=str, default="S1.raw")
    ap.add_argument("--raw_shape", type=int, nargs=3, default=[800, 800, 800])
    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--stride", type=int, default=32, help="Candidate patch scanning stride.")
    ap.add_argument(
        "--targets",
        type=float,
        nargs="+",
        default=[0.11, 0.12, 0.13, 0.14, 0.15],
    )
    ap.add_argument("--n_per_target", type=int, default=100)
    ap.add_argument("--phi_tol", type=float, default=0.003, help="Absolute porosity-error threshold.")
    ap.add_argument("--min_sep", type=int, default=64, help="Minimum origin separation between patches within one target.")
    ap.add_argument(
        "--global_unique",
        action="store_true",
        help="Do not reuse the same origin across different targets.",
    )
    ap.add_argument("--out_root", type=str, default="real256_sets_from_S1_strict")
    args = ap.parse_args()

    raw_shape = tuple(args.raw_shape)
    p = args.patch

    print("[1/5] Loading raw volume ...")
    vol = read_raw_uint8(args.raw_path, raw_shape)
    vol01 = to_binary01(vol)

    X, Y, Z = vol01.shape
    if p > X or p > Y or p > Z:
        raise ValueError(f"patch={p} is larger than raw shape={vol01.shape}")

    print("[2/5] Building integral volume ...")
    sat = build_integral_volume(vol01)

    print("[3/5] Enumerating candidate patch locations ...")
    xs = make_starts(X, p, args.stride)
    ys = make_starts(Y, p, args.stride)
    zs = make_starts(Z, p, args.stride)

    patch_voxels = p * p * p
    candidates = []

    total = len(xs) * len(ys) * len(zs)
    cnt = 0
    for x0 in xs:
        for y0 in ys:
            for z0 in zs:
                s = box_sum(sat, x0, y0, z0, p)
                phi = s / patch_voxels
                candidates.append({
                    "x0": int(x0),
                    "y0": int(y0),
                    "z0": int(z0),
                    "phi": float(phi),
                })
                cnt += 1
                if cnt % 2000 == 0 or cnt == total:
                    print(f"  candidates: {cnt}/{total}")

    print(f"Total candidates = {len(candidates)}")

    print("[4/5] Selecting and saving patches ...")
    os.makedirs(args.out_root, exist_ok=True)

    used_positions = set()
    all_metadata = []
    all_summaries = []

    for target in args.targets:
        tag = phi_tag(target)
        target_dir = os.path.join(args.out_root, f"phi{tag}")
        os.makedirs(target_dir, exist_ok=True)

        selected, n_candidates_in_tol = select_patches_strict(
            candidates=candidates,
            target_phi=target,
            n_keep=args.n_per_target,
            phi_tol=args.phi_tol,
            min_sep=args.min_sep,
            used_positions=used_positions if args.global_unique else None,
        )

        if len(selected) < args.n_per_target:
            print(
                f"[WARN] target={target:.2f}: only found {len(selected)} valid patches "
                f"within phi_tol={args.phi_tol} "
                f"(requested {args.n_per_target}, in_tol={n_candidates_in_tol})."
            )

        per_target_meta = []

        for i, row in enumerate(selected, start=1):
            x0, y0, z0 = row["x0"], row["y0"], row["z0"]
            patch = vol01[x0:x0+p, y0:y0+p, z0:z0+p]

            file_id = f"{i:04d}"
            file_name = f"real256_phi{tag}_{file_id}.raw"
            out_path = os.path.join(target_dir, file_name)
            save_patch(patch, out_path)

            meta = {
                "index": i,
                "target_phi": float(target),
                "phi_tag": f"phi{tag}",
                "phi": float(row["phi"]),
                "phi_err": float(row["phi_err"]),
                "phi_abs_err": float(row["phi_abs_err"]),
                "phi_rel_err": float(row["phi_rel_err"]),
                "phi_rel_err_pct": float(row["phi_rel_err_pct"]),
                "x0": int(x0),
                "y0": int(y0),
                "z0": int(z0),
                "patch": int(p),
                "shape": [int(p), int(p), int(p)],
                "file_name": file_name,
                "file_path": os.path.abspath(out_path),
            }
            per_target_meta.append(meta)
            all_metadata.append(meta)

            if args.global_unique:
                used_positions.add((x0, y0, z0))

        # Save metadata for each target.
        csv_path = os.path.join(target_dir, f"metadata_phi{tag}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "index",
                    "target_phi",
                    "phi_tag",
                    "phi",
                    "phi_err",
                    "phi_abs_err",
                    "phi_rel_err",
                    "phi_rel_err_pct",
                    "x0",
                    "y0",
                    "z0",
                    "patch",
                    "shape",
                    "file_name",
                    "file_path",
                ],
            )
            writer.writeheader()
            for row in per_target_meta:
                row_to_write = row.copy()
                row_to_write["shape"] = "x".join(map(str, row["shape"]))
                writer.writerow(row_to_write)

        json_path = os.path.join(target_dir, f"metadata_phi{tag}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(per_target_meta, f, indent=2, ensure_ascii=False)

        summary = summarize_target_rows(per_target_meta, target, n_candidates_in_tol)
        summary["phi_tag"] = f"phi{tag}"
        summary["folder"] = os.path.abspath(target_dir)
        all_summaries.append(summary)

        summary_path = os.path.join(target_dir, f"summary_phi{tag}.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(
            f"[OK] phi={target:.2f} -> saved {len(per_target_meta)} strict patches to {target_dir} "
            f"| mean abs err={summary.get('phi_abs_err_mean', float('nan')):.6f}"
        )

    print("[5/5] Saving global metadata ...")
    with open(os.path.join(args.out_root, "all_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=2, ensure_ascii=False)

    with open(os.path.join(args.out_root, "all_metadata.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "target_phi",
                "phi_tag",
                "phi",
                "phi_err",
                "phi_abs_err",
                "phi_rel_err",
                "phi_rel_err_pct",
                "x0",
                "y0",
                "z0",
                "patch",
                "shape",
                "file_name",
                "file_path",
            ],
        )
        writer.writeheader()
        for row in all_metadata:
            row_to_write = row.copy()
            row_to_write["shape"] = "x".join(map(str, row["shape"]))
            writer.writerow(row_to_write)

    with open(os.path.join(args.out_root, "all_summaries.json"), "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)

    print(f"\nDone. All strict outputs saved under: {os.path.abspath(args.out_root)}")


if __name__ == "__main__":
    main()