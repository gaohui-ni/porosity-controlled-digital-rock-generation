#!/usr/bin/env python3
"""Validate and stage the Fuel manuscript's paired Test-95 cohort.

The historical manifest is read-only. Four binary branches are joined only by
``crop_id``: real, fixed-threshold Gen1, historical top-N, and actual TP-AQB
v2 Gen2. TP-AQB RAW and NPZ outputs are validated as a single artifact pair;
the NPZ ``seg`` must be voxel-identical to its RAW counterpart.
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import os
import shutil

import numpy as np


EXPECTED_GROUPS = {"0.11": 8, "0.12": 29, "0.13": 39, "0.14": 19}
DEFAULT_SHAPE = (256, 256, 256)


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_rows(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fieldnames or (list(rows[0]) if rows else [])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def validate_manifest(rows, expected_groups=EXPECTED_GROUPS):
    crop_ids = [row["crop_id"] for row in rows]
    duplicates = sorted(crop for crop, count in Counter(crop_ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate crop_id values: {duplicates}")
    groups = Counter(row["group"] for row in rows)
    if dict(groups) != expected_groups:
        raise ValueError(f"Unexpected group sizes: {dict(groups)} != {expected_groups}")


def unique_existing(candidates, label):
    existing = sorted({Path(path).resolve() for path in candidates if Path(path).is_file()})
    if len(existing) != 1:
        raise FileNotFoundError(f"{label}: expected one file, found {len(existing)}: {existing}")
    return existing[0]


def require_crop_id(path, crop_id, label):
    if crop_id not in Path(path).name:
        raise ValueError(f"{label}: {path} is not keyed by crop_id={crop_id}")


def load_binary_raw(path, shape):
    values = np.fromfile(path, dtype=np.uint8)
    expected = int(np.prod(shape))
    if values.size != expected:
        raise ValueError(f"{path}: expected {expected} uint8 voxels for shape={shape}, got {values.size}")
    if not np.all(np.isin(np.unique(values), (0, 1))):
        raise ValueError(f"{path}: expected a binary uint8 volume")
    return values.reshape(shape)


def scalar(data, key, path):
    if key not in data.files:
        raise KeyError(f"{path}: missing required field {key}")
    value = np.asarray(data[key])
    if value.size != 1:
        raise ValueError(f"{path}: {key} must be scalar, got {value.shape}")
    return value.reshape(-1)[0]


def find_tpaqb_pair(root, crop_id):
    root = Path(root)
    stems = {f"{crop_id}_tp_aqb", f"{crop_id}_prob_tp_aqb"}
    raw = unique_existing(
        (path for path in root.rglob("*.raw") if path.stem in stems),
        f"{crop_id} TP-AQB RAW",
    )
    npz = unique_existing(
        (path for path in root.rglob("*.npz") if path.stem in stems),
        f"{crop_id} TP-AQB NPZ",
    )
    if raw.parent != npz.parent or raw.stem != npz.stem:
        raise ValueError(f"{crop_id}: TP-AQB RAW/NPZ are not an adjacent artifact pair")
    return raw, npz


def validate_probability_metadata(path, row):
    with np.load(path, allow_pickle=False) as data:
        target = float(scalar(data, "target_porosity", path))
        seed = int(scalar(data, "seed", path))
    if not np.isclose(target, float(row["target_phi"]), atol=1e-12, rtol=0.0):
        raise ValueError(f"{row['crop_id']}: probability target {target} != manifest {row['target_phi']}")
    if seed != int(row["seed"]):
        raise ValueError(f"{row['crop_id']}: probability seed {seed} != manifest {row['seed']}")


def validate_tpaqb_pair(raw_path, npz_path, row, shape):
    raw = load_binary_raw(raw_path, shape)
    with np.load(npz_path, allow_pickle=False) as data:
        if "seg" not in data.files:
            raise KeyError(f"{npz_path}: missing required TP-AQB seg array")
        seg = np.asarray(data["seg"])
        target = float(scalar(data, "target_porosity", npz_path))
        achieved = float(scalar(data, "achieved_porosity", npz_path))
        seed = int(scalar(data, "seed", npz_path))
        metadata_text = str(scalar(data, "metadata_json", npz_path))
    if seg.shape != tuple(shape):
        raise ValueError(f"{npz_path}: seg shape {seg.shape} != {tuple(shape)}")
    if not np.all(np.isin(np.unique(seg), (0, 1))):
        raise ValueError(f"{npz_path}: seg is not binary")
    seg = seg.astype(np.uint8, copy=False)
    if not np.array_equal(raw, seg):
        raise ValueError(f"{row['crop_id']}: TP-AQB RAW and NPZ seg differ")
    if not np.isclose(target, float(row["target_phi"]), atol=1e-12, rtol=0.0):
        raise ValueError(f"{row['crop_id']}: TP-AQB target {target} != manifest {row['target_phi']}")
    if seed != int(row["seed"]):
        raise ValueError(f"{row['crop_id']}: TP-AQB seed {seed} != manifest {row['seed']}")
    metadata = json.loads(metadata_text)
    if "segment" not in metadata or "repair" not in metadata:
        raise ValueError(f"{npz_path}: missing TP-AQB segment/repair provenance")
    expected_count = int(round(float(row["target_phi"]) * seg.size))
    if int(seg.sum()) != expected_count:
        raise ValueError(f"{row['crop_id']}: TP-AQB pore count {int(seg.sum())} != target {expected_count}")
    if not np.isclose(achieved, float(seg.mean()), atol=1e-12, rtol=0.0):
        raise ValueError(f"{row['crop_id']}: achieved_porosity does not match seg")
    return float(seg.mean())


def materialize(source, destination, mode="copy"):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "hardlink":
        os.link(source, destination)
    elif mode == "symlink":
        destination.symlink_to(source)
    else:
        raise ValueError(f"Unknown link mode: {mode}")


def validate_staged_ids(stage_root, expected_ids):
    for branch, suffix in (("real", ".raw"), ("fixed", ".raw"), ("topn", ".raw"),
                           ("tpaqb", ".raw"), ("tpaqb_npz", ".npz")):
        paths = list((stage_root / branch).rglob(f"*{suffix}"))
        ids = [path.stem for path in paths]
        if len(ids) != len(set(ids)):
            raise ValueError(f"stage/{branch}: duplicate crop IDs")
        if set(ids) != set(expected_ids):
            raise ValueError(f"stage/{branch}: IDs do not match the manifest")


def assemble(manifest, gen_root, tpaqb_root, out_root, link_mode="copy",
             expected_shape=DEFAULT_SHAPE):
    rows = read_rows(manifest)
    validate_manifest(rows)
    gen_root = Path(gen_root).resolve()
    tpaqb_root = Path(tpaqb_root).resolve()
    out_root = Path(out_root).resolve()
    augmented = []

    branch_fields = {"real": "real_path", "fixed": "pre_aqb_path", "topn": "post_aqb_path"}
    for row in sorted(rows, key=lambda item: item["crop_id"]):
        crop_id = row["crop_id"]
        target_dir = f"phi{row['group'].replace('.', 'p')}"
        sources = {
            branch: unique_existing([gen_root / row[field]], f"{crop_id} {branch}")
            for branch, field in branch_fields.items()
        }
        for branch, source in sources.items():
            require_crop_id(source, crop_id, branch)
        prob_path = unique_existing([gen_root / row["prob_path"]], f"{crop_id} probability")
        require_crop_id(prob_path, crop_id, "probability")
        validate_probability_metadata(prob_path, row)

        arrays = {branch: load_binary_raw(path, expected_shape) for branch, path in sources.items()}
        for branch, field in (("real", "phi_real"), ("fixed", "phi_pre_aqb"),
                              ("topn", "phi_post_aqb")):
            if field in row and row[field] != "":
                if not np.isclose(float(arrays[branch].mean()), float(row[field]), atol=1e-12, rtol=0.0):
                    raise ValueError(f"{crop_id}: {branch} porosity does not match {field}")

        tpaqb_raw, tpaqb_npz = find_tpaqb_pair(tpaqb_root, crop_id)
        phi_tpaqb = validate_tpaqb_pair(tpaqb_raw, tpaqb_npz, row, expected_shape)
        phis = {branch: float(array.mean()) for branch, array in arrays.items()}
        explicit = dict(row)
        explicit.update({
            "phi_fixed": repr(phis["fixed"]),
            "phi_topn": repr(phis["topn"]),
            "phi_tpaqb": repr(phi_tpaqb),
            "tpaqb_path": str(tpaqb_raw),
            "tpaqb_npz_path": str(tpaqb_npz),
        })
        augmented.append(explicit)

        for branch, source in sources.items():
            materialize(source, out_root / "stage" / branch / target_dir / f"{crop_id}.raw", link_mode)
        materialize(tpaqb_raw, out_root / "stage" / "tpaqb" / target_dir / f"{crop_id}.raw", link_mode)
        materialize(tpaqb_npz, out_root / "stage" / "tpaqb_npz" / target_dir / f"{crop_id}.npz", link_mode)

    fields = list(rows[0]) + ["phi_fixed", "phi_topn", "phi_tpaqb", "tpaqb_path", "tpaqb_npz_path"]
    write_rows(out_root / "test95_per_crop.csv", augmented, fields)
    validate_staged_ids(out_root / "stage", [row["crop_id"] for row in rows])

    grouped = defaultdict(list)
    for row in augmented:
        grouped[row["group"]].append(row)
    summaries = []
    for group in sorted(grouped, key=float):
        items = grouped[group]
        summary = {"group": group, "n": len(items)}
        for label, field in (("real", "phi_real"), ("fixed", "phi_fixed"),
                             ("topn", "phi_topn"), ("tpaqb", "phi_tpaqb")):
            values = np.asarray([float(row[field]) for row in items], dtype=float)
            summary[f"phi_{label}_mean"] = float(values.mean())
            summary[f"phi_{label}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summaries.append(summary)
    write_rows(out_root / "test95_group_summary.csv", summaries)
    return augmented, summaries


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--gen-root", required=True)
    parser.add_argument("--tpaqb-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--shape", type=int, nargs=3, default=list(DEFAULT_SHAPE))
    parser.add_argument("--link-mode", choices=("copy", "hardlink", "symlink"), default="copy")
    return parser.parse_args()


def main():
    args = parse_args()
    rows, groups = assemble(args.manifest, args.gen_root, args.tpaqb_root,
                            args.out_root, args.link_mode, tuple(args.shape))
    print(f"Validated and staged {len(rows)} Test-95 crops in {len(groups)} groups. "
          "Use stage/tpaqb for RAW evaluators, stage/tpaqb_npz for PNM, "
          "and stage/topn only as a diagnostic.")


if __name__ == "__main__":
    main()
