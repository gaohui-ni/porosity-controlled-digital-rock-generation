#!/usr/bin/env python3
"""Assemble and validate the Fuel manuscript's paired Test-95 cohort.

This script does not reimplement morphology, PNM, or permeability metrics.  It
joins the four binary branches by ``crop_id`` and stages canonical directory
trees for the existing evaluation scripts:

* ``real``: held-out real crop;
* ``fixed``: FiLM probability thresholded at 0.5 (manuscript Gen1);
* ``topn``: historical exact top-N diagnostic (legacy ``phi_post_aqb``);
* ``tpaqb``: actual TP-AQB v2 output (manuscript Gen2).

The historical manifest is read-only.  A new augmented manifest with explicit
``phi_tpaqb`` and ``tpaqb_path`` fields is written under ``--out-root``.
"""

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
import os
import shutil

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_GROUPS = {"0.11": 8, "0.12": 29, "0.13": 39, "0.14": 19}


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_rows(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
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


def raw_porosity(path):
    values = np.memmap(path, dtype=np.uint8, mode="r")
    if values.size == 0:
        raise ValueError(f"Empty raw file: {path}")
    unique = np.unique(values)
    if not np.all(np.isin(unique, (0, 1))):
        raise ValueError(f"Expected binary uint8 raw file: {path}; values={unique[:8]}")
    return float(values.mean())


def unique_existing(candidates, label):
    existing = sorted({Path(path).resolve() for path in candidates if Path(path).is_file()})
    if len(existing) != 1:
        raise FileNotFoundError(f"{label}: expected one file, found {len(existing)}: {existing}")
    return existing[0]


def find_tpaqb(root, crop_id):
    root = Path(root)
    names = {f"{crop_id}_tp_aqb.raw", f"{crop_id}_prob_tp_aqb.raw"}
    return unique_existing(
        (path for path in root.rglob("*.raw") if path.name in names),
        f"{crop_id} TP-AQB",
    )


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


def assemble(manifest, gen_root, tpaqb_root, out_root, link_mode="copy"):
    rows = read_rows(manifest)
    validate_manifest(rows)
    gen_root = Path(gen_root).resolve()
    tpaqb_root = Path(tpaqb_root).resolve()
    out_root = Path(out_root).resolve()
    augmented = []

    branch_fields = {
        "real": "real_path",
        "fixed": "pre_aqb_path",
        "topn": "post_aqb_path",
    }
    for row in sorted(rows, key=lambda item: item["crop_id"]):
        crop_id = row["crop_id"]
        target_dir = f"phi{row['group'].replace('.', 'p')}"
        sources = {
            branch: unique_existing(
                [gen_root / row[field], gen_root / Path(row[field]).name],
                f"{crop_id} {branch}",
            )
            for branch, field in branch_fields.items()
        }
        sources["tpaqb"] = find_tpaqb(tpaqb_root, crop_id)

        phis = {branch: raw_porosity(path) for branch, path in sources.items()}
        explicit = dict(row)
        explicit.update(
            {
                "phi_fixed": repr(phis["fixed"]),
                "phi_topn": repr(phis["topn"]),
                "phi_tpaqb": repr(phis["tpaqb"]),
                "tpaqb_path": str(sources["tpaqb"]),
            }
        )
        augmented.append(explicit)

        for branch, source in sources.items():
            destination = out_root / "stage" / branch / target_dir / f"{crop_id}.raw"
            materialize(source, destination, mode=link_mode)

    fields = list(rows[0]) + ["phi_fixed", "phi_topn", "phi_tpaqb", "tpaqb_path"]
    write_rows(out_root / "test95_per_crop.csv", augmented, fields)

    grouped = defaultdict(list)
    for row in augmented:
        grouped[row["group"]].append(row)
    summaries = []
    for group in sorted(grouped, key=float):
        items = grouped[group]
        summary = {"group": group, "n": len(items)}
        for label, field in (
            ("real", "phi_real"),
            ("fixed", "phi_fixed"),
            ("topn", "phi_topn"),
            ("tpaqb", "phi_tpaqb"),
        ):
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
    parser.add_argument(
        "--link-mode", choices=("copy", "hardlink", "symlink"), default="copy"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rows, groups = assemble(
        args.manifest,
        args.gen_root,
        args.tpaqb_root,
        args.out_root,
        link_mode=args.link_mode,
    )
    print(
        f"Validated and staged {len(rows)} Test-95 crops in {len(groups)} groups. "
        "Use stage/tpaqb for manuscript Gen2 and stage/topn only as a diagnostic."
    )


if __name__ == "__main__":
    main()
