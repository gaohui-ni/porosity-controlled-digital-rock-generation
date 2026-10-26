#!/usr/bin/env python3
"""Apply the frozen TP-AQB v2 recipe to generated probability volumes.

The command keeps the repository's standard generated-sample layout:

    <output-root>/phi<tag>/<sample>.npz

Every output contains a ``seg`` array compatible with the existing evaluation
scripts.  TP-AQB reads only the generated probability field and requested
porosity; no real comparison volume is used during segmentation.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sampling.tp_aqb import surface_count, swap_repair, tp_aqb_segment
from src.utils.naming import phi_dir


def load_recipe(path):
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    backbone = dict(payload.get("backbone", {}))
    swap = dict(payload.get("swap", {}))
    repair = dict(backbone)
    repair.update(swap)
    return backbone, repair, payload


def scalar(data, key):
    value = np.asarray(data[key])
    if value.size != 1:
        raise ValueError(f"{key} must be scalar, got shape {value.shape}")
    return float(value.reshape(-1)[0])


def process_probability(path, output_root, backbone, repair, save_probability=False):
    with np.load(path) as data:
        if "prob" not in data or "target_porosity" not in data:
            raise KeyError(f"{path}: expected 'prob' and 'target_porosity'")
        prob = np.ascontiguousarray(data["prob"], dtype=np.float32)
        target = scalar(data, "target_porosity")
        seed = int(scalar(data, "seed")) if "seed" in data else -1

    if prob.ndim != 3:
        raise ValueError(f"{path}: probability volume must be 3D, got {prob.shape}")

    info_segment = {}
    info_repair = {}
    off = prob >= 0.5
    sv_off = float(surface_count(off)) / float(prob.size)
    seg = tp_aqb_segment(prob, target, cfg=backbone, info=info_segment)
    seg = swap_repair(prob, seg, sv_off, cfg=repair, info=info_repair)
    seg = np.ascontiguousarray(seg, dtype=np.uint8)

    expected = int(round(target * prob.size))
    actual = int(seg.sum())
    if actual != expected:
        raise RuntimeError(
            f"{path}: TP-AQB pore count {actual} != requested {expected}"
        )

    target_dir = Path(output_root) / phi_dir(target)
    target_dir.mkdir(parents=True, exist_ok=True)
    output = target_dir / f"{Path(path).stem}_tp_aqb.npz"
    metadata = {
        "source": str(path),
        "target_porosity": target,
        "achieved_porosity": float(seg.mean()),
        "seed": seed,
        "sv_off": sv_off,
        "segment": info_segment,
        "repair": info_repair,
    }
    arrays = {
        "seg": seg,
        "target_porosity": np.float64(target),
        "achieved_porosity": np.float64(seg.mean()),
        "seed": np.int64(seed),
        "metadata_json": np.asarray(json.dumps(metadata, ensure_ascii=False)),
    }
    if save_probability:
        arrays["prob"] = prob
    np.savez_compressed(output, **arrays)
    seg.tofile(output.with_suffix(".raw"))
    return output, metadata


def main():
    parser = argparse.ArgumentParser(
        description="Topology-preserving adaptive quantile binarization v2."
    )
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config", default="configs/tp_aqb_v2.yaml")
    parser.add_argument("--pattern", default="*.npz")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--save-probability", action="store_true")
    args = parser.parse_args()

    backbone, repair, _ = load_recipe(args.config)
    paths = sorted(Path(args.input_root).rglob(args.pattern))
    if args.limit > 0:
        paths = paths[: args.limit]
    if not paths:
        raise FileNotFoundError(f"No {args.pattern} files under {args.input_root}")

    failures = []
    converted = 0
    for index, path in enumerate(paths, 1):
        try:
            output, metadata = process_probability(
                path,
                args.output_root,
                backbone,
                repair,
                save_probability=args.save_probability,
            )
            print(
                f"[{index}/{len(paths)}] {path.name} -> {output} "
                f"phi={metadata['achieved_porosity']:.8f}"
            )
            converted += 1
        except (KeyError, ValueError) as exc:
            # A condition-sweep folder may contain unrelated NPZ summaries.
            failures.append(f"{path}: {exc}")

    if failures:
        print("Skipped incompatible files:")
        for failure in failures:
            print(f"  {failure}")
    print(
        f"Converted {converted} compatible probability volume(s); "
        f"skipped {len(failures)}."
    )
    if converted == 0:
        raise SystemExit("No compatible probability volumes were converted")


if __name__ == "__main__":
    main()
