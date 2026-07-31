"""Build porosity-matched 256-cubed validation patches from four real volumes."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.naming import phi_tag


def starts(length, patch, stride):
    values = list(range(0, length - patch + 1, stride))
    if values[-1] != length - patch:
        values.append(length - patch)
    return values


def parse_volume_spec(spec):
    phi_text, path_text = spec.split("=", 1)
    return float(phi_text), Path(path_text)


def select_patches(volume, target, patch, stride, count, min_sep):
    candidates = []
    for x in starts(volume.shape[0], patch, stride):
        for y in starts(volume.shape[1], patch, stride):
            for z in starts(volume.shape[2], patch, stride):
                phi = float(volume[x : x + patch, y : y + patch, z : z + patch].mean())
                candidates.append((abs(phi - target), phi, x, y, z))
    candidates.sort()

    selected = []
    for candidate in candidates:
        _, _, x, y, z = candidate
        if all(max(abs(x - sx), abs(y - sy), abs(z - sz)) >= min_sep for _, _, sx, sy, sz in selected):
            selected.append(candidate)
        if len(selected) == count:
            break
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--volume", action="append", required=True, help="POROSITY=PATH; repeat for each real volume")
    parser.add_argument("--shape", nargs=3, type=int, default=[480, 480, 480])
    parser.add_argument("--patch", type=int, default=256)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--n-per-target", type=int, default=50)
    parser.add_argument("--min-sep", type=int, default=32)
    parser.add_argument("--out-root", default="data/fontainebleau_real_sets")
    args = parser.parse_args()

    shape = tuple(args.shape)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    all_rows = []

    for spec in args.volume:
        target, raw_path = parse_volume_spec(spec)
        raw = np.fromfile(raw_path, dtype=np.uint8)
        if raw.size != int(np.prod(shape)):
            raise ValueError(f"{raw_path}: got {raw.size} voxels, expected {int(np.prod(shape))}")
        volume = (raw.reshape(shape) > 0).astype(np.uint8)
        selected = select_patches(
            volume, target, args.patch, args.stride, args.n_per_target, args.min_sep
        )
        folder = out_root / f"phi{phi_tag(target)}"
        folder.mkdir(parents=True, exist_ok=True)
        rows = []
        for index, (_, phi, x, y, z) in enumerate(selected, start=1):
            patch = volume[x : x + args.patch, y : y + args.patch, z : z + args.patch]
            filename = f"fontainebleau_real_phi{phi_tag(target)}_{index:04d}.raw"
            patch.tofile(folder / filename)
            row = {
                "target_phi": target,
                "porosity": phi,
                "abs_error": abs(phi - target),
                "x0": x,
                "y0": y,
                "z0": z,
                "source": str(raw_path),
                "file": str(folder / filename),
            }
            rows.append(row)
            all_rows.append(row)
        with (folder / "metadata.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    with (out_root / "metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(all_rows, stream, indent=2)
    print(f"Saved {len(all_rows)} real validation patches under {out_root}")


if __name__ == "__main__":
    main()
