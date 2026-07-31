"""Export tidy manuscript source-data tables from evaluation outputs."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def export_s2(root, output):
    rows = []
    for path in sorted(Path(root).rglob("curve_summary_*.json")):
        with path.open("r", encoding="utf-8") as stream:
            summary = json.load(stream)
        for direction, curves in summary["s2"].items():
            for index, radius in enumerate(curves["r"]):
                rows.append(
                    {
                        "target_phi": summary["target_phi"],
                        "direction": direction,
                        "r_voxel": radius,
                        "real_mean": curves["real_mean"][index],
                        "real_std": curves["real_std"][index],
                        "generated_mean": curves["gen_mean"][index],
                        "generated_std": curves["gen_std"][index],
                    }
                )
    write_rows(output, rows)


def export_pnm(root, output):
    rows = []
    for path in sorted(Path(root).rglob("curves_phi*.npz")):
        with np.load(path) as data:
            if "target_porosity" in data.files:
                target = float(np.asarray(data["target_porosity"]).reshape(()))
            else:
                target = float(
                    path.stem.removeprefix("curves_phi").replace("p", ".")
                )
            for key in data.files:
                if key == "target_porosity":
                    continue
                array = np.asarray(data[key])
                if array.ndim == 0:
                    rows.append({"target_phi": target, "metric": key, "index": 0, "value": array.item()})
                else:
                    for index, value in enumerate(array.ravel()):
                        rows.append(
                            {
                                "target_phi": target,
                                "metric": key,
                                "index": index,
                                "value": value,
                            }
                        )
    write_rows(output, rows)


def write_rows(output, rows):
    if not rows:
        raise ValueError(f"No source data found for {output}")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved source data: {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s2-root", required=True)
    parser.add_argument("--pnm-root", required=True)
    parser.add_argument("--s2-output", required=True)
    parser.add_argument("--pnm-output", required=True)
    args = parser.parse_args()
    export_s2(args.s2_root, args.s2_output)
    export_pnm(args.pnm_root, args.pnm_output)


if __name__ == "__main__":
    main()
