"""Aggregate matching real/generated metric tables into one comparison table."""

import argparse
from pathlib import Path

import pandas as pd


IDENTIFIERS = {"group_name", "path", "sample_id", "target_tag", "status", "error"}


def aggregate(path, prefix):
    frame = pd.read_csv(path)
    if "status" in frame:
        frame = frame[frame["status"] == "ok"]
    numeric = [name for name in frame.select_dtypes(include="number").columns if name != "target_phi"]
    grouped = frame.groupby(["target_tag", "target_phi"], dropna=False)[numeric].agg(["mean", "std", "count"])
    grouped.columns = [f"{prefix}_{metric}_{stat}" for metric, stat in grouped.columns]
    return grouped.reset_index()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-csv", required=True)
    parser.add_argument("--gen-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--source-data-csv")
    args = parser.parse_args()

    real = aggregate(args.real_csv, "real")
    generated = aggregate(args.gen_csv, "generated")
    comparison = real.merge(generated, on=["target_tag", "target_phi"], how="outer")

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output, index=False)
    if args.source_data_csv:
        source_output = Path(args.source_data_csv)
        source_output.parent.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(source_output, index=False)
    print(f"Saved comparison table: {output}")


if __name__ == "__main__":
    main()
