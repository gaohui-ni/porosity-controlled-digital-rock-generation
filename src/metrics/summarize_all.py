import argparse
import csv
import json
from pathlib import Path


def read_csv_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def summarize_results(root):
    root = Path(root)
    summary = {
        "root": str(root),
        "csv_files": [],
        "json_files": [],
        "n_csv_rows": 0,
    }

    if not root.exists():
        return summary

    for path in sorted(root.rglob("*.csv")):
        rows = read_csv_rows(path)
        summary["csv_files"].append(
            {
                "path": str(path),
                "rows": len(rows),
                "columns": list(rows[0].keys()) if rows else [],
            }
        )
        summary["n_csv_rows"] += len(rows)

    for path in sorted(root.rglob("*.json")):
        if path.name in {"results_summary.json"}:
            continue
        summary["json_files"].append({"path": str(path)})

    return summary


def write_summary(summary, out_json, out_csv):
    out_json = Path(out_json)
    out_csv = Path(out_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    if Path(out_json).name == "results_summary.json":
        summary_alias = Path(out_json).with_name("summary.json")
        with open(summary_alias, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "rows", "columns"])
        writer.writeheader()
        for item in summary["csv_files"]:
            writer.writerow(
                {
                    "path": item["path"],
                    "rows": item["rows"],
                    "columns": ";".join(item["columns"]),
                }
            )


def main():
    parser = argparse.ArgumentParser(description="Summarize result CSV/JSON files into a compact manifest.")
    parser.add_argument("--root", default="results")
    parser.add_argument("--out_json", default=None)
    parser.add_argument("--out_csv", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    out_json = args.out_json or root / "results_summary.json"
    out_csv = args.out_csv or root / "results_summary.csv"

    summary = summarize_results(root)
    write_summary(summary, out_json, out_csv)
    print(f"Wrote {out_json}")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
