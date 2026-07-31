"""Reviewer-facing lightweight figure-output helper.

This script provides one command that points reviewers from the official
pipeline to figure-oriented outputs and validates the demo plotting path. It
does not claim to regenerate every manuscript figure from precomputed results;
manuscript-scale figures require the full evaluation outputs documented in
``docs/figure_reproduction.md``.
"""

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(command, dry_run=False):
    print(" ".join(command))
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a lightweight figure-output check from the configured workflow."
    )
    parser.add_argument("--config", default="configs/main.yaml")
    parser.add_argument("--results-root", default="results")
    parser.add_argument(
        "--skip-pipeline",
        action="store_true",
        help="Use existing outputs and only run plotting/summary steps.",
    )
    parser.add_argument(
        "--run-full-pipeline",
        action="store_true",
        help="Explicitly run the manuscript-scale full workflow before plotting.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the figure reproduction commands without executing them.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results_root = Path(args.results_root)

    if args.skip_pipeline and args.run_full_pipeline:
        raise ValueError("--skip-pipeline and --run-full-pipeline cannot be used together.")

    if args.run_full_pipeline:
        run(
            [
                sys.executable,
                "run_pipeline.py",
                "--mode",
                "full",
                "--config",
                args.config,
            ],
            dry_run=args.dry_run,
        )
    elif not args.skip_pipeline:
        run(
            [
                sys.executable,
                "run_pipeline.py",
                "--mode",
                "demo",
                "--config",
                args.config,
            ],
            dry_run=args.dry_run,
        )

    run(
        [
            sys.executable,
            "src/metrics/summarize_all.py",
            "--root",
            str(results_root),
        ],
        dry_run=args.dry_run,
    )
    run(
        [
            sys.executable,
            "scripts/plot_all.py",
            "--summary",
            str(results_root / "figure_reproduction_manifest.json"),
            "--figure",
            str(results_root / "figures" / "demo_synthetic_s2.png"),
        ],
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print("\nDry run only: no figures or summary files were written.")
    else:
        print(f"\nLightweight figure-oriented outputs are under: {results_root}")


if __name__ == "__main__":
    main()
