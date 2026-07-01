import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Run the configured evaluation commands and summarize outputs.")
    parser.add_argument("--config", default="configs/main.yaml")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run(command, dry_run=False):
    print(" ".join(command))
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def main():
    args = parse_args()
    run(
        [
            sys.executable,
            "run_pipeline.py",
            "--mode",
            "full",
            "--config",
            args.config,
            "--dry-run",
        ],
        dry_run=args.dry_run,
    )
    run([sys.executable, "src/metrics/summarize_all.py", "--root", "results"], dry_run=args.dry_run)


if __name__ == "__main__":
    main()
