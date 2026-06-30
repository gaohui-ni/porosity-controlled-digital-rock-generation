import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def command(config="configs/main.yaml", dry_run=False):
    cmd = [sys.executable, "run_pipeline.py", "--mode", "full", "--config", config]
    if dry_run:
        cmd.append("--dry-run")
    return cmd


def run(config="configs/main.yaml", dry_run=False):
    subprocess.run(command(config=config, dry_run=dry_run), cwd=ROOT, check=True)


if __name__ == "__main__":
    run(dry_run=True)
