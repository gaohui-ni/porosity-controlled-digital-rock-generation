import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def command(config="configs/main.yaml", dry_run=False):
    cmd = [sys.executable, "scripts/run_pipeline.py", "--mode", "full", "--config", config]
    if dry_run:
        cmd.append("--dry-run")
    return cmd


def run(config="configs/main.yaml", dry_run=False):
    cmd = command(config=config, dry_run=dry_run)
    subprocess.run(cmd, cwd=ROOT, check=True)
