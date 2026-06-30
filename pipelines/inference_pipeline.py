import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def command(config="configs/main.yaml"):
    return [sys.executable, "run_pipeline.py", "--mode", "full", "--config", config, "--dry-run"]


def run(config="configs/main.yaml"):
    subprocess.run(command(config=config), cwd=ROOT, check=True)
