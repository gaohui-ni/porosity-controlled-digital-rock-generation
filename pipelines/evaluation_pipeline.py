import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def command(root="results"):
    return [sys.executable, "src/metrics/summary.py", "--root", root]


def run(root="results"):
    subprocess.run(command(root=root), cwd=ROOT, check=True)
