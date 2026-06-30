import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_quick_check():
    subprocess.run([sys.executable, "run_demo.py"], cwd=ROOT, check=True)


def print_full_workflow():
    subprocess.run(
        [sys.executable, "run_pipeline.py", "--mode", "full", "--config", "configs/main.yaml", "--dry-run"],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    run_quick_check()
    print_full_workflow()
