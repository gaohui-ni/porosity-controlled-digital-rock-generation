import subprocess
import sys
from pathlib import Path


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    subprocess.run(
        [
            sys.executable,
            "scripts/run_pipeline.py",
            "--mode",
            "demo",
            "--config",
            "configs/main.yaml",
        ],
        cwd=root,
        check=True,
    )
