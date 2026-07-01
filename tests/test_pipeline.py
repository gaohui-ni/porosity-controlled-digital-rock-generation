import subprocess
import sys


def test_run_pipeline_full_dry_run():
    completed = subprocess.run(
        [
            sys.executable,
            "run_pipeline.py",
            "--mode",
            "full",
            "--config",
            "configs/main.yaml",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Train VQ-VAE" in completed.stdout
    assert "Generate controlled samples" in completed.stdout
