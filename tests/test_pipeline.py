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
    assert "Evaluate PNM six-panel descriptors" in completed.stdout
    assert "Train Fontainebleau VQ-VAE and latent DDPM" in completed.stdout
    assert "Generate Fontainebleau validation samples" in completed.stdout


def test_run_pipeline_spatial_dry_run():
    completed = subprocess.run(
        [
            sys.executable,
            "run_pipeline.py",
            "--mode",
            "spatial",
            "--config",
            "configs/main.yaml",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Train spatial-split EMA VQ-VAE" in completed.stdout
    assert "Compute spatial-split latent statistics" in completed.stdout
    assert "Train FiLM latent DDPM with validation selection" in completed.stdout
    assert "unet_best_noise.pth" in completed.stdout
    assert "Apply topology-preserving AQB v2" in completed.stdout
