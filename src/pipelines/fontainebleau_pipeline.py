import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def protocol_path():
    return ROOT / "docs" / "fontainebleau_protocol.md"


def dry_run_commands():
    return [
        [
            sys.executable,
            "scripts/train_fontainebleau.py",
            "--stage",
            "all",
            "--raw_path",
            "data/fontainebleau/fontainebleau_phi0p2045.raw",
            "--raw_shape",
            "480",
            "480",
            "480",
            "--save_dir",
            "outputs/fontainebleau_phi0p2045",
            "--poro_center",
            "0.2045",
            "--target_porosity",
            "0.2045",
            "--device",
            "cuda",
        ],
        [
            sys.executable,
            "scripts/generate_batch.py",
            "--ckpt_dir",
            "outputs/fontainebleau_phi0p2045",
            "--out_root",
            "data/generated_fontainebleau_sets",
            "--targets",
            "0.2045",
            "0.1743",
            "0.1263",
            "0.0853",
            "--n_per_target",
            "50",
            "--poro_center",
            "0.2045",
            "--device",
            "cuda",
        ],
    ]


def print_dry_run():
    for cmd in dry_run_commands():
        print(" ".join(cmd))


def run_dry_run():
    subprocess.run([sys.executable, "-c", "from src.pipelines.fontainebleau_pipeline import print_dry_run; print_dry_run()"], cwd=ROOT, check=True)
