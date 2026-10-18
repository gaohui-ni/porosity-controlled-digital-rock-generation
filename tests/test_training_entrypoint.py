import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "scripts" / "train_256_vqvae_ddpm_lat64_v6_light96_full.py"
PIPELINE_SCRIPT = ROOT / "scripts" / "run_pipeline.py"
SPATIAL_TRAIN_SCRIPT = ROOT / "scripts" / "train_256_vqvae_ddpm_spatial_vqaware_v3_film.py"


def test_official_training_script_is_complete():
    source = TRAIN_SCRIPT.read_text(encoding="utf-8-sig")

    required_symbols = [
        "class VQVAE256Down4Light",
        "class UNetLatentCond",
        "def train_vqvae",
        "def train_ddpm",
        "def sample_one",
        "def parse_args",
        "def main",
        'if __name__ == "__main__"',
        'p.add_argument("--raw_shape"',
        'choices=["vqvae", "ddpm", "sample", "all"]',
    ]

    for symbol in required_symbols:
        assert symbol in source


def test_full_pipeline_calls_training_stages():
    source = PIPELINE_SCRIPT.read_text(encoding="utf-8")

    assert "scripts/train_256_vqvae_ddpm_lat64_v6_light96_full.py" in source
    assert re.search(r'"--stage"\s*,\s*"vqvae"', source)
    assert re.search(r'"--stage"\s*,\s*"ddpm"', source)
    assert '"--raw_shape"' in source


def test_spatial_training_entrypoint_is_complete():
    source = SPATIAL_TRAIN_SCRIPT.read_text(encoding="utf-8-sig")

    required_symbols = [
        "class RockRegionDataset",
        "class VectorQuantizerEMA",
        "class LatentAE256Down4",
        "class UNetLatentCond",
        "def validate_split",
        "def local_porosity_loss",
        "def boundary_gradient_loss",
        "def codebook_consistency_loss",
        "def train_vae",
        "def train_ddpm",
        "def sample_single",
        '"--sample_checkpoint"',
        'choices=[',
    ]

    for symbol in required_symbols:
        assert symbol in source


def test_spatial_pipeline_is_additive():
    source = PIPELINE_SCRIPT.read_text(encoding="utf-8")

    assert "def spatial_vqaware_pipeline" in source
    assert '"--mode"' in source
    assert '"spatial"' in source
    assert "scripts/train_256_vqvae_ddpm_lat64_v6_light96_full.py" in source
