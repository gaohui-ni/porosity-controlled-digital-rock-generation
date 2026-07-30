from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "scripts" / "train_256_vqvae_ddpm_lat64_v6_light96_full.py"
PIPELINE_SCRIPT = ROOT / "scripts" / "run_pipeline.py"


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
    assert '"--stage",\n                "vqvae"' in source
    assert '"--stage",\n                "ddpm"' in source
    assert '"--raw_shape"' in source
