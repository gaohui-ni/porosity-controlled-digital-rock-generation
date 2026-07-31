from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.volume_io import load_npz_array
from src.utils.naming import phi_tag
from scripts.compare_metric_tables import aggregate
from scripts.prepare_fontainebleau_real_sets import select_patches


ROOT = Path(__file__).resolve().parents[1]


def test_phi_tag_preserves_significant_porosity_digits():
    expected = {
        0.11: "0p11",
        0.13: "0p13",
        0.2045: "0p2045",
        0.1743: "0p1743",
        0.1263: "0p1263",
        0.0853: "0p0853",
    }

    assert {value: phi_tag(value) for value in expected} == expected


def test_generated_npz_reader_selects_seg_and_validates_porosity(tmp_path):
    path = tmp_path / "sample.npz"
    probability = np.full((4, 4, 4), 0.37, dtype=np.float32)
    segmentation = np.zeros((4, 4, 4), dtype=np.uint8)
    segmentation.ravel()[:16] = 1
    np.savez(
        path,
        prob=probability,
        seg=segmentation,
        seg_porosity=float(segmentation.mean()),
    )

    loaded = load_npz_array(path, key="seg", validate_porosity=True)

    assert np.array_equal(loaded, segmentation)
    assert float(loaded.mean()) == 0.25


def test_pnm_requires_generated_seg_key():
    source = (ROOT / "scripts" / "evaluate_pore_network_6panel.py").read_text(encoding="utf-8")

    assert 'GEN_NPZ_KEY = "seg"' in source
    assert '"--gen-npz-key"' in source
    assert 'validate_porosity=(npz_key == "seg")' in source


def test_final_model_configuration_and_pipeline_are_separate():
    main = (ROOT / "configs" / "main.yaml").read_text(encoding="utf-8")
    fontainebleau = (ROOT / "configs" / "fontainebleau_config.yaml").read_text(encoding="utf-8")
    pipeline = (ROOT / "scripts" / "run_pipeline.py").read_text(encoding="utf-8")

    assert "raw_path: data/raw/Bei_800x800x800.raw" in main
    assert "epochs_vae: 50" in main
    assert "epochs_ddpm: 300" in main
    assert "epochs_vae: 80" in fontainebleau
    assert "epochs_ddpm: 150" in fontainebleau
    assert "poro_center: 0.13" in fontainebleau
    assert 'str(fb["epochs_vae"])' in pipeline
    assert 'str(fb["epochs_ddpm"])' in pipeline
    assert 'target_folders = [f"phi{phi_tag(value)}"' in pipeline


def test_fontainebleau_training_script_default_matches_checkpoint():
    source = (ROOT / "scripts" / "train_fontainebleau.py").read_text(encoding="utf-8")

    assert 'default=0.13' in source
    assert "Center used to normalize the porosity condition before FiLM injection." in source


def test_pipeline_contains_real_generated_comparisons():
    source = (ROOT / "scripts" / "run_pipeline.py").read_text(encoding="utf-8")

    assert "real_voxel_perm.csv" in source
    assert "generated_voxel_perm.csv" in source
    assert "permeability_comparison.csv" in source
    assert "Prepare Fontainebleau real validation patches" in source
    assert "Evaluate Fontainebleau S2, lineal path, and EDT" in source
    assert "Evaluate Fontainebleau topology" in source
    assert "Evaluate Fontainebleau PNM six-panel descriptors" in source


def test_metric_comparison_aggregates_by_target(tmp_path):
    path = tmp_path / "metrics.csv"
    pd.DataFrame(
        {
            "target_tag": ["phi0p13", "phi0p13"],
            "target_phi": [0.13, 0.13],
            "porosity": [0.129, 0.131],
            "perm_x_m2": [1.0e-12, 3.0e-12],
            "status": ["ok", "ok"],
        }
    ).to_csv(path, index=False)

    result = aggregate(path, "real")

    assert result.loc[0, "real_porosity_mean"] == 0.13
    assert result.loc[0, "real_perm_x_m2_count"] == 2


def test_fontainebleau_patch_selection_returns_closest_patches():
    volume = np.zeros((8, 8, 8), dtype=np.uint8)
    volume[::2] = 1

    selected = select_patches(volume, target=0.5, patch=4, stride=4, count=2, min_sep=1)

    assert len(selected) == 2
    assert selected[0][0] == 0.0
