from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from src.utils.volume_io import load_npz_array
from src.utils.naming import phi_tag
from src.metrics.euler_characteristic import euler_characteristic_curve
from src.utils.quality_gate import quality_gate_message
from scripts.compare_metric_tables import aggregate
from scripts.export_source_data import export_pnm
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


def test_euler_curve_at_zero_uses_original_pore_space_without_normalization():
    volume = np.zeros((7, 7, 7), dtype=bool)
    volume[1, 1, 1] = True
    volume[5, 5, 5] = True

    radii, curve = euler_characteristic_curve(
        volume,
        voxel_size_um=1.0,
        radius_step_um=1.0,
        radius_max_um=1.0,
    )

    assert radii[0] == 0.0
    assert curve[0] == 2.0


def test_evaluation_quality_gate_is_strict_by_default():
    with np.testing.assert_raises_regex(RuntimeError, "all 10 samples failed"):
        quality_gate_message("topology phi0p11 real", 10, 0, 0.9, False)

    with np.testing.assert_raises_regex(RuntimeError, "below the required"):
        quality_gate_message("permeability real", 10, 8, 0.9, False)

    warning = quality_gate_message("permeability debug", 10, 8, 0.9, True)
    assert "80.0%" in warning
    assert quality_gate_message("permeability real", 10, 9, 0.9, False) is None


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


def test_pnm_outputs_preserve_full_target_porosity(tmp_path):
    root = tmp_path / "pnm"
    root.mkdir()
    np.savez(
        root / "curves_phi0p2045.npz",
        target_porosity=np.float64(0.2045),
        edt_centers=np.array([1.0, 2.0]),
    )
    output = tmp_path / "fig7_pnm.csv"

    export_pnm(root, output)
    exported = pd.read_csv(output)

    assert exported["target_phi"].tolist() == [0.2045, 0.2045]
    assert exported["metric"].tolist() == ["edt_centers", "edt_centers"]

    source = (ROOT / "scripts" / "evaluate_pore_network_6panel.py").read_text(encoding="utf-8")
    assert "tag = phi_tag(target_value)" in source
    assert 'f"six_panel_phi{tag}.png"' in source
    assert 'f"curves_phi{tag}.npz"' in source
    assert '"target_porosity": np.float64(target_value)' in source


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


def test_figure_helper_defaults_to_demo_and_requires_full_opt_in():
    command = [
        sys.executable,
        str(ROOT / "scripts" / "reproduce_figures.py"),
        "--config",
        "configs/main.yaml",
        "--dry-run",
    ]
    result = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)

    assert "run_pipeline.py --mode demo" in result.stdout
    assert "run_pipeline.py --mode full" not in result.stdout


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


def test_metric_comparison_retains_partial_permeability_rows(tmp_path):
    path = tmp_path / "partial_metrics.csv"
    pd.DataFrame(
        {
            "target_tag": ["0p11"],
            "target_phi": [0.11],
            "porosity": [0.109],
            "Kx_m2": [1.0e-12],
            "Ky_m2": [np.nan],
            "Kz_m2": [2.0e-12],
            "Kgeom_m2": [np.nan],
            "perm_status_x": ["ok"],
            "perm_status_y": ["not_percolating"],
            "perm_status_z": ["ok"],
            "n_valid_directions": [2],
            "status": ["ok"],
        }
    ).to_csv(path, index=False)

    result = aggregate(path, "real")

    assert result.loc[0, "real_porosity_count"] == 1
    assert result.loc[0, "real_Kx_m2_count"] == 1
    assert result.loc[0, "real_Ky_m2_count"] == 0
    assert result.loc[0, "real_Kz_m2_count"] == 1
    assert result.loc[0, "real_Kgeom_m2_count"] == 0


def test_fontainebleau_patch_selection_returns_closest_patches():
    volume = np.zeros((8, 8, 8), dtype=np.uint8)
    volume[::2] = 1

    selected = select_patches(volume, target=0.5, patch=4, stride=4, count=2, min_sep=1)

    assert len(selected) == 2
    assert selected[0][0] == 0.0
