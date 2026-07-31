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
from scripts.validate_smoke_outputs import validate_npz
from scripts.plot_slice_porosity import slice_porosity_z


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


def test_release_smoke_output_validator(tmp_path):
    path = tmp_path / "sample.npz"
    segmentation = np.zeros((4, 4, 4), dtype=np.uint8)
    segmentation[:2] = 1
    np.savez(path, seg=segmentation, seg_porosity=np.float64(0.5))

    result = validate_npz(path)

    assert result["shape"] == [4, 4, 4]
    assert result["seg_porosity"] == 0.5


def test_slice_porosity_profile_uses_z_axis():
    volume = np.zeros((3, 4, 5), dtype=np.uint8)
    volume[:, :, 2] = 1

    profile = slice_porosity_z(volume)

    assert profile.tolist() == [0.0, 0.0, 1.0, 0.0, 0.0]


def test_figure_registry_maps_all_manuscript_figures():
    registry = (ROOT / "docs" / "figure_mapping.md").read_text(encoding="utf-8")

    for number in range(1, 10):
        assert f"| Fig. {number} |" in registry
    assert "| Figure | Result | Script or implementation | Required input | Local output |" in registry
    assert "scripts/plot_slice_porosity.py" in registry
    assert "Four main-text panels" in registry
    assert "(a-c)" in registry
    assert "(d) S/V comparison" in registry
    assert "used in the Supplementary Material" in registry
    assert "not a panel of main-text Fig. 6" in registry
    assert "At phi=0.15" in registry
    assert "manuscript panels exclude throat length, tortuosity, and Euler-characteristic" in registry
    assert "(a) Kx, Ky, Kz, and Kgeom at phi=0.15" in registry
    assert "Fontainebleau Kx, Ky, Kz, and Kgeom at phi=0.2045" in registry


def test_documented_main_workflow_matches_canonical_configuration():
    reproducibility = (ROOT / "docs" / "reproducibility.md").read_text(encoding="utf-8")
    user_guide = (ROOT / "docs" / "user_guide.md").read_text(encoding="utf-8")
    combined = reproducibility + user_guide

    for expected in (
        "--epochs_vae 50",
        "--epochs_ddpm 300",
        "--out_root results/fig_s2",
        "real_voxel_perm.csv",
        "generated_voxel_perm.csv",
        "permeability_comparison.csv",
    ):
        assert expected in combined

    for stale in ("results/curves", "results/topology"):
        assert stale not in combined

    assert "external ANU Fontainebleau volumes" in reproducibility
    assert "not redistributed" in reproducibility


def test_code_availability_and_readme_match_release_policy():
    statement = (ROOT / "docs" / "code_availability.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    for expected in (
        "# Computer Code Availability",
        "Gaohui Ni",
        "+86-151-9298-7839",
        "1946978288@qq.com",
        "Year first available:** 2026",
        "Program size:** Approximately 2.73 MB",
        "Git LFS",
        "MIT License",
    ):
        assert expected in statement

    assert "complete two-dataset workflow additionally requires" in readme
    assert "external ANU Fontainebleau volumes" in readme
    assert "A release archive may also be provided" not in readme
    assert "## Authors" in readme
    assert "Gaohui Ni, Yanyan Ma, Shaowei Ma, Yuxin Yang, Xuefeng Liu, and Hao Ni" in readme


def test_pnm_requires_generated_seg_key():
    source = (ROOT / "scripts" / "evaluate_pore_network_6panel.py").read_text(encoding="utf-8")

    assert 'GEN_NPZ_KEY = "seg"' in source
    assert '"--gen-npz-key"' in source
    assert 'validate_porosity=(npz_key == "seg")' in source
    assert '"--min-success-rate"' in source
    assert '"--allow-partial"' in source
    assert 'quality_gate_message(' in source
    assert '"gate_status"' in source
    assert '"n_valid_tau_directions"' in source
    assert '"tau_status_x"' in source
    assert '"tau_status_y"' in source
    assert '"tau_status_z"' in source
    assert '"n_real_tau_partial"' in source
    assert '"n_real_tau_failed"' in source


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


def test_manuscript_environment_is_version_locked():
    requirements = (ROOT / "environment" / "requirements-lock.txt").read_text(encoding="utf-8")
    conda = (ROOT / "environment" / "environment-lock.yml").read_text(encoding="utf-8")

    for requirement in (
        "torch==2.5.1",
        "numpy==2.2.6",
        "scipy==1.15.3",
        "porespy==3.0.2",
        "openpnm==3.5.2",
    ):
        assert requirement in requirements
    assert "python=3.10.12" in conda
    assert "pytorch=2.5.1" in conda
    assert "pytorch-cuda=12.1" in conda
    assert "cudnn=9.1" in conda


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
