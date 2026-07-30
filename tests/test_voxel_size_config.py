from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_manuscript_voxel_size_is_configured():
    main_cfg = (ROOT / "configs" / "main.yaml").read_text(encoding="utf-8")
    experiment_cfg = (ROOT / "configs" / "experiment_main.yaml").read_text(encoding="utf-8")

    assert "voxel_size_m: 3.5e-6" in main_cfg
    assert "voxel_size_m: 3.5e-6" in experiment_cfg


def test_pipeline_passes_voxel_size_to_permeability_script():
    source = (ROOT / "scripts" / "run_pipeline.py").read_text(encoding="utf-8")

    assert '"--voxel_size"' in source
    assert 'str(data["voxel_size_m"])' in source


def test_metric_scripts_default_to_manuscript_voxel_size():
    perm_source = (ROOT / "scripts" / "evaluate_voxel_and_perm.py").read_text(encoding="utf-8")
    pnm_source = (ROOT / "scripts" / "evaluate_pore_network_6panel.py").read_text(encoding="utf-8")

    assert "default=3.5e-6" in perm_source
    assert "VOXEL_SIZE = 3.5e-6" in pnm_source
    assert "data.voxel_size_m" in pnm_source
