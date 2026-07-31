# Journal Compliance Map

This document maps the repository contents to the Computers & Geosciences code, data, and reproducibility expectations.

## Scope Fit

The repository supports a manuscript at the interface of computing and geosciences:

- computational contribution: 3D VQ-VAE, FiLM-conditioned latent diffusion, quantile-based porosity projection, and a reproducible digital-rock generation workflow;
- geoscientific contribution: porosity-controllable reconstruction and validation of three-dimensional porous geomaterials from sandstone micro-CT data.

## Repository Requirements

| Journal expectation | Repository location |
| --- | --- |
| Public repository | `https://github.com/gaohui-ni/porosity-controlled-digital-rock-generation` |
| Clear license | `LICENSE` |
| English README with installation and usage | `README.md` |
| Dependencies and computational requirements | `requirements.txt`, `requirements_optional.txt`, `environment.yml`, `README.md`, `docs/dependency_versions.md` |
| Code for the scientific workflow | `src/`, `scripts/`, `pipelines/` |
| Official reproduction entry point | `run_pipeline.py` |
| Lightweight reproducible example | `run_demo.py`, `scripts/demo_quantile_binarization.py`, `examples/` |
| Synthetic test case when full raw experimental data are not bundled | `examples/`, `notebooks/tutorials/0001-basic-usage.ipynb` |
| Tutorials / how-to files | `notebooks/tutorials/`, `docs/user_guide.md`, `docs/reproducibility.md` |
| User guide with inputs and outputs | `docs/user_guide.md` |
| Data availability and redistribution limits | `docs/data_availability.md` |
| Trained checkpoints | Large checkpoint files are distributed through Git LFS; layout and checksums are documented in `docs/checkpoints.md` and `docs/model_manifest.md` |
| Figure/result reproduction map | `docs/figure_mapping.md`, `docs/figure_reproduction.md` |
| Automated lightweight checks | `.github/workflows/tests.yml`, `tests/` |

## Reproducibility Levels

1. **Lightweight check without raw data**

```bash
python run_demo.py
```

2. **Pipeline logic check without training**

```bash
python run_pipeline.py --mode full --config configs/main.yaml --dry-run
```

3. **Main-sandstone workflow with the released Mendeley Data volume**

```bash
python run_pipeline.py --mode main --config configs/main.yaml
```

The complete `full` workflow additionally requires the external ANU
Fontainebleau volumes, which are not redistributed in this repository.

## Data and Model Limitations

The main sandstone micro-CT volume is referenced through Mendeley Data rather
than stored in GitHub. The external ANU Fontainebleau volumes are not
redistributed. Final checkpoints are distributed through Git LFS. The
repository provides synthetic examples, configurations, and
training/generation/evaluation scripts so reviewers can inspect the workflow
and rerun the portions for which they have the required data and compute
environment.

## Computational Requirements

- Lightweight tests and tutorials can be run on CPU.
- Full 256^3 model training and batch generation are intended for a CUDA-capable workstation/server.
- Pore-network and permeability analysis require optional dependencies listed in `requirements_optional.txt`.
- Default manuscript-scale raw data path and shape are defined in `configs/main.yaml`.

## Notes Before Submission

- Confirm the repository remains public during review.
- Confirm `configs/main.yaml` matches the manuscript experiments.
- Confirm final figure numbers in the manuscript match `docs/figure_mapping.md`.
- Confirm the final checkpoint distribution route remains consistent across `README.md`, `docs/checkpoints.md`, `docs/model_manifest.md`, and the manuscript.
- Core direct dependency versions from the final experiment environment are
  recorded under `environment/`. A complete transitive environment export and
  clean-environment reconstruction remain pending release checks.
- Generate final `results/results_summary.json`, `results/summary.json`, and `results/results_summary.csv` only after the full run.
