# Porosity-Controllable 3D Porous Media Generation

Core implementation for the manuscript **A Porosity-Controllable Generative Framework for Three-Dimensional Porous Media Based on Discrete Latent-Space Diffusion**.

This repository provides a reproducible code base for 3D digital rock generation with a lightweight 3D VQ-VAE, a FiLM-conditioned latent DDPM, and quantile-based adaptive binarization for explicit porosity control.

## Scope

The repository contains:

- 3D VQ-VAE and vector-quantized latent representation learning.
- FiLM-conditioned latent DDPM for target-porosity controlled generation.
- Quantile-based binarization for matching prescribed porosity.
- Training, sampling, batch-generation, and Fontainebleau validation scripts.
- Evaluation utilities for porosity, directional two-point probability function `S2`, pore-size statistics, topology, pore-network features, and OpenPNM permeability workflows.
- Synthetic examples that can be used without restricted raw micro-CT data.

## Repository Structure

```text
configs/      Example YAML configurations.
src/          Importable model, data, training, sampling, metric, and utility code.
scripts/      Command-line workflows for training, generation, and evaluation.
examples/     Small synthetic demo data and demo outputs.
docs/         Installation, user guide, reproducibility, and data/code availability notes.
data/         Placeholder folders for user-provided raw data.
outputs/      Placeholder folder for generated checkpoints and outputs.
results/      Placeholder folder for generated figures, curves, and tables.
savedmodels/  Final trained model checkpoints distributed through Git LFS.
environment/  Exact package versions used for the manuscript-scale workflow.
```

## Installation

Conda is recommended:

```bash
conda env create -f environment/environment-lock.yml
conda activate vq256_cuda
pip install -e .
```

Alternatively:

```bash
pip install -r requirements.txt
pip install -r requirements_optional.txt
pip install -e .
```

`porespy` and `openpnm` are optional for the lightweight demo but required for
pore-network and permeability workflows. Core direct dependency versions from
the manuscript environment are pinned in
`environment/environment-lock.yml` and `environment/requirements-lock.txt`.
A complete package-manager snapshot and clean-environment reconstruction remain
release-validation steps; see
[docs/dependency_versions.md](docs/dependency_versions.md).

## Computational Requirements

- Lightweight demo, notebooks, and unit tests: CPU only; no raw micro-CT data or trained checkpoint is required.
- Manuscript-scale training and generation: CUDA-capable PyTorch installation recommended; default generated/comparison sample size is `256 x 256 x 256` voxels.
- Pore-network and permeability workflows: install `requirements_optional.txt`; these steps can be memory- and time-intensive.
- Raw data: the default full configuration expects the released Mendeley file at `data/raw/Bei_800x800x800.raw` with shape `800 x 800 x 800`, following [docs/data_availability.md](docs/data_availability.md).
- Voxel size: the manuscript-scale laboratory sandstone data use `3.5e-6` m/voxel, configured as `data.voxel_size_m` in [configs/main.yaml](configs/main.yaml).

## Minimal Demo

Run the synthetic porosity-matching demo:

```bash
python scripts/demo_quantile_binarization.py
```

Expected behavior:

- writes `examples/demo_input.npy`;
- writes `examples/demo_output/demo_seg.npy`;
- prints the target porosity, achieved porosity, check porosity, and adaptive threshold.

This demo is intentionally lightweight and does not require restricted micro-CT data or trained checkpoints.

A notebook version of this workflow is available at:

```text
notebooks/tutorials/0001-basic-usage.ipynb
notebooks/tutorials/0001-basic-usage-synthetic-rock.ipynb
notebooks/tutorials/0002-porosity-control.ipynb
notebooks/tutorials/0003-fontainebleau-validation.ipynb
```

## Official Reproduction Entry Point

`run_pipeline.py` is the ONLY official entry point for reproducing the manuscript workflow.
`scripts/run_pipeline.py` is the internal modular implementation called by the top-level entry point and should not be used directly for reviewer reproduction.

The complete manuscript workflow can be reproduced through model retraining using the released code, configurations, and public data:

```bash
python run_pipeline.py --mode full --config configs/main.yaml
```

Available modes:

- `demo`: lightweight synthetic porosity-control check;
- `main`: main laboratory sandstone workflow, including training, generation, spatial statistics, permeability, topology, and PNM six-panel descriptors;
- `fontainebleau`: independent Fontainebleau validation workflow;
- `full`: `main` followed by `fontainebleau`;
- `final`: generate and evaluate using final checkpoint files under `savedmodels/`, without retraining.

For a lightweight reviewer sanity check:

```bash
python run_demo.py
```

To validate the full pipeline logic without launching training, generation, or evaluation:

```bash
python run_pipeline.py --mode full --config configs/main.yaml --dry-run
```

`--dry-run` is a pipeline logic test. It builds the same ordered command sequence as the selected reproduction run and writes `results/pipeline_<mode>_manifest.json`, but it does not train models, generate samples, or evaluate outputs.

The full run writes a standardized execution trace and result structure:

- command manifest: `results/pipeline_full_manifest.json`;
- checkpoint and model outputs: `outputs/main_sandstone/`;
- figure-oriented outputs: `results/fig_s2/`, `results/fig_perm/`, `results/fig_pnm/`, and `results/fig_fontainebleau/`;
- manuscript tables: `results/tables/`;
- summary files: `results/results_summary.json`, `results/summary.json`, and `results/results_summary.csv`.

## Tests

Install the lightweight test dependency and run:

```bash
pip install -r requirements_dev.txt
pytest tests/
```

The tests cover quantile-based porosity matching, porosity calculation, directional two-point probability-function output shapes, and pipeline dry-run behavior.

## Reproduce Manuscript Outputs

The official manuscript-scale experiment is configured in [configs/main.yaml](configs/main.yaml). [configs/experiment_main.yaml](configs/experiment_main.yaml) is retained as a backward-compatible copy/example. See [docs/figure_reproduction.md](docs/figure_reproduction.md) for a command-by-command figure reproduction map. The pipeline writes figure-oriented outputs under `results/`:

- `results/fig_s2/`: directional two-point probability function `S2`, lineal-path, and EDT pore-size curves.
- `results/fig_perm/`: permeability-related tables and plots.
- `results/fig_pnm/`: topology and pore-network descriptors.
- `results/fig_fontainebleau/`: Fontainebleau validation outputs.
- `results/tables/`: CSV tables used for manuscript plots.

Typical commands:

```bash
python run_pipeline.py --mode full --config configs/main.yaml
python run_pipeline.py --mode main --config configs/main.yaml
python run_pipeline.py --mode fontainebleau --config configs/main.yaml
python run_pipeline.py --mode final --config configs/main.yaml
python src/metrics/summarize_all.py --root results
python scripts/plot_all.py
```

The summary command writes `results/results_summary.json`, `results/summary.json`, and `results/results_summary.csv`. A short manuscript-to-code map is provided in [docs/figure_mapping.md](docs/figure_mapping.md).

For a lightweight figure-output sanity check and figure-path helper:

```bash
python scripts/reproduce_figures.py --config configs/main.yaml
```

By default, this helper runs only the synthetic demo and validates the
figure-output path. It never starts manuscript-scale training unless
`--run-full-pipeline` is supplied explicitly. To plot from existing outputs
without running even the demo, use `--skip-pipeline`. Manuscript-scale figures
are generated from the full evaluation outputs listed in
[docs/figure_reproduction.md](docs/figure_reproduction.md).

Before submission, use [docs/submission_checklist.md](docs/submission_checklist.md) and [docs/journal_compliance.md](docs/journal_compliance.md) for the final repository audit. Precomputed summary files are intentionally not bundled before the final full run.

## Main Manuscript Workflow

See [docs/reproducibility.md](docs/reproducibility.md) and [docs/user_guide.md](docs/user_guide.md) for the full workflow.

At a high level:

1. Place a binary raw digital rock volume under `data/raw/`.
2. Build porosity-matched real comparison groups with `scripts/build_real_phi_groups.py`.
3. Train the 3D VQ-VAE and latent DDPM.
4. Generate batches at target porosity values.
5. Evaluate spatial statistics, topology, pore-network metrics, and permeability.
6. Repeat the validation workflow for Fontainebleau sandstone data when available.

## Data Availability

The laboratory sandstone micro-CT volume data and associated metadata used in this study are available in Mendeley Data:

https://doi.org/10.17632/vp2yw9c7jj.1

The repository also includes synthetic examples and scripts that allow reviewers and users to test the workflow on provided or user-supplied binary 3D volumes.

See [docs/data_availability.md](docs/data_availability.md).

## Final Models and Checkpoints

### Download pretrained checkpoints

The final checkpoints are tracked under `savedmodels/` with Git LFS. Clone the
repository and retrieve the model files before running checkpoint-based
inference:

```bash
git lfs install
git clone https://github.com/gaohui-ni/porosity-controlled-digital-rock-generation.git
cd porosity-controlled-digital-rock-generation
git lfs pull
```

Verify the downloaded files against the SHA256 values in
[docs/model_manifest.md](docs/model_manifest.md), then run:

```bash
python run_pipeline.py --mode final --config configs/main.yaml
```

The final trained model layout is:

```text
savedmodels/
|-- main_sandstone/
|   |-- vqvae_final.pth
|   |-- unet_final.pth
|   `-- latent_stats.npz
`-- fontainebleau_phi0p2045/
    |-- vqvae_final.pth
    |-- unet_final.pth
    `-- latent_stats.npz
```

The two `unet_final.pth` files exceed the normal GitHub 100 MB single-file limit
and therefore require Git LFS. A release archive may also be provided as a
browser-download alternative. Checksums are provided in
[docs/model_manifest.md](docs/model_manifest.md).

The complete VQ-VAE and latent-DDPM training workflow is also provided, so users can retrain the models using the publicly available raw data and `configs/main.yaml`.

A completed training run produces:

- `vqvae_final.pth`
- `unet_final.pth`
- `latent_stats.npz`

The released checkpoints enable reproduction of the reported model inference
and analysis workflow. Because diffusion sampling is stochastic and numerical
results can depend on hardware and software versions, generated volumes may not
be bitwise identical across computational environments. Independently retrained
models are expected to reproduce the workflow and statistical behavior rather
than identical samples.

See [docs/checkpoints.md](docs/checkpoints.md) for details.

## Citation

If you use this repository, please cite the associated manuscript. A machine-readable citation file is provided in [CITATION.cff](CITATION.cff).

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
