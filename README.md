# Porosity-Controllable 3D Porous Media Generation

Core implementation for the manuscript **A Porosity-Controllable Generative Framework for Three-Dimensional Porous Media Based on Discrete Latent-Space Diffusion**.

## Authors

Gaohui Ni, Yanyan Ma, Shaowei Ma, Yuxin Yang, Xuefeng Liu, and Hao Ni

This repository implements the second-version manuscript workflow for
porosity-controllable 3D digital rock generation: a spatially separated 3D
VQ-VAE, a FiLM-conditioned latent DDPM, and topology-preserving adaptive
quantile binarization (TP-AQB) for precise output porosity matching.

## Methodological Contributions

The method addresses three practical problems in 3D digital-rock generation:
the cost of voxel-space diffusion, learning from limited physical samples,
and controlling porosity without discarding important pore connections.

The principal methodological novelty lies in a dual-control discrete
latent-space framework that couples learned porosity-dependent structural
generation with exact output-space porosity enforcement for efficient and
physically evaluated `256^3` digital-rock generation under limited-sample
conditions.

A lightweight 3D VQ-VAE compresses `256^3`-voxel micro-CT volumes into a
`32 x 64 x 64 x 64` discrete latent tensor. Continuous target porosity
modulates latent denoising features through FiLM. TP-AQB starts from the
model's fixed-threshold structure, adjusts the pore count under skeleton,
throat-neighborhood, and topology constraints, and uses equal-count local
swaps to reduce specific-surface drift.

The main sandstone evaluation uses 95 real samples from the spatially held-out
test region and 95 generated samples conditioned on their corresponding real
porosities. The four nominal groups contain 8, 29, 39, and 19 pairs at target
porosities `0.11`, `0.12`, `0.13`, and `0.14`. Evaluation covers directional
two-point correlation, specific surface area, pore-network geometry and local
topology, and directional and geometric-mean permeability.

A separately trained Fontainebleau experiment evaluates cross-sample
porosity extrapolation within the same lithology. One `480^3` rock is used for
training and two lower-porosity rocks are used for validation. This is not a
zero-shot transfer of the laboratory-sandstone model and is not presented as
cross-lithology generalization.

Implementation locations, manuscript mappings, validation evidence, and the
scope of these framework-level claims are documented in
[docs/methodological_novelty.md](docs/methodological_novelty.md).

## Scope

The repository contains:

- 3D VQ-VAE and vector-quantized latent representation learning.
- FiLM-conditioned latent DDPM for target-porosity controlled generation.
- TP-AQB v2 for exact pore-count matching with topology, skeleton,
  throat-neighborhood, and specific-surface safeguards.
- A lightweight top-N quantile implementation retained for demonstrations and
  legacy checkpoint compatibility.
- A spatially disjoint train/validation/test workflow with EMA vector
  quantization, validation-based checkpoint selection, FiLM ablation, and
  calibrated conditional sampling.
- Training, sampling, batch-generation, and Fontainebleau validation scripts.
- Evaluation utilities for porosity, directional two-point correlation function `S2`, pore-size statistics, topology, pore-network features, and OpenPNM permeability workflows.
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
savedmodels/  Earlier-workflow compatibility checkpoints distributed through Git LFS.
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
Environment reconstruction guidance and the recorded runtime versions are
provided in [docs/dependency_versions.md](docs/dependency_versions.md).

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

`run_pipeline.py` is the official top-level entry point for the repository workflows.
`scripts/run_pipeline.py` is the internal modular implementation called by the top-level entry point and should not be used directly for reviewer reproduction.

The second-version main-sandstone implementation uses the publicly released
Mendeley Data volume:

```bash
python run_pipeline.py --mode spatial --config configs/main.yaml
```

Run `--mode fontainebleau` separately when the three external ANU volumes are
available. They are not redistributed in this repository. The `full` command
is retained for the earlier compatibility workflow and should not be confused
with the second-version `spatial` path.

Available modes:

- `demo`: lightweight synthetic porosity-control check;
- `spatial`: second-version manuscript implementation for spatially separated
  EMA-VQ/DDPM training, calibrated FiLM sampling, and TP-AQB v2;
- `fontainebleau`: separately trained Fontainebleau validation and
  porosity-extrapolation workflow;
- `main`: compatibility workflow for the earlier main-sandstone checkpoint;
- `full`: compatibility `main` workflow followed by `fontainebleau`;
- `final`: inference and evaluation with the compatibility checkpoints under
  `savedmodels/`, without retraining.

The `spatial` workflow is the primary implementation corresponding to the
second-version manuscript. It uses the released `800^3` parent sandstone
volume and reserves disjoint regions for training, validation, and final
testing. Its configuration is under `spatial_vqaware` in `configs/main.yaml`;
implementation and provenance are documented in
[docs/spatial_vqaware_workflow.md](docs/spatial_vqaware_workflow.md).

The second-version final sampling package is released under
`savedmodels/spatial_vqaware_v2/`. It contains the calibrated R03 checkpoint,
the matching VQ-VAE and latent statistics, the frozen 100-crop table (95 crops
used in the four manuscript groups), and the final seed and pair manifests.
The earlier compatibility checkpoint sets remain available in their existing
directories.

For a lightweight reviewer sanity check:

```bash
python run_demo.py
```

To inspect the second-version main-sandstone command sequence without
launching training or generation:

```bash
python run_pipeline.py --mode spatial --config configs/main.yaml --dry-run
```

`--dry-run` is a pipeline logic test. It builds the same ordered command sequence as the selected reproduction run and writes `results/pipeline_<mode>_manifest.json`, but it does not train models, generate samples, or evaluate outputs.

To regenerate the frozen Test-95 probability volumes from the released final
sampling package, first run `git lfs pull`, place the published raw volume at
the configured path, and execute:

```bash
python scripts/generate_spatial_test95.py \
  --raw-path data/raw/Bei_800x800x800.raw \
  --device cuda

python scripts/apply_tp_aqb.py \
  --input-root outputs/spatial_test95/gen \
  --output-root data/generated_phi_sets_spatial_test95 \
  --config configs/tp_aqb_v2.yaml
```

The first command reproduces the single probability field and fixed/top-N
baselines for each crop. The second command applies the manuscript TP-AQB v2
recipe to those probability fields.

Compatibility `full` runs and current `spatial` runs write separate command
manifests and output roots. In particular:

- current command manifest: `results/pipeline_spatial_manifest.json`;
- current checkpoint and model outputs: `outputs/spatial_vqaware/`;
- TP-AQB volumes: `data/generated_phi_sets_spatial/`;
- compatibility command manifest: `results/pipeline_full_manifest.json`;
- figure-oriented outputs: `results/fig_s2/`, `results/fig_perm/`, `results/fig_pnm/`, and `results/fig_fontainebleau/`;
- manuscript tables: `results/tables/`;
- summary files: `results/results_summary.json`, `results/summary.json`, and `results/results_summary.csv`.

## Tests

Install the lightweight test dependency and run:

```bash
pip install -r requirements_dev.txt
pytest tests/
```

The tests cover conventional quantile matching, TP-AQB exact pore counts and
topology preservation, porosity calculation, directional two-point
correlation-function output shapes, and pipeline dry-run behavior.

## Reproduce Manuscript Outputs

The second-version manuscript-scale experiment is configured in [configs/main.yaml](configs/main.yaml). [configs/experiment_main.yaml](configs/experiment_main.yaml) is retained as a backward-compatible copy/example. See [docs/figure_reproduction.md](docs/figure_reproduction.md) for a command-by-command figure reproduction map. The pipeline writes figure-oriented outputs under `results/`:

- `results/fig_s2/`: directional two-point correlation function `S2`, lineal-path, and EDT pore-size curves.
- `results/fig_perm/`: permeability-related tables and plots.
- `results/fig_pnm/`: topology and pore-network descriptors.
- `results/fig_fontainebleau/`: Fontainebleau validation outputs.
- `results/tables/`: CSV tables used for manuscript plots.

Current manuscript commands:

```bash
python run_pipeline.py --mode spatial --config configs/main.yaml
python run_pipeline.py --mode fontainebleau --config configs/main.yaml
python src/metrics/summarize_all.py --root results
python scripts/plot_all.py
```

The `main`, `full`, and `final` modes remain available for compatibility with
the earlier checkpoint layout.

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

The repository-to-journal requirements map is provided in
[docs/journal_compliance.md](docs/journal_compliance.md). Summary files are
generated locally by the evaluation workflow and are not distributed in the
public repository.

## Main Manuscript Workflow

See [docs/reproducibility.md](docs/reproducibility.md) and [docs/user_guide.md](docs/user_guide.md) for the full workflow.

At a high level:

1. Place the binary `800^3` laboratory sandstone volume under `data/raw/`.
2. Run the spatially separated 3D VQ-VAE and FiLM latent-DDPM workflow.
3. Generate calibrated probability volumes and apply TP-AQB v2.
4. Compare the 95 held-out real samples and 95 matched-condition generated
   samples across the four nominal porosity groups.
5. Evaluate spatial statistics, specific surface area, pore-network geometry,
   local topology, and permeability.
6. Run the separately trained three-volume Fontainebleau experiment when the
   third-party data are available.

## Data Availability

The laboratory sandstone micro-CT volume data and associated metadata used in this study are available in Mendeley Data:

https://doi.org/10.17632/vp2yw9c7jj.1

The repository also includes synthetic examples and scripts that allow reviewers and users to test the workflow on provided or user-supplied binary 3D volumes.

See [docs/data_availability.md](docs/data_availability.md).

## Released Models and Checkpoints

### Download pretrained checkpoints

All `.pth` and `.npz` files under `savedmodels/` are tracked with Git LFS.
Clone the repository and retrieve the LFS objects before checkpoint inference:

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

The released model layout is:

```text
savedmodels/
|-- spatial_vqaware_v2/
|   |-- R03_calibrated/
|   |   |-- ae_best.pth
|   |   |-- calib_A_scale.pth
|   |   `-- latent_stats.npz
|   |-- optional_precalibration/unet_best_noise.pth
|   |-- test_cohort_v1/test_crop_porosity_all_100.csv
|   `-- final_test_v2/*_MANIFEST.csv
|-- main_sandstone/
|   |-- vqvae_final.pth
|   |-- unet_final.pth
|   `-- latent_stats.npz
`-- fontainebleau_phi0p2045/
    |-- vqvae_final.pth
    |-- unet_final.pth
    `-- latent_stats.npz
```

Checksums are provided in [docs/model_manifest.md](docs/model_manifest.md) and
`savedmodels/spatial_vqaware_v2/MANIFEST.sha256`.

After `git lfs pull`, verify the spatial release with:

```bash
python scripts/verify_spatial_release.py --check-tensors
```

The VQ-VAE, FiLM-conditioned latent-DDPM, calibrated sampling, and TP-AQB
workflow is provided so users can retrain using the publicly available raw
data and `configs/main.yaml`.

A completed training run produces:

- `vqvae_final.pth`
- `unet_final.pth`
- `latent_stats.npz`

The released spatial package records the exact weights, crop selection, and
seed mapping used by the 95-pair test. Generated probability and binary volumes
are not bundled because they occupy approximately 5.9 GB; they can be
regenerated from the released package and the published `800^3` source volume.
Because diffusion sampling is stochastic and numerical
results can depend on hardware and software versions, generated volumes may not
be bitwise identical across computational environments. Independently retrained
models are expected to reproduce the workflow and statistical behavior rather
than identical samples.

See [docs/checkpoints.md](docs/checkpoints.md) for details.

## Citation

If you use this repository, please cite the associated manuscript. A machine-readable citation file is provided in [CITATION.cff](CITATION.cff).

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
