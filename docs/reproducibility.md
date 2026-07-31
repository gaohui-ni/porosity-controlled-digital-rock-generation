# Reproducibility

This repository is organized to support review of the computational workflow for the associated Computers & Geosciences manuscript.

## Reproducibility Levels

This repository supports the following levels of reproducibility:

1. **Lightweight functional reproduction**: run the synthetic demonstration and unit tests without raw micro-CT data or trained checkpoints.
2. **Main-sandstone workflow reproduction through retraining**: use the public Mendeley Data volume, supplied configuration files, and training scripts to retrain the VQ-VAE and latent DDPM, generate porosity-controlled samples, and rerun the main-sandstone evaluation workflow.
3. **Statistical manuscript-result reproduction**: independently retrained models are expected to reproduce the reported porosity-control accuracy, structural statistics, pore-network characteristics, and permeability trends within normal stochastic variation.

The final trained checkpoints are distributed under `savedmodels/` through Git
LFS and support reproduction of the inference and analysis workflow. Because
diffusion sampling and numerical results can depend on random seeds, hardware,
and software versions, bitwise-identical generated volumes are not guaranteed.

## Lightweight Functional Test

```bash
python scripts/demo_quantile_binarization.py
```

This validates the porosity-matching step and writes example arrays under `examples/`.

The same check can be run through the pipeline orchestrator:

```bash
python run_pipeline.py --mode demo --config configs/main.yaml
```

To inspect the manuscript-scale command sequence without launching training:

```bash
python run_pipeline.py --mode full --config configs/main.yaml --dry-run
```

Available pipeline modes are:

- `demo`: lightweight synthetic porosity-control check;
- `main`: main laboratory sandstone workflow, including PNM six-panel descriptors;
- `fontainebleau`: independent Fontainebleau validation workflow;
- `full`: `main` followed by `fontainebleau`;
- `final`: generate and evaluate using final checkpoint files under `savedmodels/`, without retraining.

## Main Workflow

The complete two-dataset pipeline is configured in `configs/main.yaml`.
It additionally requires access to the external ANU Fontainebleau volumes,
which are not redistributed in this repository:

```bash
python run_pipeline.py --mode full --config configs/main.yaml
```

To reproduce only the main laboratory sandstone results:

```bash
python run_pipeline.py --mode main --config configs/main.yaml
```

To use the final supplied checkpoints without retraining:

```bash
python run_pipeline.py --mode final --config configs/main.yaml
```

The `final` mode still requires the raw comparison volumes used by the
evaluation stages, including the external Fontainebleau volumes. For
main-sandstone-only reproduction, use `--mode main` with the released
Mendeley Data volume.

The equivalent step-by-step commands are:

```bash
python scripts/build_real_phi_groups.py \
  --raw_path data/raw/Bei_800x800x800.raw \
  --raw_shape 800 800 800 \
  --patch 256 \
  --stride 32 \
  --targets 0.11 0.12 0.13 0.14 0.15 \
  --n_per_target 100 \
  --out_root data/real256_sets_from_S1_strict
```

```bash
python scripts/train_256_vqvae_ddpm_lat64_v6_light96_full.py \
  --stage all \
  --raw_path data/raw/Bei_800x800x800.raw \
  --raw_shape 800 800 800 \
  --save_dir outputs/main_sandstone \
  --batch_vae 1 \
  --batch_ddpm 1 \
  --epochs_vae 50 \
  --epochs_ddpm 300 \
  --n_samples 1000 \
  --target_porosity 0.13 \
  --poro_center 0.13 \
  --poro_scale 0.02 \
  --device cuda
```

```bash
python scripts/generate_batch.py \
  --ckpt_dir outputs/main_sandstone \
  --out_root data/generated_phi_sets \
  --targets 0.11 0.12 0.13 0.14 0.15 \
  --n_per_target 100 \
  --poro_center 0.13 \
  --device cuda
```

After `git lfs pull`, use `--ckpt_dir savedmodels/main_sandstone` for
main-sandstone generation.

## Evaluation Workflow

Directional two-point correlation function `S2`, lineal-path statistics, and EDT pore-size statistics:

```bash
python scripts/evaluate_s2_lineal_edt.py \
  --real_root data/real256_sets_from_S1_strict \
  --gen_root data/generated_phi_sets \
  --out_root results/fig_s2 \
  --targets 0.11 0.12 0.13 0.14 0.15
```

Voxel metrics and permeability for both real and generated groups:

```bash
python scripts/evaluate_voxel_and_perm.py \
  --input_root data/real256_sets_from_S1_strict \
  --output_csv results/tables/real_voxel_perm.csv \
  --group_name real \
  --shape 256 256 256 \
  --voxel_size 3.5e-6 \
  --recursive \
  --extensions .raw

python scripts/evaluate_voxel_and_perm.py \
  --input_root data/generated_phi_sets \
  --output_csv results/tables/generated_voxel_perm.csv \
  --group_name gen \
  --shape 256 256 256 \
  --voxel_size 3.5e-6 \
  --recursive \
  --extensions .npz

python scripts/compare_metric_tables.py \
  --real-csv results/tables/real_voxel_perm.csv \
  --gen-csv results/tables/generated_voxel_perm.csv \
  --output-csv results/tables/permeability_comparison.csv
```

Permeability is solved independently along the three Cartesian directions.
Basic voxel metrics are retained even when one or more directional OpenPNM
solves fail. A non-percolating or failed direction is recorded as `NaN`, with
the reason in `perm_status_x`, `perm_status_y`, or `perm_status_z`, and is
excluded only from that directional aggregate. `Kgeom` is reported only when
all three directional permeabilities are finite and positive. The
`n_valid_directions` column records the number of successful directions for
each sample.

Evaluation commands are strict by default: every requested topology group must
contain at least one successful sample, and topology, permeability, and PNM
target success rates must be at least 90%. Each run writes an
evaluation-quality summary. The PNM `all_targets_summary.json` contains both
the per-target results and the quality-gate outcome.
Use `--min-success-rate` to set a different declared threshold.
`--allow-partial` is intended only for debugging: it converts a below-threshold
rate to a warning, but a group with zero successful results still exits
non-zero.

Topology:

```bash
python scripts/evaluate_coordination_euler.py \
  --real-root data/real256_sets_from_S1_strict \
  --gen-root data/generated_phi_sets \
  --out-root results/fig_pnm
```

Extended pore-network diagnostic analysis:

```bash
python scripts/evaluate_pore_network_6panel.py --config configs/main.yaml
```

Verify paths and target values in `configs/main.yaml`; modification of the
Python source file is not required. The manuscript-scale voxel size is read
from `configs/main.yaml`.

Summarize generated result files:

```bash
python src/metrics/summarize_all.py --root results
```

For a figure-oriented mapping from manuscript result groups to commands and output folders, see `docs/figure_reproduction.md`.

## Fontainebleau Workflow

The Fontainebleau validation can be launched through the official entry point
after all four prepared raw volumes exist at the paths configured in
`configs/fontainebleau_config.yaml`. These external ANU volumes are not
redistributed in this repository:

```bash
python run_pipeline.py --mode fontainebleau --config configs/main.yaml
```

Prepare each of the four volumes separately:

```bash
python scripts/prepare_fontainebleau_data.py \
  --input /path/to/fontainebleau.raw \
  --output_raw data/fontainebleau/phi0p2045.raw \
  --raw_shape 480 480 480 \
  --pore_value 1
```

Repeat for target porosities `0.1743`, `0.1263`, and `0.0853`, writing
`phi0p1743.raw`, `phi0p1263.raw`, and `phi0p0853.raw`.

Train and generate:

```bash
python scripts/train_fontainebleau.py --stage all \
  --raw_path data/fontainebleau/phi0p2045.raw \
  --raw_shape 480 480 480 \
  --save_dir outputs/fontainebleau_phi0p2045 \
  --poro_center 0.13 \
  --target_porosity 0.2045 \
  --device cuda

python scripts/generate_batch.py \
  --ckpt_dir outputs/fontainebleau_phi0p2045 \
  --out_root data/generated_fontainebleau_sets \
  --targets 0.2045 0.1743 0.1263 0.0853 \
  --n_per_target 50 \
  --poro_center 0.13 \
  --device cuda
```

After `git lfs pull`, use
`--ckpt_dir savedmodels/fontainebleau_phi0p2045`.

## Current Limitations

- Final trained checkpoints are distributed under `savedmodels/` through Git
  LFS and verified by `scripts/verify_final_models.py`. The complete workflow
  can also be reproduced through model retraining. See `docs/checkpoints.md`.
- Manuscript-scale permeability and pore-network results require `porespy`, `openpnm`, and substantial compute time.
- Because training and sampling are stochastic, exact generated volumes and numerically identical outputs are not guaranteed.

## Release GPU Smoke Test

Run these commands in the original GPU environment before tagging the release:

```bash
python scripts/generate_batch.py \
  --ckpt_dir savedmodels/main_sandstone \
  --out_root outputs/release_smoke/main \
  --targets 0.13 \
  --n_per_target 1 \
  --seed_start 0 \
  --device cuda \
  --poro_center 0.13 \
  --poro_scale 0.02 \
  --n_steps 1000

python scripts/generate_batch.py \
  --ckpt_dir savedmodels/fontainebleau_phi0p2045 \
  --out_root outputs/release_smoke/fontainebleau \
  --targets 0.2045 \
  --n_per_target 1 \
  --seed_start 0 \
  --device cuda \
  --poro_center 0.13 \
  --poro_scale 0.02 \
  --n_steps 1000

python scripts/validate_smoke_outputs.py \
  --main-root outputs/release_smoke/main \
  --fontainebleau-root outputs/release_smoke/fontainebleau \
  --output results/release_smoke_validation.json
```

Successful generation demonstrates checkpoint loading, latent-statistics
loading, UNet sampling, VQ-VAE decoding, and NPZ writing. The validation command
then verifies that both outputs contain readable, three-dimensional binary
`seg` arrays with consistent porosity metadata.
