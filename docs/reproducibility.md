# Reproducibility

This repository is organized to support review of the computational workflow for the associated Computers & Geosciences manuscript.

## Reproducibility Levels

This repository supports the following levels of reproducibility:

1. **Lightweight functional reproduction**: run the synthetic demonstration and unit tests without raw micro-CT data or trained checkpoints.
2. **Main-sandstone workflow reproduction through retraining**: use the public Mendeley Data volume, supplied configuration files, and training scripts to retrain the VQ-VAE and latent DDPM, generate porosity-controlled samples, and rerun the main-sandstone evaluation workflow.
3. **Statistical manuscript-result reproduction**: independently retrained models are expected to reproduce the reported porosity-control accuracy, structural statistics, pore-network characteristics, and permeability trends within normal stochastic variation.

Compatibility checkpoints are distributed under `savedmodels/` through Git
LFS. The second-version spatial checkpoints and frozen 95-crop coordinate
manifest are not bundled. The code reproduces the training, calibrated
sampling, and TP-AQB workflow, but exact recreation of the reported paired
cohort requires those artifacts. Bitwise-identical generated volumes are not
guaranteed.

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
python run_pipeline.py --mode spatial --config configs/main.yaml --dry-run
```

Available pipeline modes are:

- `demo`: lightweight synthetic porosity-control check;
- `spatial`: primary second-version main-sandstone workflow with spatially
  disjoint splits, calibrated FiLM sampling, and TP-AQB;
- `fontainebleau`: independent Fontainebleau validation workflow;
- `main`, `full`, and `final`: earlier-workflow compatibility paths.

## Main Workflow

The primary second-version main-sandstone workflow is configured in
`configs/main.yaml`:

```bash
python run_pipeline.py --mode spatial --config configs/main.yaml
```

This uses `x=[0,512), z=[0,512)` for training, `x=[0,512), z=[544,800)`
for validation, and `x=[544,800)` for testing, with a 32-voxel exclusion gap.
The reported evaluation compares 95 held-out real crops with 95 corresponding
generated samples across target groups 0.11–0.14.

Run the separate Fontainebleau experiment after its three external volumes are
available:

```bash
python run_pipeline.py --mode fontainebleau --config configs/main.yaml
```

The earlier commands remain available for checkpoint compatibility:

```bash
python run_pipeline.py --mode final --config configs/main.yaml
```

The following step-by-step commands document that compatibility path. Their
training schedule is intentionally left unchanged pending separate
confirmation and does not define the second-version spatial experiment:

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
after the three user-obtained, standardized volumes exist at the paths declared
in `configs/fontainebleau_config.yaml`. These external third-party ANU
Fontainebleau volumes are not redistributed because the authors do not hold
redistribution rights. Users
must obtain them from the original data provider and standardize them to binary
`uint8` volumes with `0 = solid`, `1 = pore`, and a common size of
`480 x 480 x 480` voxels:

```bash
python run_pipeline.py --mode fontainebleau --config configs/main.yaml
```

Original file packaging and phase-label conventions may vary with the version
supplied by the provider. Local paths are therefore declared in the
configuration rather than prescribed in the public documentation. Train and
generate through the official entry point shown above, or use the released
checkpoints with `--mode final`.

An equivalent generation-only command is:

```bash
python scripts/generate_batch.py \
  --ckpt_dir outputs/fontainebleau_phi0p2045 \
  --out_root data/generated_fontainebleau_sets \
  --targets 0.2045 0.1743 0.1263 \
  --n_per_target 20 \
  --poro_center 0.13 \
  --device cuda
```

After `git lfs pull`, use
`--ckpt_dir savedmodels/fontainebleau_phi0p2045`.

## Current Limitations

- Compatibility checkpoints are distributed under `savedmodels/` through Git
  LFS and verified by `scripts/verify_final_models.py`. See
  `docs/checkpoints.md` for their scope.
- The second-version spatial checkpoints and frozen 95-crop coordinate manifest
  are currently absent; exact cohort-level reproduction needs those files.
- Manuscript-scale permeability and pore-network results require `porespy`, `openpnm`, and substantial compute time.
- Because training and sampling are stochastic, exact generated volumes and numerically identical outputs are not guaranteed.

## Release GPU Smoke Test Status

The released compatibility main-sandstone and Fontainebleau checkpoints were each loaded on
GPU and used to generate one binary sample. Both outputs contained readable
three-dimensional `seg` arrays with consistent porosity metadata. The
main-sandstone sample also entered the post-processing smoke workflow
successfully.

The commands used for checkpoint generation and output validation were:

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
