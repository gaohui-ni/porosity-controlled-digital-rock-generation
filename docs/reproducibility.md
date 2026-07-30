# Reproducibility

This repository is organized to support review of the computational workflow for the associated Computers & Geosciences manuscript.

## Reproducibility Levels

This repository supports the following levels of reproducibility:

1. **Lightweight functional reproduction**: run the synthetic demonstration and unit tests without raw micro-CT data or trained checkpoints.
2. **Full workflow reproduction through retraining**: use the public raw data, supplied configuration files, and training scripts to retrain the VQ-VAE and latent DDPM, generate porosity-controlled samples, and rerun the evaluation workflow.
3. **Statistical manuscript-result reproduction**: independently retrained models are expected to reproduce the reported porosity-control accuracy, structural statistics, pore-network characteristics, and permeability trends within normal stochastic variation.

The final trained checkpoints used to produce the manuscript figures are not distributed. Therefore, exact sample-level, bitwise, or numerically identical reproduction of the published generated volumes is not supported.

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
- `full`: `main` followed by `fontainebleau`.

## Main Workflow

The official full pipeline is configured in `configs/main.yaml` and can be launched with:

```bash
python run_pipeline.py --mode full --config configs/main.yaml
```

To reproduce only the main laboratory sandstone results:

```bash
python run_pipeline.py --mode main --config configs/main.yaml
```

The equivalent step-by-step commands are:

```bash
python scripts/build_real_phi_groups.py \
  --raw_path data/raw/S1.raw \
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
  --raw_path data/raw/S1.raw \
  --save_dir outputs/main_sandstone \
  --target_porosity 0.13 \
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

## Evaluation Workflow

Directional two-point probability function `S2`, lineal-path statistics, and EDT pore-size statistics:

```bash
python scripts/evaluate_s2_lineal_edt.py \
  --real_root data/real256_sets_from_S1_strict \
  --gen_root data/generated_phi_sets \
  --out_root results/curves \
  --targets 0.11 0.12 0.13 0.14 0.15
```

Voxel metrics and permeability:

```bash
python scripts/evaluate_voxel_and_perm.py \
  --input_root data/generated_phi_sets \
  --output_csv results/tables/generated_voxel_perm.csv \
  --group_name gen \
  --shape 256 256 256 \
  --voxel_size 3.5e-6 \
  --recursive
```

Topology:

```bash
python scripts/evaluate_coordination_euler.py \
  --real-root data/real256_sets_from_S1_strict \
  --gen-root data/generated_phi_sets \
  --out-root results/topology
```

Pore-network six-panel analysis:

```bash
python scripts/evaluate_pore_network_6panel.py --config configs/main.yaml
```

Before running `evaluate_pore_network_6panel.py`, review the configuration block at the top of the script and adjust paths, target folders, and sample limits. The manuscript-scale voxel size is read from `configs/main.yaml`.

Summarize generated result files:

```bash
python src/metrics/summarize_all.py --root results
```

For a figure-oriented mapping from manuscript result groups to commands and output folders, see `docs/figure_reproduction.md`.

## Fontainebleau Workflow

The Fontainebleau validation can be launched through the official entry point after the prepared raw volume exists at the path configured in `configs/fontainebleau_config.yaml`:

```bash
python run_pipeline.py --mode fontainebleau --config configs/main.yaml
```

Prepare the volume:

```bash
python scripts/prepare_fontainebleau_data.py \
  --input /path/to/fontainebleau.raw \
  --output_raw data/fontainebleau/fontainebleau_phi0p2045.raw \
  --raw_shape 480 480 480 \
  --pore_value 1
```

Train and generate:

```bash
python scripts/train_fontainebleau.py --stage all \
  --raw_path data/fontainebleau/fontainebleau_phi0p2045.raw \
  --raw_shape 480 480 480 \
  --save_dir outputs/fontainebleau_phi0p2045 \
  --poro_center 0.2045 \
  --target_porosity 0.2045 \
  --device cuda

python scripts/generate_batch.py \
  --ckpt_dir outputs/fontainebleau_phi0p2045 \
  --out_root data/generated_fontainebleau_sets \
  --targets 0.2045 0.1743 0.1263 0.0853 \
  --n_per_target 50 \
  --poro_center 0.2045 \
  --device cuda
```

## Current Limitations

- Trained checkpoints are not distributed with this release. The complete workflow can be reproduced through model retraining using the released code, configurations, and public raw data. See `docs/checkpoints.md`.
- Manuscript-scale permeability and pore-network results require `porespy`, `openpnm`, and substantial compute time.
- Because training and sampling are stochastic, exact generated volumes and numerically identical outputs are not guaranteed.
