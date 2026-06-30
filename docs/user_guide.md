# User Guide

This guide describes the repository inputs, outputs, and common workflows.

## Inputs

The main workflows expect binary 3D volumes:

- format: `.raw`, `.npy`, or `.npz`;
- default manuscript-scale shape: `256 256 256` for generated and comparison samples;
- raw training volume shape is supplied by `--raw_shape`;
- phase convention: `0=solid`, `1=pore`.

## Outputs

Training scripts produce:

- `vqvae_final.pth`;
- `unet_final.pth`;
- `latent_stats.npz`;
- logs and intermediate checkpoints.

Generation scripts produce:

- compressed `.npz` files containing probability and binary segmentation arrays;
- `.raw` binary segmentations;
- metadata `.csv` and `.json` files;
- optional slice visualizations.

Evaluation scripts produce:

- curve summaries;
- figures;
- table-ready `.csv` outputs.

## Minimal Synthetic Demo

```bash
python scripts/demo_quantile_binarization.py
```

This checks the adaptive binarization step without requiring trained models.

## Build Real Porosity Groups

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

## Train the Main Model

```bash
python scripts/train_256_vqvae_ddpm_lat64_v6_light96_full.py \
  --stage all \
  --raw_path data/raw/S1.raw \
  --save_dir outputs/main_sandstone \
  --target_porosity 0.13 \
  --device cuda
```

## Batch Generation

```bash
python scripts/generate_batch.py \
  --ckpt_dir outputs/main_sandstone \
  --out_root data/generated_phi_sets \
  --targets 0.11 0.12 0.13 0.14 0.15 \
  --n_per_target 100 \
  --poro_center 0.13 \
  --device cuda
```

## Evaluation

```bash
python scripts/evaluate_s2_lineal_edt.py \
  --real_root data/real256_sets_from_S1_strict \
  --gen_root data/generated_phi_sets \
  --out_root results/curves \
  --targets 0.11 0.12 0.13 0.14 0.15

python scripts/evaluate_voxel_and_perm.py \
  --input_root data/generated_phi_sets \
  --output_csv results/tables/generated_voxel_perm.csv \
  --group_name gen \
  --shape 256 256 256 \
  --recursive
```

## Hardware Notes

The synthetic demo is CPU friendly. Full 256^3 training and pore-network/permeability evaluation are GPU- and memory-intensive. Use CUDA for model training when available.
