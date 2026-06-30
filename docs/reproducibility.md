# Reproducibility

This repository is organized to support review of the computational workflow for the associated Computers & Geosciences manuscript.

## Reproducibility Levels

Because trained checkpoints are not redistributed, the repository supports three levels of reproducibility:

1. **Lightweight functional test**: run the synthetic quantile-binarization demo.
2. **Workflow reproduction with published or user-provided data**: run data preparation, training, generation, and evaluation on a binary raw volume from the finalized public data DOI cited in the manuscript or another user-supplied source.
3. **Manuscript-scale reproduction**: reproduce the reported figures and tables when the same raw data, trained checkpoints, and random seeds are available locally.

## Lightweight Functional Test

```bash
python scripts/demo_quantile_binarization.py
```

This validates the porosity-matching step and writes example arrays under `examples/`.

## Main Workflow

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

Spatial statistics:

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
python scripts/evaluate_pore_network_6panel.py
```

Before running `evaluate_pore_network_6panel.py`, review the configuration block at the top of the script and adjust paths, target folders, voxel size, and sample limits.

## Fontainebleau Workflow

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

- Large trained checkpoints are not included.
- Manuscript-scale permeability and pore-network results require `porespy`, `openpnm`, and substantial compute time.
- Some exact manuscript figures require the same raw data, checkpoints, and random seeds used in the manuscript experiments.
