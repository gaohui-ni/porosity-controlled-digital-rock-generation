# Reproducibility workflow for Computers & Geosciences

This repository is organized for review and reproducibility. The raw micro-CT data and trained checkpoints are not included because of data restrictions and file size. The public repository provides model definitions, training scripts, sampling scripts, metric evaluation scripts, and a synthetic demo.

## 1. Install

```bash
conda env create -f environment.yml
conda activate digitalrock
# or
pip install -r requirements.txt
pip install -r requirements_optional.txt
```

## 2. Minimal demo

```bash
python scripts/demo_quantile_binarization.py
```

## 3. Main sandstone workflow

### Build real comparison groups

```bash
python scripts/build_real_phi_groups.py   --raw_path data/raw/S1.raw   --raw_shape 800 800 800   --patch 256   --stride 32   --targets 0.11 0.12 0.13 0.14 0.15   --n_per_target 100   --out_root data/real256_sets_from_S1_strict
```

### Train VQ-VAE and latent DDPM

```bash
python scripts/train_256_vqvae_ddpm_lat64_v6_light96_full.py --stage all   --raw_path data/raw/S1.raw   --save_dir outputs/main_sandstone   --target_porosity 0.13   --device cuda
```

### Batch generate samples

```bash
python scripts/generate_batch.py   --ckpt_dir outputs/main_sandstone   --out_root data/generated_phi_sets   --targets 0.11 0.12 0.13 0.14 0.15   --n_per_target 100   --poro_center 0.13   --device cuda
```

### Evaluate curves and physical metrics

```bash
python scripts/evaluate_s2_lineal_edt.py   --real_root data/real256_sets_from_S1_strict   --gen_root data/generated_phi_sets   --out_root results/curves   --targets 0.11 0.12 0.13 0.14 0.15

python scripts/evaluate_voxel_and_perm.py   --input_root data/generated_phi_sets   --output_csv results/tables/generated_voxel_perm.csv   --group_name gen   --shape 256 256 256   --recursive

python scripts/evaluate_pore_network_6panel.py

python scripts/evaluate_coordination_euler.py   --real-root data/real256_sets_from_S1_strict   --gen-root data/generated_phi_sets   --out-root results/topology
```

## 4. Fontainebleau workflow

Prepare a 0/1 raw volume first:

```bash
python scripts/prepare_fontainebleau_data.py   --input /path/to/fontainebleau.raw   --output_raw data/fontainebleau/fontainebleau_phi0p2045.raw   --raw_shape 480 480 480   --pore_value 1
```

Train and generate:

```bash
python scripts/train_fontainebleau.py --stage all   --raw_path data/fontainebleau/fontainebleau_phi0p2045.raw   --raw_shape 480 480 480   --save_dir outputs/fontainebleau_phi0p2045   --poro_center 0.2045   --target_porosity 0.2045   --device cuda

python scripts/generate_batch.py   --ckpt_dir outputs/fontainebleau_phi0p2045   --out_root data/generated_fontainebleau_sets   --targets 0.2045 0.1743 0.1263 0.0853   --n_per_target 50   --poro_center 0.2045   --device cuda
```
