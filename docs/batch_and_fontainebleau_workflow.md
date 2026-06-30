# Batch generation and Fontainebleau validation workflow

## 1. Batch generation for the main sandstone experiment

```bash
python scripts/generate_batch.py \
  --ckpt_dir outputs/main_sandstone \
  --out_root generated_phi_sets \
  --targets 0.11 0.12 0.13 0.14 0.15 \
  --n_per_target 100 \
  --poro_center 0.13 \
  --device cuda
```

The output folder is organized as:

```text
generated_phi_sets/
├── phi0p11/
├── phi0p12/
├── phi0p13/
├── phi0p14/
└── phi0p15/
```

Each target folder contains `.npz`, `.raw`, `metadata_*.csv`, `metadata_*.json`, and `summary_*.json` files.

## 2. Prepare Fontainebleau data

The input volume should be converted to uint8 raw format where `0=solid` and `1=pore`.

```bash
python scripts/prepare_fontainebleau_data.py \
  --input /path/to/fontainebleau_volume.raw \
  --output_raw data/fontainebleau/fontainebleau_phi0p2045.raw \
  --raw_shape 480 480 480 \
  --pore_value 1
```

If the input semantics are reversed, add `--invert`. If the input is grayscale or probability-like, use `--threshold`.

## 3. Train on Fontainebleau

```bash
python scripts/train_fontainebleau.py \
  --stage all \
  --raw_path data/fontainebleau/fontainebleau_phi0p2045.raw \
  --raw_shape 480 480 480 \
  --save_dir outputs/fontainebleau_phi0p2045 \
  --poro_center 0.2045 \
  --target_porosity 0.2045 \
  --device cuda
```

## 4. Generate Fontainebleau validation samples

```bash
python scripts/generate_batch.py \
  --ckpt_dir outputs/fontainebleau_phi0p2045 \
  --out_root generated_fontainebleau_sets \
  --targets 0.2045 0.1743 0.1263 0.0853 \
  --n_per_target 50 \
  --poro_center 0.2045 \
  --device cuda
```

The generated samples can then be evaluated using the same S2, pore-network, and OpenPNM permeability scripts used for the main sandstone experiment.
