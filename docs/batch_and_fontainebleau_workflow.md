# Batch generation and Fontainebleau validation workflow

## 1. Batch generation for the main sandstone experiment

```bash
python scripts/generate_batch.py \
  --ckpt_dir outputs/main_sandstone \
  --out_root data/generated_phi_sets \
  --targets 0.11 0.12 0.13 0.14 0.15 \
  --n_per_target 100 \
  --poro_center 0.13 \
  --device cuda
```

To generate from the final supplied main-sandstone checkpoint package instead of a newly trained `outputs/` directory, use:

```bash
python scripts/generate_batch.py \
  --ckpt_dir savedmodels/main_sandstone \
  --out_root data/generated_phi_sets \
  --targets 0.11 0.12 0.13 0.14 0.15 \
  --n_per_target 100 \
  --poro_center 0.13 \
  --device cuda
```

The output folder is organized as:

```text
data/generated_phi_sets/
|-- phi0p11/
|-- phi0p12/
|-- phi0p13/
|-- phi0p14/
`-- phi0p15/
```

Each target folder contains `.npz`, `.raw`, `metadata_*.csv`, `metadata_*.json`, and `summary_*.json` files.

## 2. Prepare Fontainebleau data

The Fontainebleau volumes are third-party data and are not redistributed
because the authors do not hold redistribution rights. Users must obtain them
from the original provider. Before analysis, standardize the locally obtained
volumes to binary `uint8` format with `0 = solid`, `1 = pore`, and a common
size of `480 x 480 x 480` voxels. Original file packaging and phase-label
conventions may vary by provider version. Declare the four standardized local
paths in `configs/fontainebleau_config.yaml`.

## 3. Train on Fontainebleau

```bash
python run_pipeline.py --mode fontainebleau --config configs/main.yaml
```

## 4. Generate Fontainebleau validation samples

```bash
python scripts/generate_batch.py \
  --ckpt_dir outputs/fontainebleau_phi0p2045 \
  --out_root data/generated_fontainebleau_sets \
  --targets 0.2045 0.1743 0.1263 0.0853 \
  --n_per_target 50 \
  --poro_center 0.13 \
  --device cuda
```

To generate from the final supplied Fontainebleau checkpoint package, use `--ckpt_dir savedmodels/fontainebleau_phi0p2045`.

The four real validation volumes must be placed at the paths listed in
`configs/fontainebleau_config.yaml`. The official pipeline extracts matched
real patches and evaluates directional `S2`, lineal path, EDT, topology, PNM,
and permeability for both real and generated groups:

```bash
python run_pipeline.py --mode fontainebleau --config configs/main.yaml
```
