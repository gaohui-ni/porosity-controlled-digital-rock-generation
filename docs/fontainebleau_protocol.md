# Fontainebleau Validation Protocol

This protocol documents how Fontainebleau sandstone samples are used for independent validation and porosity extrapolation.

## Dataset

The validation data are four Fontainebleau sandstone digital rock samples from a previously reported Australian National University digital rock dataset. The samples are not redistributed in this repository.

- Volume size: 480 x 480 x 480 voxels
- Voxel resolution: 5.68 um/voxel
- Porosities: 0.1743, 0.1263, 0.0853, and 0.2045

## Training and Test Protocol

The standard validation setting trains the model on one Fontainebleau volume and evaluates generation at both the training porosity and unseen porosities.

- Training porosity: 0.2045 by default
- Test porosities: 0.2045, 0.1743, 0.1263, and 0.0853
- Generated samples per target: 50 by default
- Patch size: 256 x 256 x 256 voxels unless otherwise specified

The lower-porosity targets test extrapolation because they are not included as direct training targets in the default setting.

## Commands

Prepare a raw validation volume:

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

## Evaluation Metrics

Use the same metrics as the main sandstone experiment:

- Achieved porosity
- Two-point correlation `S2`
- Lineal-path statistics
- Euclidean distance transform pore-size statistics
- Connectivity and Euler characteristic
- Pore-network descriptors and OpenPNM permeability when dependencies are available
