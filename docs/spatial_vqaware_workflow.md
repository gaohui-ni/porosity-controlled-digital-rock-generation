# Spatial-split EMA-VQ/DDPM and TP-AQB v2 workflow

## Status and compatibility

This is the primary laboratory-sandstone implementation for the second-version
manuscript. It was added without removing the repository's earlier checkpoint
workflow. The top-level `run_pipeline.py` remains the public entry point, and
the current workflow is selected explicitly with:

```bash
python run_pipeline.py --mode spatial --config configs/main.yaml
```

The released compatibility checkpoints retain their original architecture and
file names. Spatial-workflow checkpoints are written separately under
`outputs/spatial_vqaware/` and are not interchangeable with those files.

## Delivered training run

The frozen second-version package records the run that actually produced the
delivered checkpoints:

| Module | Actual training schedule | Selected checkpoint |
| --- | ---: | ---: |
| 3D VQ-VAE | 15 epochs | epoch 13 (one-based; checkpoint `epoch=12`) |
| Latent DDPM | 40 epochs | epoch 35 (one-based; checkpoint `epoch=34`) |

- best VAE validation loss: `0.019152`;
- best DDPM validation noise loss: `0.052244`;
- VAE elapsed time: `3.03 h`;
- DDPM elapsed time: `1.64 h`;
- total training time: `4.67 h`.

These values come from the frozen training histories and checkpoint metadata.
Some serialized checkpoint `cfg` fields and
`optional_precalibration/run_cfg.json` retain exploratory/default maxima such
as `80` or `150`; they are provenance records, not evidence that the delivered
models were trained for those maxima. The canonical retraining schedule is
`epochs_vae: 15` and `epochs_ddpm: 40` under `spatial_vqaware` in
`configs/main.yaml`. Frozen checkpoint bytes are intentionally unchanged.

## Data isolation

The workflow uses the same binary `800 x 800 x 800` parent sandstone volume
as the manuscript workflow, with `0=solid` and `1=pore`. It enforces the
following half-open spatial regions for `256^3` crops:

| split | x | y | z | role |
|---|---:|---:|---:|---|
| train | `[0,512)` | `[0,800)` | `[0,512)` | VAE/DDPM fitting and latent statistics |
| validation | `[0,512)` | `[0,800)` | `[544,800)` | fixed crops and checkpoint selection |
| test | `[544,800)` | `[0,800)` | `[0,800)` | reserved for post-freeze evaluation |

The training entry point verifies that these regions do not overlap. Test is
not instantiated by the training functions.

## VQ-VAE changes

The spatial workflow retains the `256^3 -> 32 x 64^3 -> 256^3` architecture,
but replaces gradient-trained codebook updates with exponential moving
averages. It adds chunked nearest-code lookup, usage/perplexity diagnostics,
and periodic reset of persistently inactive codes.

The reconstruction objective is:

```text
weighted BCE
+ 0.20 * Dice loss
+ 0.10 * local porosity loss
+ 0.05 * boundary-gradient loss
+ VQ commitment loss
```

The best VAE checkpoint is selected on the fixed validation crops and stored
as `ae_best.pth`.

## Latent DDPM and FiLM

The U-Net and bottleneck FiLM placement remain unchanged. The entry point
supports three controlled variants:

- `continuous`: continuous autoencoder latent;
- `vq`: quantized latent without the optional codebook-consistency term;
- `vqaware`: quantized latent with codebook-manifold consistency.

The delivered configuration in `configs/main.yaml` uses `variant: vq` and
`lambda_codebook: 0.0`. This distinction is intentional: the implementation
supports the vq-aware ablation, but the selected workflow does not claim that
its final DDPM was trained with that auxiliary term.

For comparisons across variants, `unet_best_noise.pth` is selected by
validation noise loss, which is defined identically for every arm. Conditional
sampling uses the frozen weight with `poro_scale=0.011804`; training uses
`poro_scale=0.02`.

## TP-AQB v2

`src/sampling/tp_aqb.py` implements topology- and throat-preserving adaptive
quantile binarization. It starts from the model's own `P >= 0.5` structure,
then adds or removes voxels until the requested pore count is reached while:

- preserving 26-connected pore and 6-connected solid topology;
- deleting only simple points;
- protecting skeleton and narrow-throat neighborhoods;
- preventing interacting deletions within one wave;
- checking global topology after every wave;
- optionally repairing surface drift by equal-count local swaps.

TP-AQB v2 fixes a low-porosity defect in v1: the remaining deletion deficit is
recomputed at every relaxation level and every wave, so the algorithm cannot
reuse a stale deficit and over-delete below the requested pore count.

The frozen recipe is `configs/tp_aqb_v2.yaml`. Apply it independently with:

```bash
python scripts/apply_tp_aqb.py \
  --input-root outputs/spatial_vqaware/condition_sweep \
  --output-root data/generated_phi_sets_spatial \
  --config configs/tp_aqb_v2.yaml
```

Outputs contain the standard `seg` key and can be consumed by the existing
evaluation scripts.

## Checkpoint and output contract

```text
outputs/spatial_vqaware/
|-- ae_best.pth
|-- latent_stats.npz
|-- unet_best_noise.pth
`-- condition_sweep/*.npz

data/generated_phi_sets_spatial/
`-- phi<target>/*_tp_aqb.npz
```

New training runs write the files above under `outputs/`. For final Test-95
sampling without retraining, use the released artifacts under
`savedmodels/spatial_vqaware_v2/R03_calibrated/`. The final DDPM filename is
`calib_A_scale.pth`; its model tensors match the selected
`unet_best_noise.pth`, while its saved conditioning scale is calibrated to
`0.011804`.

## Manuscript evaluation cohort

The manuscript's final main-sandstone evaluation uses 95 real crops from the
held-out testing region and 95 generated samples conditioned on the real
porosity of each crop. The nominal group sizes are `8`, `29`, `39`, and `19`
for target porosities `0.11`, `0.12`, `0.13`, and `0.14`, respectively.

The frozen 100-crop table (95 included in the four manuscript groups), final
seed mapping, pair manifest, matching VQ-VAE, latent statistics, and calibrated
R03 DDPM checkpoint are released under `savedmodels/spatial_vqaware_v2/`.
The 5.9 GB generated probability and binary volumes are not bundled and must be
regenerated locally from the published raw volume.

```bash
git lfs pull
python scripts/generate_spatial_test95.py \
  --raw-path data/raw/Bei_800x800x800.raw \
  --device cuda

python scripts/apply_tp_aqb.py \
  --input-root outputs/spatial_test95/gen \
  --output-root data/generated_phi_sets_spatial_test95 \
  --config configs/tp_aqb_v2.yaml

python scripts/evaluate_spatial_test95.py \
  --manifest outputs/spatial_test95/FINAL_TEST_PAIR_MANIFEST.csv \
  --gen-root outputs/spatial_test95/gen \
  --tpaqb-root data/generated_phi_sets_spatial_test95 \
  --out-root results/spatial_test95
```

`generate_spatial_test95.py` uses the exact crop porosity and released seed for
each of the 95 samples. Its `on` output is the conventional exact top-N
baseline; the second command produces the TP-AQB v2 segmentation used by the
current workflow. The third command validates and joins all branches by
`crop_id`, then stages the canonical `phi0p11`-style directory trees.
