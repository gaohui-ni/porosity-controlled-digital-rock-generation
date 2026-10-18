# Spatial-split EMA-VQ/DDPM and TP-AQB v2 workflow

## Status and compatibility

This workflow integrates the post-v0.1.1 experimental code while preserving
the repository's released structure. The top-level `run_pipeline.py` remains
the public entry point. Existing `main`, `fontainebleau`, `full`, and `final`
modes are unchanged; the newer workflow is selected explicitly with:

```bash
python run_pipeline.py --mode spatial --config configs/main.yaml
```

The released v0.1.1 checkpoints retain their original architecture and file
names. Spatial-workflow checkpoints are written separately under
`outputs/spatial_vqaware/` and are not interchangeable with those files.

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
`-- phi_<target>/*_tp_aqb.npz
```

No new trained checkpoint is bundled by this code update. Reproducing the
spatial workflow therefore requires running its VAE, latent-statistics, and
DDPM stages before sampling.
