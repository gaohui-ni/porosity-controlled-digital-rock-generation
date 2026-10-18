# Methodological Contributions

This document maps the second-version manuscript claims to the released
implementation. The method combines discrete latent-space diffusion with two
levels of porosity control: FiLM provides a learned response to target
porosity, and TP-AQB enforces the final pore count while limiting changes to
important pore connections.

## Discrete latent-space generation

The 3D VQ-VAE maps each `256 x 256 x 256` binary micro-CT subvolume to a
`32 x 64 x 64 x 64` quantized latent tensor with a 1024-entry codebook. The
DDPM therefore denoises a `64^3` spatial representation instead of the
original `256^3` voxel field. The second-version implementation is in
`scripts/train_256_vqvae_ddpm_spatial_vqaware_v3_film.py`.

The laboratory sandstone volume is partitioned before crops are drawn:

| split | x range | y range | z range | use |
| --- | --- | --- | --- | --- |
| training | `[0,512)` | `[0,800)` | `[0,512)` | model fitting and latent statistics |
| validation | `[0,512)` | `[0,800)` | `[544,800)` | checkpoint selection and calibration |
| testing | `[544,800)` | `[0,800)` | `[0,800)` | final structural and physical evaluation |

The 32-voxel gap prevents overlap between the training and validation regions.
The testing region is excluded from training, model selection, and calibration.

## Learned porosity response through FiLM

The target porosity is normalized and concatenated with the sinusoidal
diffusion-time embedding. A small MLP produces channel-wise affine parameters
for the latent U-Net bottleneck:

```text
u_phi = clip((phi_target - poro_center) / poro_scale)
F_mod = F * (1 + gamma(t, u_phi)) + beta(t, u_phi)
```

The FiLM ablation keeps the network and sampling procedure fixed while setting
the porosity input to zero. Under fixed-threshold binarization, the manuscript
reports a response slope close to zero without the condition and a monotonic
response with a slope of `1.009` and MAE of `0.00482` with the calibrated FiLM
condition. This establishes that the target signal reaches the sampler. It does
not by itself establish exact output porosity.

## Precise porosity matching through TP-AQB

`src/sampling/tp_aqb.py` implements topology-preserving adaptive quantile
binarization. The fixed-threshold volume `P >= 0.5` supplies the generated
structure and specific-surface reference. TP-AQB then adds or removes voxels
until `round(phi_target * N)` pore voxels are present while:

- using 26-connectivity for pore space and 6-connectivity for solid space;
- allowing deletion only at simple points;
- protecting skeleton and narrow-throat neighborhoods;
- preventing interacting deletions within one wave;
- verifying global topology after each wave; and
- using equal-count local pore-solid swaps to reduce specific-surface drift.

This differs from the conventional top-N quantile helper retained in
`src/sampling/quantile_binarization.py`. The helper is useful for demonstrations
and compatibility workflows, but TP-AQB is the method described in the
second-version manuscript.

## Main sandstone evidence

The final main-sandstone comparison contains 95 real samples from the held-out
testing region and 95 generated samples conditioned on the corresponding real
porosities. The nominal porosity groups are:

| target porosity | paired samples |
| ---: | ---: |
| 0.11 | 8 |
| 0.12 | 29 |
| 0.13 | 39 |
| 0.14 | 19 |

The reported evaluation covers directional two-point correlation functions,
specific surface area, pore and throat size, throat shape factor, tortuosity,
throat length, coordination number, directional permeability, and geometric
mean permeability. The claims concern ensemble agreement and porosity trends,
not voxel-wise reconstruction of a particular real crop.

## Fontainebleau extrapolation evidence

The Fontainebleau experiment is trained separately from the laboratory
sandstone model. It uses one `480^3` sandstone volume for training and two
lower-porosity volumes of the same lithology for validation. The reported
whole-volume porosities are `0.2041`, `0.1766`, and `0.1289`; the repository's
legacy filename and target labels are `0.2045`, `0.1743`, and `0.1263`.

Twenty generated and twenty real `256^3` subvolumes are compared at each
condition. The two validation rocks lie below the training-patch porosity range
of `0.1926` to `0.2215`. This supports cross-sample porosity extrapolation
within Fontainebleau sandstone. It is not zero-shot transfer from the main
laboratory sandstone and does not establish universal cross-lithology
generalization.

## Manuscript figure evidence

| Evidence | Manuscript figure |
| --- | --- |
| VQ-VAE, conditional DDPM, and TP-AQB schematics | Figs. 1-3 |
| Generated morphology, FiLM ablation, porosity matching, and slice variation | Figs. 4-7 |
| Main-sandstone spatial statistics, pore-network descriptors, and permeability | Figs. 8-10 |
| Fontainebleau porosity ranges, spatial statistics, pore-network descriptors, and permeability | Figs. 11-14 |

See `docs/figure_mapping.md` for the corresponding scripts and inputs.

## Scope and current release status

- The main model learns from subvolumes of one parent laboratory sandstone
  volume.
- Fontainebleau comparison patches come from three parent volumes and are not
  independent rock specimens.
- TP-AQB enforces pore count and preserves the generated reference topology; it
  does not directly impose real-rock pore-network or permeability targets.
- The current Git LFS checkpoints belong to the earlier compatibility
  workflow. Second-version spatial checkpoints and the frozen 95-crop cohort
  manifest are not included in this revision.
