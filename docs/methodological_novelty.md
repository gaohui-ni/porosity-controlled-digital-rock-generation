# Methodological Contributions

This document maps the methodological claims in the manuscript to the released
implementation and evaluation workflow. The principal novelty lies in a
dual-control discrete latent-space framework that couples learned
porosity-dependent structural generation with explicit output-space porosity
enforcement for physically evaluated `256^3` digital-rock generation under
limited-sample conditions. It does not claim that VQ-VAE, FiLM, DDPM, or
quantile thresholding is independently a new algorithm.

## 1. Computational Problem

Generating a `256 x 256 x 256` binary digital rock is demanding because the
model must represent three-dimensional pore geometry, generate stochastic
structural variation, respond continuously to a prescribed porosity, and
produce a binary volume suitable for morphological and transport analysis.
Matching only the mean pore fraction is insufficient: connectivity,
two-point correlation, pore and throat geometry, coordination, and
permeability must also remain geologically meaningful.

In the reported setting, high-resolution physical volumes are limited and
direct diffusion over `256^3` voxels would carry substantial memory and
computational cost. Learned conditioning alone also does not mathematically
guarantee that a binarized output will attain a prescribed global pore
fraction. The framework therefore combines spatial compression, learned
continuous conditioning, and explicit output projection in one control path.

The framework addresses these requirements with three coupled stages:

1. a 3D VQ-VAE learns a compact discrete latent representation;
2. a latent DDPM generates normalized quantized embeddings under a continuous
   target-porosity condition;
3. a deterministic quantile-based projection converts the decoded probability
   field into a binary volume with the requested global pore count.

These stages correspond to manuscript Figs. 1, 2, and 3, respectively.

## 2. Discrete Latent-Space Diffusion Under Limited Data

A lightweight 3D VQ-VAE maps each `256 x 256 x 256` binary micro-CT
subvolume to a `32 x 64 x 64 x 64` vector-quantized latent tensor. Diffusion
therefore operates in a `64^3` latent spatial domain rather than directly in
the original `256^3` voxel domain. The finite 1024-entry codebook constrains
the representation to a learned set of reusable embeddings.

In this application, the design reduces the spatial cost of three-dimensional
diffusion and provides a discrete representational constraint for learning
from subvolumes of a parent sandstone sample. These are architectural roles,
not claims that a separate ablation has proved the codebook alone to be the
cause of improved stability.

The VQ-VAE architecture and quantizer are implemented in
`src/models/vqvae3d.py`. Latent-statistics estimation and the full training
path are implemented in
`scripts/train_256_vqvae_ddpm_lat64_v6_light96_full.py`.

## 3. Dual Porosity-Control Mechanism

The framework separates learned porosity-dependent generation from exact
global pore-count enforcement. The first level conditions latent denoising;
the second level projects the decoded probability field to the requested
porosity.

### 3.1 Learned Porosity-Dependent Structural Control

For each training patch, porosity is measured from its binary voxel field. The
condition supplied to the denoiser is normalized consistently during training
and inference:

```text
phi_scaled = (phi_target - poro_center) / poro_scale
```

The manuscript configuration uses `poro_center = 0.13` and
`poro_scale = 0.02`, as recorded in `configs/main.yaml`. The normalized scalar
is concatenated with the sinusoidal diffusion-time embedding. An MLP maps this
joint condition to channel-wise `gamma` and `beta` parameters. The modulation
is applied at the latent 3D U-Net bottleneck immediately before self-attention:

```text
mid = attention(x3 * (1 + gamma) + beta)
```

Thus, porosity is not appended as another voxel channel. It modulates latent
denoising features jointly with diffusion time. The implementation is in
`src/models/unet3d_film.py`, and the same normalization is used by the training
and sampling paths in
`scripts/train_256_vqvae_ddpm_lat64_v6_light96_full.py` and
`scripts/generate_batch.py`.

The contribution claimed here is the adaptation of continuous FiLM
conditioning to this 3D discrete latent-rock workflow and its coupling to the
output-space porosity constraint. In the absence of a dedicated conditioning
ablation, the repository does not claim that FiLM is universally superior to
all alternative conditioning mechanisms.

### 3.2 Exact Output-Space Porosity Constraint

Quantile binarization is accurately described as an inference-time output
projection, not as a replacement for the generative model. Given a decoded
pore-probability volume `P` with `N` voxels and target porosity `phi_target`, it
computes

```text
N_target = round(phi_target * N)
```

and uses a partition-based quantile threshold to select the highest-probability
voxels as pore space. Voxels tied at the threshold are selected with a seeded
tie-breaking step so that the final pore count matches `N_target` up to the
integer discretization implied by the finite volume.

This operation imposes the global pore fraction without retraining while
preserving the rank ordering of the decoded probability field. It does not, by
itself, prove that pore geometry or transport behavior is correct. Those
properties are evaluated separately. The implementation is in
`src/sampling/quantile_binarization.py`; the lightweight executable example is
`scripts/demo_quantile_binarization.py`; and the target-versus-achieved
porosity behavior is covered by `tests/test_quantile_binarization.py` and
`tests/test_porosity.py`.

## 4. Distinction From Existing 3D Generative Workflows

The methodological advance lies in the complete control architecture:

- compared with direct 3D VAE sampling, stochastic generation is performed
  through diffusion in a vector-quantized latent representation;
- compared with GAN-based generation, the model uses a denoising objective
  without adversarial optimization;
- compared with voxel-space 3D DDPMs, diffusion operates in a spatially
  compressed `64^3` latent domain rather than the `256^3` binary domain;
- compared with unconditional or discrete class-conditional generation,
  porosity is a continuous physical condition during training and sampling;
- compared with purely learned conditioning, output-space projection enforces
  the prescribed global pore count; and
- compared with fixed-threshold decoding, the threshold adapts to the requested
  porosity and the decoded probability ranking.

The framework should therefore be understood as a unified physical-control
architecture, not as a claim that its established components were individually
invented in this study.

## 5. Geoscientific Evidence Chain

The validation strategy asks whether porosity control is accompanied by
reasonable structural and transport behavior rather than treating porosity
agreement as sufficient evidence. The repository maps the manuscript evidence
as follows:

| Evidence | Scientific role | Implementation | Manuscript mapping |
| --- | --- | --- | --- |
| Generated volumes and slices | Qualitative 3D and sectional morphology | `scripts/generate_batch.py`, `scripts/plot_slice_porosity.py` | Figs. 4-5 |
| Directional two-point correlation function `S2` | Direction-dependent spatial organization | `scripts/evaluate_s2_lineal_edt.py` | Fig. 6 |
| Pore, throat, EDT, shape-factor, and coordination distributions | Pore-scale geometry and topology | `scripts/evaluate_pore_network_6panel.py`, `scripts/evaluate_coordination_euler.py` | Fig. 7 |
| Directional and geometric-mean permeability | Transport response and anisotropy | `scripts/evaluate_voxel_and_perm.py`, `scripts/compare_metric_tables.py` | Fig. 8 |
| Independent Fontainebleau validation | A separately trained Fontainebleau experiment evaluating porosity extrapolation from the `0.2045` training volume to three unseen porosity conditions | `python run_pipeline.py --mode fontainebleau --config configs/main.yaml` | Fig. 9 |

The complete mapping from figures to commands, inputs, and outputs is provided
in `docs/figure_mapping.md` and `docs/figure_reproduction.md`.

## 6. Out-of-Range Conditional Generation

The experiments investigate whether local structural variation extracted from
a parent sandstone volume can support generation across prescribed porosity
conditions. Porosity extrapolation is examined in a separately trained
Fontainebleau experiment using one training volume and three independently
obtained volumes at unseen porosity conditions, with evaluation extending from
voxel morphology to pore-network descriptors and permeability.

The purpose is to test whether the learned conditional relationship among
porosity, pore structure, and transport response can be extrapolated beyond
the training porosity within the examined Fontainebleau sandstone domain.
This evidence supports porosity extrapolation within the separately trained
experiment and tested porosity interval. It is not a zero-shot transfer of the
main-sandstone model and does not establish universal generalization across
lithologies. Agreement in porosity is interpreted together with spatial,
topological, and transport metrics, not as a standalone guarantee of physical
fidelity.

## 7. Scope and Limitations

- The individual building blocks are established methods; the contribution is
  their task-specific integration and validation.
- The main model learns from subvolumes of one parent laboratory sandstone
  volume, so conclusions concern the represented sandstone structure rather
  than all porous geomaterials.
- Fontainebleau comparison subvolumes overlap within each parent volume and are
  not independent rock specimens.
- The external validation covers four Fontainebleau volumes and does not imply
  universal cross-lithology generalization.
- Quantile projection enforces global porosity but cannot independently ensure
  connectivity, topology, or permeability; these require the reported
  evaluations.
- No new ablation result is asserted in this document. All claims are limited
  to the released implementation and the experiments reported in the
  manuscript.
