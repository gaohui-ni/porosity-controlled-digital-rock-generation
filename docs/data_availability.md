# Data Availability

This repository separates raw experimental data, external validation data, derived results, and lightweight synthetic examples.

## Primary Laboratory Sandstone Data

The laboratory sandstone micro-CT volume data and associated metadata used for the main experiments are available in Mendeley Data:

https://doi.org/10.17632/vp2yw9c7jj.1

The released dataset contains the binary sandstone digital core volume used for training-sample construction, together with metadata describing the voxel convention and image resolution.

The dataset was published under an earlier working title using the term
“porous geomaterials.” The formal manuscript title is **A
Porosity-Controllable Generative Framework for Three-Dimensional Porous Media
Based on Discrete Latent-Space Diffusion**. When citing the dataset, use its
published title, *Laboratory Sandstone Micro-CT Volume Data for
Porosity-Controllable Generation of Three-Dimensional Porous Geomaterials*,
version 1, and DOI https://doi.org/10.17632/vp2yw9c7jj.1.

## External Fontainebleau Validation Data

The Fontainebleau sandstone digital rock samples used for validation were obtained from a previously reported Australian National University (ANU) digital rock dataset (Arns et al., 2007; Xiao et al., 2024).

The Fontainebleau volumes are third-party data and are not redistributed
because the authors do not hold redistribution rights. Users must obtain the
data from the original data provider. Before analysis, the locally obtained
volumes were standardized to binary `uint8` format using the convention
`0 = solid` and `1 = pore` and resized or cropped, where necessary, to a common
volume size of `480 × 480 × 480` voxels. Original file packaging and
phase-label conventions may vary depending on the version supplied by the data
provider.

The validation dataset contains four three-dimensional sandstone samples:

- volume size: 480 x 480 x 480 voxels;
- voxel resolution: 5.68 um/voxel;
- porosities: 0.1743, 0.1263, 0.0853, and 0.2045.

See `docs/fontainebleau_protocol.md` for the validation protocol.

## Derived Results

Derived statistical results, figure source tables, and analysis data can be
regenerated using the released checkpoints, or by independently retraining the
models from the provided workflows, after the required raw data are available
locally.

Typical derived outputs include:

- `results/fig_s2/`: directional two-point correlation function `S2`, lineal-path, and EDT curves;
- `results/tables/`: porosity, voxel, pore-network, and permeability tables;
- `results/fig_pnm/`: topology and pore-network summaries;
- `results/summary.json` and `results/results_summary.csv`: compact result manifests.

Derived statistical results supporting the findings of the study are available from the corresponding author upon reasonable request.

## Trained Checkpoints

The final trained VQ-VAE and latent-DDPM checkpoints are publicly distributed
under `savedmodels/` through Git LFS. File sizes and SHA256 checksums are listed
in `docs/model_manifest.md`. The repository also provides the complete
main-sandstone training workflow for independently retraining the models from
the released Mendeley Data volume.

## Lightweight Synthetic Examples

The repository includes small synthetic 64^3 examples for testing the code path without downloading restricted or large raw data. These files are not used as manuscript-scale training data.

## Expected Data Convention

Unless otherwise specified, binary volumes follow this convention:

- `0` = solid matrix;
- `1` = pore space;
- raw files are stored as `uint8`;
- primary laboratory sandstone voxel size is `3.5e-6` m/voxel;
- default manuscript-scale generated samples use shape `256 256 256`.

## User-Provided Data Configuration

The released main-sandstone volume and user-obtained Fontainebleau volumes are
kept outside version control. Users declare their local paths in the repository
configuration files. Local Fontainebleau packaging and original phase-label
values are intentionally not prescribed here because they depend on the
version obtained from the original provider.

## Manuscript Data Availability Statement

Suggested wording for the manuscript:

> The laboratory sandstone micro-CT volume data and associated metadata used in
> this study are available in Mendeley Data at
> https://doi.org/10.17632/vp2yw9c7jj.1. The source code, pretrained
> checkpoints, model checksums, and reproduction instructions are publicly
> available at
> https://github.com/gaohui-ni/porosity-controlled-digital-rock-generation.
> The Fontainebleau sandstone digital rock samples used for validation were
> obtained from a previously reported Australian National University (ANU)
> digital rock dataset (Arns et al., 2007; Xiao et al., 2024) and are not
> redistributed by the authors. Derived statistical results and analysis data
> supporting the findings of this study are available from the corresponding
> author upon reasonable request.
