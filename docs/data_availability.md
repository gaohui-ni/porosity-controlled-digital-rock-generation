# Data Availability

This repository separates raw experimental data, external validation data, derived results, and lightweight synthetic examples.

## Primary Laboratory Sandstone Data

The laboratory sandstone micro-CT volume data and associated metadata used for the main experiments are available in Mendeley Data:

https://doi.org/10.17632/vp2yw9c7jj.1

The released dataset contains the binary sandstone digital core volume used for training-sample construction, together with metadata describing the voxel convention and image resolution.

## External Fontainebleau Validation Data

The Fontainebleau sandstone digital rock samples used for validation were obtained from a previously reported Australian National University (ANU) digital rock dataset (Arns et al., 2007; Xiao et al., 2024) and are not redistributed by the authors.

The validation dataset contains four three-dimensional sandstone samples:

- volume size: 480 x 480 x 480 voxels;
- voxel resolution: 5.68 um/voxel;
- porosities: 0.1743, 0.1263, 0.0853, and 0.2045.

See `docs/fontainebleau_protocol.md` for the validation protocol.

## Derived Results

Derived statistical results, figure source tables, and analysis data can be regenerated with the scripts in this repository when the raw data and trained checkpoints are available locally.

Typical derived outputs include:

- `results/fig_s2/`: S2, lineal-path, and EDT curves;
- `results/tables/`: porosity, voxel, pore-network, and permeability tables;
- `results/fig_pnm/`: topology and pore-network summaries;
- `results/summary.json` and `results/results_summary.csv`: compact result manifests.

Derived statistical results supporting the findings of the study are available from the corresponding author upon reasonable request.

## Lightweight Synthetic Examples

The repository includes small synthetic 64^3 examples for testing the code path without downloading restricted or large raw data. These files are not used as manuscript-scale training data.

## Expected Data Convention

Unless otherwise specified, binary volumes follow this convention:

- `0` = solid matrix;
- `1` = pore space;
- raw files are stored as `uint8`;
- default manuscript-scale generated samples use shape `256 256 256`.

## User-Provided Data Layout

```text
data/
  raw/
    S1.raw
  fontainebleau/
    fontainebleau_phi0p2045.raw
```

The scripts accept alternative paths through command-line arguments.

## Manuscript Data Availability Statement

Suggested wording for the manuscript:

> The laboratory sandstone micro-CT volume data and associated metadata used in this study are available in Mendeley Data at https://doi.org/10.17632/vp2yw9c7jj.1. The Fontainebleau sandstone digital rock samples used for validation were obtained from a previously reported Australian National University (ANU) digital rock dataset (Arns et al., 2007; Xiao et al., 2024) and are not redistributed by the authors. Derived statistical results and analysis data supporting the findings of this study are available from the corresponding author upon reasonable request.
