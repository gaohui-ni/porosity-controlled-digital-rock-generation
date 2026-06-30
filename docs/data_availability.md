# Data Availability

The laboratory sandstone micro-CT volume data and associated metadata used in this study are available in Mendeley Data:

https://doi.org/10.17632/vp2yw9c7jj.1

The Fontainebleau sandstone digital rock samples used for validation were obtained from a previously reported Australian National University (ANU) digital rock dataset (Arns et al., 2007; Xiao et al., 2024) and are not redistributed by the authors. This dataset contains four three-dimensional sandstone samples, each with a size of 480 x 480 x 480 voxels and a spatial resolution of 5.68 um/voxel. The corresponding porosities are 0.1743, 0.1263, 0.0853, and 0.2045. Derived statistical results and analysis data supporting the findings of this study are available from the corresponding author upon reasonable request.

This repository therefore provides:

- scripts for preparing user-provided binary raw volumes;
- scripts for constructing real comparison groups from a large raw volume;
- synthetic 64^3 examples for lightweight testing;
- placeholders documenting where raw data, generated samples, checkpoints, and results should be placed.

## Expected Data Convention

Unless otherwise specified, binary volumes follow this convention:

- `0` = solid matrix;
- `1` = pore space;
- raw files are stored as `uint8`;
- default manuscript-scale samples use shape `256 256 256`.

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
