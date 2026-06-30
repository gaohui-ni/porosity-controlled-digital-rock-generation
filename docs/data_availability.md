# Data Availability

The laboratory sandstone micro-CT volume data and associated metadata used in this study are available in Mendeley Data:

https://doi.org/10.17632/yp2yw9c7jj.1

The Fontainebleau sandstone digital rock samples used for validation were obtained from a previously reported Australian National University (ANU) digital rock dataset and are not redistributed by the authors. Derived statistical results and analysis data supporting the findings of this study are available from the corresponding author upon reasonable request.

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

> The laboratory sandstone micro-CT volume data and associated metadata used in this study are available in Mendeley Data at https://doi.org/10.17632/yp2yw9c7jj.1. The Fontainebleau sandstone digital rock samples used for validation were obtained from a previously reported Australian National University (ANU) digital rock dataset and are not redistributed by the authors. Derived statistical results and analysis data supporting the findings of this study are available from the corresponding author upon reasonable request.
