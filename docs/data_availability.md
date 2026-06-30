# Data Availability

The training data used in the manuscript are available from Mendeley Data:

https://doi.org/10.17632/yp2yw9c7jj

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

> The training data used in this study are available from Mendeley Data at https://doi.org/10.17632/yp2yw9c7jj. The code repository provides the complete model implementation, training and inference scripts, evaluation workflows, and synthetic examples for testing. Users may reproduce the workflow using the published binary micro-CT data or their own binary 3D volumes following the documented `0=solid, 1=pore` raw-volume convention.

If a public source is later available for the Fontainebleau sandstone volume or any other validation data, add the URL and license here and cite it in the manuscript.
