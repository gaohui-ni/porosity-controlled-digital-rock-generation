# savedmodels

This directory contains the second-version spatial final-sampling package and
the earlier-workflow compatibility model files. Model and latent-statistics
files are distributed through Git LFS.

Expected layout:

```text
savedmodels/
|-- spatial_vqaware_v2/
|   |-- R03_calibrated/
|   |-- optional_precalibration/
|   |-- test_cohort_v1/
|   `-- final_test_v2/
|-- main_sandstone/
|   |-- vqvae_final.pth
|   |-- unet_final.pth
|   `-- latent_stats.npz
`-- fontainebleau_phi0p2045/
    |-- vqvae_final.pth
    |-- unet_final.pth
    `-- latent_stats.npz
```

`spatial_vqaware_v2/` contains the calibrated checkpoint and frozen Test-95
provenance used by the second-version manuscript. `main_sandstone/` and
`fontainebleau_phi0p2045/` retain the earlier compatibility checkpoints.
`outputs/` remains the default location for newly trained model outputs.
