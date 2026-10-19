# Checkpoint Policy

The files publicly distributed under `savedmodels/` include the second-version
spatial final-sampling package and compatibility checkpoints from the earlier
workflow. File sizes and SHA256 checksums are listed in
`docs/model_manifest.md` and the spatial package's `MANIFEST.sha256`.

The model layout is:

```text
savedmodels/
|-- spatial_vqaware_v2/
|   |-- R03_calibrated/
|   |   |-- ae_best.pth
|   |   |-- calib_A_scale.pth
|   |   `-- latent_stats.npz
|   |-- optional_precalibration/unet_best_noise.pth
|   |-- test_cohort_v1/test_crop_porosity_all_100.csv
|   `-- final_test_v2/
|       |-- final_test_seed_manifest.csv
|       `-- FINAL_TEST_PAIR_MANIFEST.csv
|-- main_sandstone/
|   |-- vqvae_final.pth
|   |-- unet_final.pth
|   `-- latent_stats.npz
`-- fontainebleau_phi0p2045/
    |-- vqvae_final.pth
    |-- unet_final.pth
    `-- latent_stats.npz
```

All `.pth` and `.npz` files below `savedmodels/` are stored as Git LFS objects
rather than ordinary GitHub blobs.

Git LFS under `savedmodels/` is the canonical distribution route for these
files. The same files are not duplicated as GitHub Release assets;
users should run `git lfs pull` after cloning. Identity is verified against the
sizes and SHA256 checksums in `docs/model_manifest.md`.

The repository also provides the source code, configuration files, public
raw-data reference, random-seed settings, and documentation required to train
the second-version spatial 3D VQ-VAE and FiLM-conditioned latent DDPM and to
apply TP-AQB. New compatibility-path training runs write checkpoints under
`outputs/` by default:

- `outputs/main_sandstone/vqvae_final.pth`
- `outputs/main_sandstone/unet_final.pth`
- `outputs/main_sandstone/latent_stats.npz`
- `outputs/fontainebleau_phi0p2045/vqvae_final.pth`
- `outputs/fontainebleau_phi0p2045/unet_final.pth`
- `outputs/fontainebleau_phi0p2045/latent_stats.npz`

The final manuscript sampling run uses `R03_calibrated/ae_best.pth`,
`R03_calibrated/latent_stats.npz`, and
`R03_calibrated/calib_A_scale.pth`. The last file contains model tensors
bitwise identical to the optional pre-calibration `unet_best_noise.pth`, but
records the calibrated `poro_scale=0.011804` instead of `0.02`. The frozen crop,
seed, and pair manifests permit exact cohort reconstruction from the published
raw volume. Generated volumes are omitted because they occupy about 5.9 GB.
