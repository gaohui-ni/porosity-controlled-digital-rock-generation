# Checkpoint Policy

The files publicly distributed under `savedmodels/` through Git LFS are
compatibility checkpoints from the earlier workflow. File sizes and SHA256
checksums are listed in
`docs/model_manifest.md`.

The compatibility model layout is:

```text
savedmodels/
|-- main_sandstone/
|   |-- vqvae_final.pth
|   |-- unet_final.pth
|   `-- latent_stats.npz
`-- fontainebleau_phi0p2045/
    |-- vqvae_final.pth
    |-- unet_final.pth
    `-- latent_stats.npz
```

The two `unet_final.pth` files are larger than 100 MB and are therefore stored
as Git LFS objects rather than ordinary GitHub blobs.

Git LFS under `savedmodels/` is the canonical distribution route for these
compatibility files. The same files are not duplicated as GitHub Release assets;
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

The manuscript's second-version spatial workflow uses `ae_best.pth`,
`unet_best_noise.pth`, calibrated sampling outputs, and a frozen 95-crop test
cohort. Those files are not currently bundled. Therefore, the Git LFS files
support exact checkpoint-level reuse only for the compatibility path, not exact
reproduction of the reported second-version cohort. Independent retraining is
expected to reproduce the methodology and statistical behavior rather than
bitwise-identical samples.
