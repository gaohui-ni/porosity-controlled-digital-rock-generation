# Checkpoint Policy

The final trained checkpoints are publicly distributed under `savedmodels/`
through Git LFS. File sizes and SHA256 checksums are listed in
`docs/model_manifest.md`.

The expected final model layout is:

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

The repository also provides the complete source code, configuration files, public raw-data reference, random-seed settings, and documentation required to retrain the 3D VQ-VAE and the FiLM-conditioned latent DDPM. New training runs write checkpoints under `outputs/` by default:

- `outputs/main_sandstone/vqvae_final.pth`
- `outputs/main_sandstone/unet_final.pth`
- `outputs/main_sandstone/latent_stats.npz`
- `outputs/fontainebleau_phi0p2045/vqvae_final.pth`
- `outputs/fontainebleau_phi0p2045/unet_final.pth`
- `outputs/fontainebleau_phi0p2045/latent_stats.npz`

Using the supplied final checkpoints supports exact checkpoint-level reuse for generation. Independent retraining is expected to reproduce the methodology, porosity-control behavior, and statistical trends reported in the manuscript, but may not reproduce identical generated volumes because neural-network training and diffusion sampling are stochastic.
