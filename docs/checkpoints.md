# Checkpoint Policy

Large trained model checkpoints are not distributed with this repository.

The repository provides the complete source code, configuration files, public raw-data reference, random-seed settings, and documentation required to retrain the 3D VQ-VAE and the FiLM-conditioned latent DDPM.

After retraining, the workflow produces the following files:

- `outputs/main_sandstone/vqvae_final.pth`
- `outputs/main_sandstone/unet_final.pth`
- `outputs/main_sandstone/latent_stats.npz`
- `outputs/fontainebleau_phi0p2045/vqvae_final.pth`
- `outputs/fontainebleau_phi0p2045/unet_final.pth`
- `outputs/fontainebleau_phi0p2045/latent_stats.npz`

Because neural-network training and diffusion sampling are stochastic, independently retrained models are expected to reproduce the methodology, porosity-control behavior, and statistical trends reported in the manuscript, but they may not reproduce the exact generated volumes or numerically identical figure values.

Exact sample-level or bitwise reproduction of the manuscript outputs is therefore not supported in this release.
