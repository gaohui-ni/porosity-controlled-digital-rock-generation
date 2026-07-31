# savedmodels

This directory contains the final trained model files publicly distributed
through Git LFS.

Expected layout:

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

`outputs/` remains the default location for newly trained model outputs. `savedmodels/` is reserved for final released or locally supplied checkpoints.
