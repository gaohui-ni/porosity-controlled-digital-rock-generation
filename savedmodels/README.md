# savedmodels

This directory contains earlier-workflow compatibility model files publicly
distributed through Git LFS. The second-version spatial checkpoints are not
bundled here.

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

`outputs/` remains the default location for newly trained model outputs.
`savedmodels/` contains released or locally supplied compatibility checkpoints.
