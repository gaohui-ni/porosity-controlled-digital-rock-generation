# Final Model Manifest

This manifest records the final trained model files currently arranged under `savedmodels/`.

The two `unet_final.pth` files exceed the normal GitHub 100 MB single-file limit. If these files are published through GitHub, they should be tracked with Git LFS or uploaded as release assets. The SHA256 values below can be used to verify the copied or downloaded files.

## File Layout

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

## Checksums

| File | Size bytes | SHA256 |
| --- | ---: | --- |
| `savedmodels/main_sandstone/vqvae_final.pth` | 31129562 | `de0ca508d6cf431ec68838d8a022077a3ab366bc36983bc0fcd148dbc516dd5d` |
| `savedmodels/main_sandstone/unet_final.pth` | 140116174 | `b07eb121878509621f015366d801762a875fd360f745e7d47a003e36088986a3` |
| `savedmodels/main_sandstone/latent_stats.npz` | 1651 | `aa6c495709132e367a06931313c72472e8473fbc21de26a7708ebd99613f1753` |
| `savedmodels/fontainebleau_phi0p2045/vqvae_final.pth` | 31130266 | `4bb59607415ad3deeb20b1876d3c223fe95a63c6c24fa52afaa246840acfffe4` |
| `savedmodels/fontainebleau_phi0p2045/unet_final.pth` | 140116814 | `240203a9f040c27d0bf9a45eff849a501677a24732eb60c64603877dd3073761` |
| `savedmodels/fontainebleau_phi0p2045/latent_stats.npz` | 1652 | `042c167cc4377dd1bac1936740ff633a9eb2d87971b6726132cc869778f88d73` |

## Verification

On Windows PowerShell:

```powershell
Get-FileHash -Algorithm SHA256 savedmodels/main_sandstone/unet_final.pth
Get-FileHash -Algorithm SHA256 savedmodels/fontainebleau_phi0p2045/unet_final.pth
```
