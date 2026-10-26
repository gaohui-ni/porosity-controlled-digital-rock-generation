# Model Manifest

This manifest records the released second-version spatial package and the
earlier-workflow compatibility files under `savedmodels/`.

The compatibility checkpoints are publicly distributed under `savedmodels/`
through Git LFS. The two `unet_final.pth` files exceed GitHub's normal 100 MB
single-file limit and are stored as LFS objects. The SHA256 values below verify
downloaded file identity.

## Second-version spatial package

The exact file hashes for all 14 release artifacts are recorded in
`savedmodels/spatial_vqaware_v2/MANIFEST.sha256`. The principal model files are:

| File | Size bytes | SHA256 |
| --- | ---: | --- |
| `savedmodels/spatial_vqaware_v2/R03_calibrated/ae_best.pth` | 10510532 | `4ee13a88647be589c318bc811b5ae97b186c9a9cde2ffa23ea6355d8b5695041` |
| `savedmodels/spatial_vqaware_v2/R03_calibrated/calib_A_scale.pth` | 46702970 | `1d5ef756a068fbd5d180b68f6f7f86e7b294c6bd312fb2a09486b679b439ce15` |
| `savedmodels/spatial_vqaware_v2/R03_calibrated/latent_stats.npz` | 1649 | `25df1f39889214817b62a8eb8d6d68c8fc48cbcde7d12cc8320618bf82d5a921` |
| `savedmodels/spatial_vqaware_v2/optional_precalibration/unet_best_noise.pth` | 46703882 | `f3e68a21c843998dd9f6536a4c69fb35a9582b40bac463a80c0090b3d66c8ba9` |

The calibrated and pre-calibration DDPM files contain 68 bitwise-identical
model tensors. They differ only in the saved conditioning scale.

This spatial package is the frozen main-sandstone sampling release. Its file
identity is verified, while exact VQ-VAE training-history reproduction remains
blocked by the 15-versus-80 metadata conflict recorded in
`savedmodels/spatial_vqaware_v2/MANUSCRIPT_PROVENANCE.json`.

## Compatibility File Layout

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
| `savedmodels/main_sandstone/vqvae_final.pth` | 31130266 | `4bb59607415ad3deeb20b1876d3c223fe95a63c6c24fa52afaa246840acfffe4` |
| `savedmodels/main_sandstone/unet_final.pth` | 140116814 | `240203a9f040c27d0bf9a45eff849a501677a24732eb60c64603877dd3073761` |
| `savedmodels/main_sandstone/latent_stats.npz` | 1652 | `042c167cc4377dd1bac1936740ff633a9eb2d87971b6726132cc869778f88d73` |
| `savedmodels/fontainebleau_phi0p2045/vqvae_final.pth` | 31129562 | `de0ca508d6cf431ec68838d8a022077a3ab366bc36983bc0fcd148dbc516dd5d` |
| `savedmodels/fontainebleau_phi0p2045/unet_final.pth` | 140116174 | `b07eb121878509621f015366d801762a875fd360f745e7d47a003e36088986a3` |
| `savedmodels/fontainebleau_phi0p2045/latent_stats.npz` | 1651 | `aa6c495709132e367a06931313c72472e8473fbc21de26a7708ebd99613f1753` |

## Verification

On Windows PowerShell:

```powershell
Get-FileHash -Algorithm SHA256 savedmodels/main_sandstone/unet_final.pth
Get-FileHash -Algorithm SHA256 savedmodels/fontainebleau_phi0p2045/unet_final.pth
```

The `fontainebleau_phi0p2045` files are compatibility checkpoints only. They
must not be used to claim direct reproduction of Fuel Fig. 11. The exact
epoch-75/EXT3X checkpoint, sampler, seed manifest, and source table are not
currently released; see `docs/fontainebleau_provenance_audit.md`.

For checksum and checkpoint-metadata validation:

```bash
python scripts/verify_final_models.py
```
