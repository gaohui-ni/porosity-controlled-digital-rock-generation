# Checkpoint Availability

Large trained model checkpoints are not committed to this GitHub repository. They should be distributed through an external archival location such as Mendeley Data, Zenodo, or a GitHub Release.

## Required Files for Exact Figure Reproduction

The main manuscript workflow expects the following files under the checkpoint directory configured as `outputs/main_sandstone/`:

```text
outputs/main_sandstone/
  vqvae_final.pth
  unet_final.pth
  latent_stats.npz
```

The Fontainebleau validation workflow expects the corresponding files under:

```text
outputs/fontainebleau_phi0p2045/
  vqvae_final.pth
  unet_final.pth
  latent_stats.npz
```

## Archive Manifest

Fill this table before manuscript submission after uploading the final checkpoint archive.

| File | External URL / DOI | SHA256 |
| --- | --- | --- |
| `outputs/main_sandstone/vqvae_final.pth` | TODO | TODO |
| `outputs/main_sandstone/unet_final.pth` | TODO | TODO |
| `outputs/main_sandstone/latent_stats.npz` | TODO | TODO |
| `outputs/fontainebleau_phi0p2045/vqvae_final.pth` | TODO | TODO |
| `outputs/fontainebleau_phi0p2045/unet_final.pth` | TODO | TODO |
| `outputs/fontainebleau_phi0p2045/latent_stats.npz` | TODO | TODO |

## Reproduction Modes

- **Workflow reproduction from raw data:** run `python run_pipeline.py --mode full --config configs/main.yaml` to train the models, generate samples, evaluate metrics, and summarize outputs.
- **Exact generated-sample reproduction:** download the archived checkpoints, place them in the paths above, and run the generation/evaluation steps with the released configuration and seeds.

## SHA256 Calculation

On Linux or macOS:

```bash
sha256sum outputs/main_sandstone/vqvae_final.pth
sha256sum outputs/main_sandstone/unet_final.pth
sha256sum outputs/main_sandstone/latent_stats.npz
```

On Windows PowerShell:

```powershell
Get-FileHash outputs/main_sandstone/vqvae_final.pth -Algorithm SHA256
Get-FileHash outputs/main_sandstone/unet_final.pth -Algorithm SHA256
Get-FileHash outputs/main_sandstone/latent_stats.npz -Algorithm SHA256
```
