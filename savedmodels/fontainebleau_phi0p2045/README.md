# Fontainebleau Checkpoints

These files are compatibility checkpoints for the earlier Fontainebleau
workflow. The second-version manuscript uses three conditions and reports the
corresponding measured whole-volume porosities separately.

- Input data: binary Fontainebleau sandstone micro-CT volume
- Raw volume: `480 x 480 x 480` voxels
- Voxel size: `5.68 um/voxel`
- Generated volume: `256 x 256 x 256` voxels
- VQ-VAE training: 80 epochs
- Latent DDPM training: 150 epochs
- FiLM porosity normalization: `(phi - 0.13) / 0.02`
- Training seed: 123
- Default inference seed sequence: starts at 0
- Repository labels: `0.2045`, `0.1743`, and `0.1263`
- Manuscript measured whole-volume porosities: `0.2041`, `0.1766`, and `0.1289`

Files:

- `vqvae_final.pth`: final VQ-VAE checkpoint
- `unet_final.pth`: final FiLM-conditioned latent-DDPM U-Net checkpoint
- `latent_stats.npz`: latent normalization statistics used for sampling

Use this checkpoint set through:

```bash
python run_pipeline.py --mode final --config configs/main.yaml
```

SHA256 values are listed in `docs/model_manifest.md`.
