# Main Sandstone Checkpoints

These files are the final checkpoints for the laboratory sandstone experiment.

- Input data: binary laboratory sandstone micro-CT volume
- Raw volume: `800 x 800 x 800` voxels
- Voxel size: `3.5 um/voxel`
- Generated volume: `256 x 256 x 256` voxels
- VQ-VAE training: 50 epochs
- Latent DDPM training: 300 epochs
- Training seed: 123
- Default inference seed sequence: starts at 0

Files:

- `vqvae_final.pth`: final VQ-VAE checkpoint
- `unet_final.pth`: final FiLM-conditioned latent-DDPM U-Net checkpoint
- `latent_stats.npz`: latent normalization statistics used for sampling

Use this checkpoint set through:

```bash
python run_pipeline.py --mode final --config configs/main.yaml
```

SHA256 values are listed in `docs/model_manifest.md`.
