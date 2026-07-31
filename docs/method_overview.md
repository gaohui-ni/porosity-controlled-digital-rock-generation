# Method overview

The framework contains two model-training stages and one inference-time
physical-matching step:

1. Discrete latent representation learning using a 3D VQ-VAE.
2. Porosity-conditioned latent DDPM training with FiLM modulation.
3. Inference-time adaptive quantile-threshold binarization for explicit
   target-porosity matching.
