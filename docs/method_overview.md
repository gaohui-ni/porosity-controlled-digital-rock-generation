# Method overview

The framework contains two model-training stages and one inference-time
physical-matching step:

1. Discrete latent representation learning using a 3D VQ-VAE.
2. Porosity-conditioned latent DDPM training with FiLM modulation.
3. Inference-time topology-preserving adaptive quantile binarization (TP-AQB),
   initialized from the fixed-threshold structure and constrained around
   skeleton and throat neighborhoods, for exact target-porosity matching with
   reduced specific-surface drift.

The main experiment uses spatially disjoint train/validation/test regions.
The separate Fontainebleau experiment trains on one rock and evaluates two
lower-porosity rocks from the same lithology.
