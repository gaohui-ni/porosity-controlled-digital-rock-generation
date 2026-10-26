# Method overview

The framework contains two model-training stages and one inference-time
physical-matching step:

1. Discrete latent representation learning using a 3D VQ-VAE.
2. Porosity-conditioned latent DDPM training with FiLM modulation.
3. Inference-time adaptive quantile binarization (TP-AQB) with constrained
   voxel adjustment and topological consistency checking, initialized from the
   fixed-threshold structure and safeguarded around skeleton and throat
   neighborhoods. The constraints are intended to reduce alterations to
   critical pore connections while matching target porosity and limiting
   specific-surface drift; they do not guarantee invariant topology.

The main experiment uses spatially disjoint train/validation/test regions.
The separate Fontainebleau experiment trains on one rock and evaluates two
lower-porosity rocks from the same lithology.
