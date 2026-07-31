# Code Availability

The code developed for this study is available at:

https://github.com/gaohui-ni/porosity-controlled-digital-rock-generation

The repository includes the 3D VQ-VAE, FiLM-conditioned latent DDPM, quantile-based porosity matching, configuration files, training scripts, batch generation scripts, evaluation scripts, and synthetic examples.

## Suggested Manuscript Statement

> The code developed for this study is available at https://github.com/gaohui-ni/porosity-controlled-digital-rock-generation. The repository contains the implementation of the 3D VQ-VAE, the FiLM-conditioned latent DDPM, quantile-based porosity matching, configuration files, training and inference scripts, evaluation workflows, and synthetic examples for reproducibility.

## Detailed Manuscript Code Availability Statement

Name of code/library: Porosity-Controlled 3D Digital Rock Generation

Contact: Hao Ni, nihao@upc.edu.cn

Hardware requirements: A standard CPU is sufficient for the lightweight demonstration and unit tests. A CUDA-capable NVIDIA GPU is recommended for manuscript-scale model training and generation. The experiments reported in this study were conducted using an NVIDIA A100 40 GB GPU.

Program language: Python 3.10.

Software required: PyTorch, NumPy, SciPy, pandas, scikit-image, Matplotlib, PyYAML, PoreSpy, and OpenPNM.

The source code, configuration files, synthetic examples, documentation, and reproduction instructions are publicly available at:

https://github.com/gaohui-ni/porosity-controlled-digital-rock-generation

The final trained VQ-VAE and latent-DDPM checkpoint layout is provided under `savedmodels/` when the checkpoint package is included locally or published through Git LFS / release assets. The repository also provides the complete training workflow for independently retraining the models from the publicly available raw data. Because model training and diffusion sampling are stochastic, independently reproduced results may differ in exact generated structures and numerical values while retaining the reported methodological and statistical behavior.

## Notes for Review

Large raw micro-CT volumes are not stored in the repository. Raw data availability is documented in `docs/data_availability.md`; final model file layout and checksums are documented in `docs/checkpoints.md` and `docs/model_manifest.md`.
