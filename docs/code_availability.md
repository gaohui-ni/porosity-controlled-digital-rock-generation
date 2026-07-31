# Code Availability

The code developed for this study is available at:

https://github.com/gaohui-ni/porosity-controlled-digital-rock-generation

The repository includes the 3D VQ-VAE, FiLM-conditioned latent DDPM, quantile-based porosity matching, configuration files, training scripts, batch generation scripts, evaluation scripts, and synthetic examples.

## Manuscript Code Availability Statement

> The source code and pretrained model checkpoints used in this study are
> publicly available in the GitHub repository *Porosity-Controlled 3D Digital
> Rock Generation*. The final checkpoints are distributed through Git LFS,
> with file sizes and SHA-256 checksums documented in the repository. The
> repository also provides configuration files, synthetic examples,
> documentation, and workflows for both checkpoint-based inference and
> independent model retraining. The code is released under the MIT License.

## Repository Details

Name of code/library: Porosity-Controlled 3D Digital Rock Generation

Contact: Hao Ni, nihao@upc.edu.cn

Hardware requirements: A standard CPU is sufficient for the lightweight demonstration and unit tests. A CUDA-capable NVIDIA GPU is recommended for manuscript-scale model training and generation. The experiments reported in this study were conducted using an NVIDIA A100 40 GB GPU.

Program language: Python 3.10.

Software required: PyTorch, NumPy, SciPy, pandas, scikit-image, Matplotlib, PyYAML, PoreSpy, and OpenPNM.

The source code, configuration files, synthetic examples, documentation, and reproduction instructions are publicly available at:

https://github.com/gaohui-ni/porosity-controlled-digital-rock-generation

The final trained VQ-VAE and latent-DDPM checkpoints are distributed under
`savedmodels/` through Git LFS. After cloning, users must run `git lfs pull`
before checkpoint-based inference. The repository also provides the complete
main-sandstone training workflow for independently retraining the models from
the publicly available raw data. Complete Fontainebleau validation additionally
requires the external ANU volumes, which are not redistributed.

The released checkpoints enable reproduction of the reported model inference
and analysis workflow. Due to stochastic sampling and hardware- or
software-dependent numerical differences, generated volumes may not be bitwise
identical across computational environments. Independently retrained models are
expected to reproduce the reported methodological and statistical behavior
rather than identical generated samples.

## Notes for Review

Large raw micro-CT volumes are not stored in the repository. Raw data availability is documented in `docs/data_availability.md`; final model file layout and checksums are documented in `docs/checkpoints.md` and `docs/model_manifest.md`.
