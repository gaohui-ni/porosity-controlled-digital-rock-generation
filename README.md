# Porosity-Controlled 3D Digital Rock Generation

Core implementation for the manuscript **Porosity-controlled 3D digital rock generation using discrete latent diffusion**.

This repository provides a reproducible code base for 3D digital rock generation with a lightweight 3D VQ-VAE, a FiLM-conditioned latent DDPM, and quantile-based adaptive binarization for explicit porosity control.

## Scope

The repository contains:

- 3D VQ-VAE and vector-quantized latent representation learning.
- FiLM-conditioned latent DDPM for target-porosity controlled generation.
- Quantile-based binarization for matching prescribed porosity.
- Training, sampling, batch-generation, and Fontainebleau validation scripts.
- Evaluation utilities for porosity, two-point statistics, pore-size statistics, topology, pore-network features, and OpenPNM permeability workflows.
- Synthetic examples that can be used without restricted raw micro-CT data.

## Repository Structure

```text
configs/      Example YAML configurations.
src/          Importable model, data, training, sampling, metric, and utility code.
scripts/      Command-line workflows for training, generation, and evaluation.
examples/     Small synthetic demo data and demo outputs.
docs/         Installation, user guide, reproducibility, and data/code availability notes.
data/         Placeholder folders for user-provided raw data.
outputs/      Placeholder folder for generated checkpoints and outputs.
results/      Placeholder folder for generated figures, curves, and tables.
savedmodels/  Placeholder folder for trained model checkpoints.
```

## Installation

Conda is recommended:

```bash
conda env create -f environment.yml
conda activate digitalrock
pip install -e .
```

Alternatively:

```bash
pip install -r requirements.txt
pip install -r requirements_optional.txt
pip install -e .
```

`porespy` and `openpnm` are optional but required for pore-network and permeability workflows. GPU training requires a CUDA-compatible PyTorch installation.

## Minimal Demo

Run the synthetic porosity-matching demo:

```bash
python scripts/demo_quantile_binarization.py
```

Expected behavior:

- writes `examples/demo_input.npy`;
- writes `examples/demo_output/demo_seg.npy`;
- prints the target porosity, achieved porosity, check porosity, and adaptive threshold.

This demo is intentionally lightweight and does not require restricted micro-CT data or trained checkpoints.

A notebook version of this workflow is available at:

```text
notebooks/tutorials/0001-basic-usage-synthetic-rock.ipynb
```

## Main Manuscript Workflow

See [docs/reproducibility.md](docs/reproducibility.md) and [docs/user_guide.md](docs/user_guide.md) for the full workflow.

At a high level:

1. Place a binary raw digital rock volume under `data/raw/`.
2. Build porosity-matched real comparison groups with `scripts/build_real_phi_groups.py`.
3. Train the 3D VQ-VAE and latent DDPM.
4. Generate batches at target porosity values.
5. Evaluate spatial statistics, topology, pore-network metrics, and permeability.
6. Repeat the validation workflow for Fontainebleau sandstone data when available.

## Data Availability

The laboratory sandstone micro-CT volume data and associated metadata used in this study are available in Mendeley Data:

https://doi.org/10.17632/yp2yw9c7jj.1

The repository also includes synthetic examples and scripts that allow reviewers and users to test the workflow on provided or user-supplied binary 3D volumes.

See [docs/data_availability.md](docs/data_availability.md).

## Checkpoints

Large trained checkpoints are not included. Training scripts save:

- `vqvae_final.pth`
- `unet_final.pth`
- `latent_stats.npz`

Generation scripts expect these files under the checkpoint directory passed by `--ckpt_dir`.

## Citation

If you use this repository, please cite the associated manuscript. A machine-readable citation file is provided in [CITATION.cff](CITATION.cff).

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
