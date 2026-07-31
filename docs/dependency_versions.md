# Dependency Versions

The exact core package versions reported by the manuscript-scale experiment
environment are recorded in:

- `environment/environment-lock.yml`
- `environment/requirements-lock.txt`

## Runtime

| Component | Version |
| --- | --- |
| Python | 3.10.12 |
| PyTorch | 2.5.1 |
| PyTorch CUDA runtime (`torch.version.cuda`) | 12.1 |
| cuDNN | 9.1 |
| NumPy | 2.2.6 |
| SciPy | 1.15.3 |
| pandas | 2.3.3 |
| matplotlib | 3.10.8 |
| scikit-image | 0.25.2 |
| PoreSpy | 3.0.2 |
| OpenPNM | 3.5.2 |

The CUDA version in this table is the runtime used by PyTorch. The host driver
reported CUDA compatibility 13.2 and the server `nvcc` compiler reported 12.6;
these are not substituted for `torch.version.cuda` in the reproducibility
claim.

`h5py` and `torchvision` are not included because the released workflow does
not import them and `torchvision` was not installed in the reported experiment
environment.

## Installation

```bash
conda env create -f environment/environment-lock.yml
conda activate vq256_cuda
pip install -e .
```

The shorter `requirements.txt`, `requirements_optional.txt`, and
`environment.yml` remain convenient, unpinned setup files. Use the files under
`environment/` for manuscript reproduction.
