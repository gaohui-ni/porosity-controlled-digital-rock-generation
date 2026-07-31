# Dependency Versions

The core direct dependency versions reported by the manuscript-scale experiment
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

## Reproducibility Status

- Core direct dependency versions: recorded.
- Complete transitive environment snapshot: pending server export.
- Clean-environment reconstruction: pending server validation.
- GPU checkpoint inference smoke test: pending server validation.

The two current lock files are intentionally concise and are not represented as
the complete output of `pip freeze` or `conda env export`.

From the original experiment server, export:

```bash
pip freeze > environment/pip-freeze-full.txt
conda env export --no-builds > environment/conda-export-full.yml
```

Then validate from a new environment:

```bash
conda env create -f environment/environment-lock.yml -n vq256_release_test
conda activate vq256_release_test
pip install -e .
pytest tests/
python run_pipeline.py --mode demo --config configs/main.yaml
```
