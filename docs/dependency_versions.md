# Dependency Versions

This repository documents the package set in `requirements.txt`, `requirements_optional.txt`, and `environment.yml`.

For the final manuscript release, exact versions should be exported from the experiment environment used for the full GPU workflow. They are not filled in here without the actual environment to avoid reporting fabricated versions.

## Minimum Versions to Record

Record the final values for:

- Python: `3.10`
- PyTorch: TBD from final experiment environment
- NumPy: TBD from final experiment environment
- SciPy: TBD from final experiment environment
- scikit-image: TBD from final experiment environment
- porespy: TBD from final experiment environment
- openpnm: TBD from final experiment environment
- CUDA toolkit / driver used for training: TBD from final experiment environment

## Suggested Export Commands

From the final experiment environment:

```bash
python -V
python -c "import torch, numpy, scipy, skimage; print('torch', torch.__version__); print('numpy', numpy.__version__); print('scipy', scipy.__version__); print('skimage', skimage.__version__)"
python -c "import porespy, openpnm; print('porespy', porespy.__version__); print('openpnm', openpnm.__version__)"
pip freeze > requirements.lock.txt
conda env export --from-history > environment.lock.yml
```

Commit the resulting lock files, or paste the exact versions into this document, after the final full run has been completed.
