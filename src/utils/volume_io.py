"""Small, dependency-light volume loading helpers."""

from pathlib import Path

import numpy as np


def load_npz_array(path, key="seg", validate_porosity=False, atol=1.0e-8):
    path = Path(path)
    with np.load(path) as data:
        if key not in data.files:
            raise KeyError(f"{path} does not contain key={key}; available keys={data.files}")
        array = np.asarray(data[key])
        if validate_porosity and "seg_porosity" in data.files:
            stored = float(np.asarray(data["seg_porosity"]).reshape(()))
            achieved = float(array.mean())
            if not np.isclose(achieved, stored, atol=atol, rtol=0.0):
                raise ValueError(
                    f"{path}: {key} porosity={achieved:.12g} does not match "
                    f"seg_porosity={stored:.12g}"
                )
    return array
