import numpy as np


def compute_porosity(binary_volume: np.ndarray) -> float:
    if binary_volume.ndim != 3:
        raise ValueError("binary_volume must be a 3D array.")
    return float(np.mean(binary_volume > 0))
