import numpy as np
import pytest

from src.metrics.porosity import compute_porosity


def test_compute_porosity_counts_positive_voxels():
    volume = np.zeros((4, 4, 4), dtype=np.uint8)
    volume[:2] = 1

    assert compute_porosity(volume) == 0.5


def test_compute_porosity_treats_any_positive_value_as_pore():
    volume = np.zeros((2, 2, 2), dtype=np.uint8)
    volume[0, 0, 0] = 1
    volume[1, 1, 1] = 255

    assert compute_porosity(volume) == 0.25


def test_compute_porosity_rejects_non_3d_arrays():
    with pytest.raises(ValueError, match="3D"):
        compute_porosity(np.zeros((4, 4), dtype=np.uint8))
