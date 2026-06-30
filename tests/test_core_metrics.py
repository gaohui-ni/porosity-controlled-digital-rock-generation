import numpy as np

from src.metrics.porosity import compute_porosity
from src.metrics.two_point_correlation import two_point_correlation_xyz
from src.sampling.quantile_binarization import quantile_binarize


def test_compute_porosity_counts_positive_voxels():
    volume = np.zeros((4, 4, 4), dtype=np.uint8)
    volume[:2] = 1

    assert compute_porosity(volume) == 0.5


def test_quantile_binarize_matches_target_count():
    prob = np.linspace(0.0, 1.0, 1000, dtype=np.float32).reshape(10, 10, 10)
    seg, threshold, achieved = quantile_binarize(prob, target_porosity=0.2, seed=0)

    assert seg.shape == prob.shape
    assert seg.dtype == np.uint8
    assert abs(achieved - 0.2) <= 1.0 / prob.size
    assert 0.0 <= threshold <= 1.0


def test_two_point_correlation_returns_xyz_and_average():
    volume = np.ones((4, 4, 4), dtype=np.uint8)
    curves = two_point_correlation_xyz(volume, max_lag=2)

    assert set(curves) == {"x", "y", "z", "R"}
    for values in curves.values():
        np.testing.assert_allclose(values, np.ones(3))
