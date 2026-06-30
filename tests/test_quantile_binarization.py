import numpy as np

from src.sampling.quantile_binarization import quantile_binarize


def test_quantile_binarize_matches_target_porosity():
    prob = np.linspace(0.0, 1.0, 1000, dtype=np.float32).reshape(10, 10, 10)

    seg, threshold, achieved_porosity = quantile_binarize(
        prob, target_porosity=0.2, seed=0
    )

    assert seg.shape == prob.shape
    assert seg.dtype == np.uint8
    assert abs(achieved_porosity - 0.2) <= 1.0 / prob.size
    assert 0.0 <= threshold <= 1.0


def test_quantile_binarize_handles_tied_values_deterministically():
    prob = np.ones((4, 4, 4), dtype=np.float32)

    seg, _, achieved_porosity = quantile_binarize(prob, target_porosity=0.25, seed=7)

    assert int(seg.sum()) == 16
    assert achieved_porosity == 0.25
