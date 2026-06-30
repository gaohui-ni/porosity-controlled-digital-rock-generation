import numpy as np

from src.metrics.two_point_correlation import two_point_correlation_xyz


def test_two_point_correlation_returns_expected_keys_and_lengths():
    volume = np.ones((4, 4, 4), dtype=np.uint8)
    curves = two_point_correlation_xyz(volume, max_lag=2)

    assert set(curves) == {"x", "y", "z", "R"}
    for values in curves.values():
        assert values.shape == (3,)


def test_two_point_correlation_is_one_for_all_pore_volume():
    volume = np.ones((4, 4, 4), dtype=np.uint8)
    curves = two_point_correlation_xyz(volume, max_lag=3)

    for values in curves.values():
        np.testing.assert_allclose(values, np.ones(4))


def test_two_point_correlation_zero_lag_equals_porosity():
    volume = np.zeros((4, 4, 4), dtype=np.uint8)
    volume[:2] = 1

    curves = two_point_correlation_xyz(volume, max_lag=2)

    assert curves["R"][0] == 0.5
