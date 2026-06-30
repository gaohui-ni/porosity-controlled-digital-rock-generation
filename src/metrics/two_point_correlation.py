import numpy as np


def two_point_correlation_axis(binary_volume: np.ndarray, axis: int, max_lag: int) -> np.ndarray:
    pore = (binary_volume > 0).astype(np.float32)
    values = []

    for r in range(max_lag + 1):
        if r == 0:
            values.append(float(np.mean(pore * pore)))
        else:
            slicer1 = [slice(None)] * 3
            slicer2 = [slice(None)] * 3
            slicer1[axis] = slice(0, -r)
            slicer2[axis] = slice(r, None)
            a = pore[tuple(slicer1)]
            b = pore[tuple(slicer2)]
            values.append(float(np.mean(a * b)))

    return np.asarray(values)


def two_point_correlation_xyz(binary_volume: np.ndarray, max_lag: int):
    sx = two_point_correlation_axis(binary_volume, axis=0, max_lag=max_lag)
    sy = two_point_correlation_axis(binary_volume, axis=1, max_lag=max_lag)
    sz = two_point_correlation_axis(binary_volume, axis=2, max_lag=max_lag)
    sr = (sx + sy + sz) / 3.0
    return {"x": sx, "y": sy, "z": sz, "R": sr}
