import numpy as np


def compute_specific_surface_area(binary_volume: np.ndarray) -> float:
    pore = (binary_volume > 0).astype(np.uint8)
    interface = 0

    interface += np.sum(pore[1:, :, :] != pore[:-1, :, :])
    interface += np.sum(pore[:, 1:, :] != pore[:, :-1, :])
    interface += np.sum(pore[:, :, 1:] != pore[:, :, :-1])

    return float(interface) / float(pore.size)
