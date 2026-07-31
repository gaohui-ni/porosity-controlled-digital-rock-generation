import numpy as np
from scipy import ndimage as ndi
from skimage import measure


def euler_characteristic_curve(
    volume_bool,
    voxel_size_um=3.5,
    radius_step_um=3.5,
    radius_max_um=None,
):
    """Return the unnormalized Euler characteristic of the eroded pore space."""
    volume_bool = np.asarray(volume_bool, dtype=bool)
    edt = ndi.distance_transform_edt(volume_bool) * voxel_size_um
    pore_vals = edt[volume_bool]

    if pore_vals.size == 0:
        raise ValueError("The sample has no pore voxels, so the Euler curve cannot be computed.")

    if radius_max_um is None:
        radius_max_um = float(np.percentile(pore_vals, 99))

    radii = np.arange(0, radius_max_um + 1e-9, radius_step_um)
    curve = []
    for radius in radii:
        if np.isclose(radius, 0.0):
            mask = volume_bool.copy()
        else:
            mask = edt >= radius
        curve.append(measure.euler_number(mask.astype(np.uint8), connectivity=3))

    return radii, np.asarray(curve, dtype=float)
