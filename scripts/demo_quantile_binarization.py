import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import os
import numpy as np
from scipy.ndimage import gaussian_filter

from src.sampling.quantile_binarization import quantile_binarize
from src.metrics.porosity import compute_porosity


def main():
    os.makedirs("examples/demo_output", exist_ok=True)

    rng = np.random.default_rng(0)
    noise = rng.random((64, 64, 64))
    prob = gaussian_filter(noise, sigma=2)
    prob = (prob - prob.min()) / (prob.max() - prob.min())

    target_porosity = 0.13
    seg, threshold, seg_poro = quantile_binarize(prob, target_porosity, seed=0)

    np.save("examples/demo_input.npy", prob.astype(np.float32))
    np.save("examples/demo_output/demo_seg.npy", seg.astype(np.uint8))

    print(f"Target porosity:   {target_porosity:.6f}")
    print(f"Achieved porosity: {seg_poro:.6f}")
    print(f"Check porosity:    {compute_porosity(seg):.6f}")
    print(f"Threshold:         {threshold:.6f}")


if __name__ == "__main__":
    main()
