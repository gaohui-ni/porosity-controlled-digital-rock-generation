import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import argparse
import numpy as np

from src.metrics.porosity import compute_porosity
from src.metrics.two_point_correlation import two_point_correlation_xyz
from src.metrics.specific_surface_area import compute_specific_surface_area


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--max_lag", type=int, default=64)
    args = parser.parse_args()

    vol = np.load(args.input)
    print(f"Porosity: {compute_porosity(vol):.6f}")
    print(f"Specific surface area: {compute_specific_surface_area(vol):.6f}")

    s2 = two_point_correlation_xyz(vol, max_lag=args.max_lag)
    for key, val in s2.items():
        print(f"S2-{key}: shape={val.shape}, first={val[0]:.6f}")


if __name__ == "__main__":
    main()
