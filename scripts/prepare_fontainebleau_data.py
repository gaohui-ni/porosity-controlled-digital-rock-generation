import argparse
from pathlib import Path

import numpy as np


def load_volume(path: Path, raw_shape=None, raw_dtype="uint8", npz_key=None):
    suffix = path.suffix.lower()
    if suffix in [".raw", ".bin"]:
        if raw_shape is None:
            raise ValueError("raw_shape is required for raw/bin input")
        arr = np.fromfile(path, dtype=np.dtype(raw_dtype))
        expected = int(np.prod(raw_shape))
        if arr.size != expected:
            raise ValueError(f"size mismatch: got {arr.size}, expected {expected}")
        return arr.reshape(tuple(raw_shape))
    if suffix == ".npy":
        return np.load(path)
    if suffix == ".npz":
        z = np.load(path)
        if npz_key is None:
            # Prefer common binary keys; otherwise use the first 3D array.
            for key in ["seg", "binary", "volume", "arr_0"]:
                if key in z.files and np.asarray(z[key]).ndim == 3:
                    return z[key]
            for key in z.files:
                if np.asarray(z[key]).ndim == 3:
                    return z[key]
            raise ValueError(f"no 3D array found in {path}, keys={z.files}")
        if npz_key not in z.files:
            raise ValueError(f"npz_key={npz_key} not found, available={z.files}")
        return z[npz_key]
    raise ValueError(f"unsupported input type: {path}")


def binarize(arr, pore_value=1, threshold=None, invert=False):
    arr = np.asarray(arr)
    if arr.dtype == np.bool_:
        vol01 = arr.astype(np.uint8)
    elif threshold is None:
        vol01 = (arr == pore_value).astype(np.uint8)
    else:
        vol01 = (arr.astype(np.float32) > float(threshold)).astype(np.uint8)
    if invert:
        vol01 = 1 - vol01
    return vol01


def main():
    p = argparse.ArgumentParser(description="Prepare Fontainebleau sandstone volume as uint8 raw: 0=solid, 1=pore.")
    p.add_argument("--input", required=True)
    p.add_argument("--output_raw", required=True)
    p.add_argument("--raw_shape", type=int, nargs=3, default=[480, 480, 480])
    p.add_argument("--raw_dtype", default="uint8")
    p.add_argument("--npz_key", default=None)
    p.add_argument("--pore_value", type=float, default=1)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--invert", action="store_true")
    args = p.parse_args()

    input_path = Path(args.input)
    out_path = Path(args.output_raw)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    arr = load_volume(input_path, raw_shape=args.raw_shape, raw_dtype=args.raw_dtype, npz_key=args.npz_key)
    vol01 = binarize(arr, pore_value=args.pore_value, threshold=args.threshold, invert=args.invert)

    if vol01.ndim != 3:
        raise ValueError(f"expected 3D volume, got shape={vol01.shape}")

    vol01.astype(np.uint8).tofile(out_path)
    print(f"Saved: {out_path}")
    print(f"shape={vol01.shape}, porosity={float(vol01.mean()):.6f}, unique={np.unique(vol01).tolist()}")


if __name__ == "__main__":
    main()
