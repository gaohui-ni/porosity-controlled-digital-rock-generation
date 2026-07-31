import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_binary(path, shape, npz_key):
    path = Path(path)
    if path.suffix.lower() == ".npz":
        with np.load(path) as data:
            if npz_key not in data.files:
                raise ValueError(f"{path}: missing NPZ key {npz_key!r}")
            volume = np.asarray(data[npz_key])
    elif path.suffix.lower() == ".npy":
        volume = np.load(path)
    else:
        volume = np.fromfile(path, dtype=np.uint8)
        expected = int(np.prod(shape))
        if volume.size != expected:
            raise ValueError(f"{path}: got {volume.size} voxels, expected {expected}")
        volume = volume.reshape(shape)

    if volume.ndim != 3:
        raise ValueError(f"{path}: expected a 3D volume, got shape={volume.shape}")
    return (volume > 0).astype(np.uint8)


def slice_porosity_z(volume):
    return np.asarray(volume, dtype=float).mean(axis=(0, 1))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Plot real/generated slice-wise porosity along z.")
    parser.add_argument("--real", required=True)
    parser.add_argument("--generated", required=True)
    parser.add_argument("--shape", nargs=3, type=int, default=[256, 256, 256])
    parser.add_argument("--npz-key", default="seg")
    parser.add_argument("--target", type=float, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-csv")
    args = parser.parse_args()

    shape = tuple(args.shape)
    real_curve = slice_porosity_z(load_binary(args.real, shape, args.npz_key))
    generated_curve = slice_porosity_z(load_binary(args.generated, shape, args.npz_key))
    if real_curve.shape != generated_curve.shape:
        raise ValueError("Real and generated z-axis lengths do not match.")

    z = np.arange(real_curve.size)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.output_csv) if args.output_csv else output.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "z_index": z,
            "real_porosity": real_curve,
            "generated_porosity": generated_curve,
            "target_porosity": args.target,
        }
    ).to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(z, real_curve, label="Real", linewidth=1.5)
    ax.plot(z, generated_curve, label="Generated", linewidth=1.5)
    ax.axhline(args.target, color="tab:blue", linestyle="--", label="Target")
    ax.set_xlabel("Slice index along z")
    ax.set_ylabel("Slice porosity")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)
    print(f"Wrote {output}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
