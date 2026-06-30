import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics.two_point_correlation import two_point_correlation_xyz


def load_demo_volume(path):
    path = Path(path)
    if path.exists():
        return np.load(path)
    fallback = Path("examples/demo_output/demo_seg.npy")
    if fallback.exists():
        return np.load(fallback)
    raise FileNotFoundError("No demo segmentation found. Run `python run_demo.py` first.")


def write_placeholder_png(path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    volume = load_demo_volume("examples/demo_output/demo_seg.npy")
    mid = volume.shape[2] // 2
    curves = two_point_correlation_xyz(volume, max_lag=min(32, volume.shape[0] - 1))

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    axes[0].imshow(volume[:, :, mid], cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Synthetic slice")
    axes[0].axis("off")
    axes[1].plot(curves["R"])
    axes[1].set_title("S2_R")
    axes[1].set_xlabel("Lag r")
    axes[1].set_ylabel("Correlation")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description="Generate lightweight reviewer demo figures and summary files.")
    parser.add_argument("--summary", default="results/summary.json")
    parser.add_argument("--figure", default="results/figures/demo_synthetic_s2.png")
    args = parser.parse_args()

    wrote_figure = write_placeholder_png(args.figure)
    summary = {
        "mode": "demo",
        "figure": args.figure if wrote_figure else None,
        "notes": "Demo figure generated from the synthetic 64^3 workflow. Manuscript-scale figures require full outputs.",
    }
    out = Path(args.summary)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out}")
    if wrote_figure:
        print(f"Wrote {args.figure}")


if __name__ == "__main__":
    main()
