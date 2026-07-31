import argparse
import json
from pathlib import Path

import numpy as np


def validate_npz(path):
    with np.load(path) as data:
        if "seg" not in data.files:
            raise ValueError(f"{path}: missing required 'seg' array")
        seg = np.asarray(data["seg"])
        if seg.ndim != 3:
            raise ValueError(f"{path}: seg must be 3D, got shape={seg.shape}")
        if not set(np.unique(seg).tolist()).issubset({0, 1}):
            raise ValueError(f"{path}: seg is not binary")

        porosity = float(seg.mean())
        if "seg_porosity" in data.files:
            recorded = float(np.asarray(data["seg_porosity"]).reshape(()))
            if not np.isclose(recorded, porosity, atol=1e-8):
                raise ValueError(
                    f"{path}: seg_porosity={recorded} does not match seg mean={porosity}"
                )

    return {
        "path": str(path),
        "shape": list(seg.shape),
        "seg_porosity": porosity,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate release GPU smoke-test NPZ outputs.")
    parser.add_argument("--main-root", required=True)
    parser.add_argument("--fontainebleau-root", required=True)
    parser.add_argument("--output", default="results/release_smoke_validation.json")
    args = parser.parse_args()

    summary = {}
    for name, root in (
        ("main_sandstone", Path(args.main_root)),
        ("fontainebleau", Path(args.fontainebleau_root)),
    ):
        files = sorted(root.rglob("*.npz"))
        if not files:
            raise RuntimeError(f"{name}: no NPZ output found under {root}")
        summary[name] = validate_npz(files[0])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Validated both release smoke-test outputs: {output}")


if __name__ == "__main__":
    main()
