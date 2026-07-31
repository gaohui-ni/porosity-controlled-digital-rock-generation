"""Verify final checkpoint identity, placement, and training metadata."""

import argparse
import hashlib
import io
import pickle
import sys
import zipfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

MODELS = {
    "main_sandstone": {
        "raw_shape": (800, 800, 800),
        "vqvae_epoch": 49,
        "unet_epoch": 299,
        "poro_center": 0.13,
        "files": {
            "vqvae_final.pth": "4bb59607415ad3deeb20b1876d3c223fe95a63c6c24fa52afaa246840acfffe4",
            "unet_final.pth": "240203a9f040c27d0bf9a45eff849a501677a24732eb60c64603877dd3073761",
            "latent_stats.npz": "042c167cc4377dd1bac1936740ff633a9eb2d87971b6726132cc869778f88d73",
        },
    },
    "fontainebleau_phi0p2045": {
        "raw_shape": (480, 480, 480),
        "vqvae_epoch": 79,
        "unet_epoch": 149,
        "poro_center": 0.13,
        "files": {
            "vqvae_final.pth": "de0ca508d6cf431ec68838d8a022077a3ab366bc36983bc0fcd148dbc516dd5d",
            "unet_final.pth": "b07eb121878509621f015366d801762a875fd360f745e7d47a003e36088986a3",
            "latent_stats.npz": "aa6c495709132e367a06931313c72472e8473fbc21de26a7708ebd99613f1753",
        },
    },
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _TorchStub:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return None


class _MetadataUnpickler(pickle.Unpickler):
    def persistent_load(self, persistent_id):
        return None

    def find_class(self, module, name):
        if module.startswith("torch"):
            return _TorchStub()
        return super().find_class(module, name)


def checkpoint_metadata_without_torch(path):
    with zipfile.ZipFile(path) as archive:
        metadata_name = next(
            name for name in archive.namelist() if name.endswith("/data.pkl")
        )
        checkpoint = _MetadataUnpickler(
            io.BytesIO(archive.read(metadata_name))
        ).load()
    cfg = (checkpoint.get("extra") or {}).get("cfg") or {}
    return int(checkpoint["epoch"]), cfg


def checkpoint_metadata(path):
    try:
        import torch
    except ImportError:
        return checkpoint_metadata_without_torch(path)

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"{path} is not a checkpoint dictionary")
    cfg = (checkpoint.get("extra") or {}).get("cfg") or {}
    return int(checkpoint["epoch"]), cfg


def validate_checkpoint(path, expected_epoch, expected):
    epoch, cfg = checkpoint_metadata(path)
    errors = []
    if epoch != expected_epoch:
        errors.append(f"epoch={epoch}, expected {expected_epoch}")
    if tuple(cfg.get("raw_shape", ())) != expected["raw_shape"]:
        errors.append(f"raw_shape={cfg.get('raw_shape')}, expected {expected['raw_shape']}")
    if abs(float(cfg.get("poro_center", float("nan"))) - expected["poro_center"]) > 1.0e-12:
        errors.append(f"poro_center={cfg.get('poro_center')}, expected {expected['poro_center']}")
    if errors:
        raise ValueError(f"{path}: " + "; ".join(errors))


def validate_latent_stats(path):
    with np.load(path) as data:
        missing = {"mean", "std"} - set(data.files)
        if missing:
            raise ValueError(f"{path}: missing latent statistics {sorted(missing)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT / "savedmodels"))
    parser.add_argument("--hash-only", action="store_true")
    args = parser.parse_args()

    model_root = Path(args.root)
    failures = []
    for model_name, expected in MODELS.items():
        folder = model_root / model_name
        for filename, expected_hash in expected["files"].items():
            path = folder / filename
            if not path.is_file():
                failures.append(f"missing: {path}")
                continue
            actual_hash = sha256(path)
            if actual_hash != expected_hash:
                failures.append(f"{path}: SHA256 {actual_hash}, expected {expected_hash}")
                continue
            print(f"[OK] SHA256 {model_name}/{filename}")

        if not args.hash_only and not failures:
            try:
                validate_checkpoint(folder / "vqvae_final.pth", expected["vqvae_epoch"], expected)
                validate_checkpoint(folder / "unet_final.pth", expected["unet_epoch"], expected)
                validate_latent_stats(folder / "latent_stats.npz")
                print(f"[OK] metadata {model_name}")
            except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                failures.append(str(exc))

    if failures:
        print("\nModel verification failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        raise SystemExit(1)
    print("\nAll final models passed verification.")


if __name__ == "__main__":
    main()
