#!/usr/bin/env python3
"""Regenerate the spatial-release SHA256 manifest from final file bytes."""

import argparse
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE = ROOT / "savedmodels" / "spatial_vqaware_v2"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def regenerate(release):
    release = Path(release).resolve()
    manifest = release / "MANIFEST.sha256"
    files = sorted(
        (path for path in release.rglob("*") if path.is_file() and path != manifest),
        key=lambda path: path.relative_to(release).as_posix(),
    )
    lines = [f"{sha256(path)}  {path.relative_to(release).as_posix()}" for path in files]
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return len(files)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-dir", default=str(DEFAULT_RELEASE))
    args = parser.parse_args()
    count = regenerate(args.release_dir)
    print(f"Regenerated manifest for {count} release files.")


if __name__ == "__main__":
    main()
