"""Verify the second-version spatial release after ``git lfs pull``."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE = ROOT / "savedmodels" / "spatial_vqaware_v2"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-dir", default=str(DEFAULT_RELEASE))
    parser.add_argument("--check-tensors", action="store_true")
    args = parser.parse_args()
    release = Path(args.release_dir).resolve()

    manifest = release / "MANIFEST.sha256"
    failures = []
    entries = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = release / relative
        if not path.is_file():
            failures.append(f"missing: {relative}")
            continue
        actual = sha256(path)
        entries.append(relative)
        if actual != expected:
            failures.append(
                f"checksum: {relative}: expected {expected}, got {actual}"
            )

    json_paths = [
        "spatial_split.json",
        "optional_precalibration/run_cfg.json",
        "R03_calibrated/calibration_definition.json",
        "R03_calibrated/r03_identity_gates.json",
        "R03_calibrated/r03_verdict_20seed.json",
    ]
    for relative in json_paths:
        json.loads((release / relative).read_text(encoding="utf-8"))

    cohort = read_csv(release / "test_cohort_v1/test_crop_porosity_all_100.csv")
    seeds = read_csv(release / "final_test_v2/final_test_seed_manifest.csv")
    pairs = read_csv(release / "final_test_v2/FINAL_TEST_PAIR_MANIFEST.csv")
    used = [row for row in cohort if row["target_group"] != "none"]
    groups = Counter(row["target_group"] for row in cohort)
    expected_groups = {"none": 5, "0.11": 8, "0.12": 29, "0.13": 39, "0.14": 19}
    if len(cohort) != 100 or len(used) != 95 or dict(groups) != expected_groups:
        failures.append(
            f"cohort: total={len(cohort)}, used={len(used)}, groups={dict(groups)}"
        )
    if len(seeds) != 95 or len(pairs) != 95:
        failures.append(f"manifests: seeds={len(seeds)}, pairs={len(pairs)}")
    used_ids = {row["crop_id"] for row in used}
    if used_ids != {row["crop_id"] for row in seeds}:
        failures.append("seed-manifest crop IDs do not match the 95 used crops")
    if used_ids != {row["crop_id"] for row in pairs}:
        failures.append("pair-manifest crop IDs do not match the 95 used crops")

    if args.check_tensors and not failures:
        import torch

        source = torch.load(
            release / "optional_precalibration/unet_best_noise.pth",
            map_location="cpu",
            weights_only=False,
        )
        calibrated = torch.load(
            release / "R03_calibrated/calib_A_scale.pth",
            map_location="cpu",
            weights_only=False,
        )
        source_state = source["model_state_dict"]
        calibrated_state = calibrated["model_state_dict"]
        if source_state.keys() != calibrated_state.keys() or not all(
            torch.equal(source_state[key], calibrated_state[key])
            for key in source_state
        ):
            failures.append("calibrated and source DDPM tensors differ")
        if abs(float(dict(source["cfg"])["poro_scale"]) - 0.02) >= 1e-12:
            failures.append("source checkpoint poro_scale is not 0.02")
        if abs(float(dict(calibrated["cfg"])["poro_scale"]) - 0.011804) >= 1e-12:
            failures.append("calibrated checkpoint poro_scale is not 0.011804")

    if failures:
        raise SystemExit("Spatial release verification failed:\n- " + "\n- ".join(failures))
    print(
        f"Spatial release verified: {len(entries)} files, "
        f"100 cohort rows, 95 used samples."
    )


if __name__ == "__main__":
    main()
