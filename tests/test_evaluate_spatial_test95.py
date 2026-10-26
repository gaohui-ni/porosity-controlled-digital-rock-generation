import csv
import json

import numpy as np
import pytest

from scripts.evaluate_spatial_test95 import EXPECTED_GROUPS, assemble


SHAPE = (2, 2, 2)


def build_fixture(tmp_path):
    manifest = tmp_path / "manifest.csv"
    gen_root = tmp_path / "gen"
    tpaqb_root = tmp_path / "tpaqb"
    out_root = tmp_path / "results"
    gen_root.mkdir()
    rows = []
    index = 0
    for group, count in EXPECTED_GROUPS.items():
        group_dir = tpaqb_root / f"phi{group.replace('.', 'p')}"
        group_dir.mkdir(parents=True)
        for _ in range(count):
            crop_id = f"crop_{index:03d}"
            seed = 1000 + index
            target = float(group)
            real = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.uint8).reshape(SHAPE)
            fixed = np.asarray([0, 0, 0, 1, 0, 0, 0, 1], dtype=np.uint8).reshape(SHAPE)
            topn = np.asarray([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.uint8).reshape(SHAPE)
            tpaqb = np.asarray([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint8).reshape(SHAPE)
            names = {
                "real_path": f"{crop_id}_real.raw",
                "prob_path": f"{crop_id}_prob.npz",
                "pre_aqb_path": f"{crop_id}_off.raw",
                "post_aqb_path": f"{crop_id}_on.raw",
            }
            real.tofile(gen_root / names["real_path"])
            fixed.tofile(gen_root / names["pre_aqb_path"])
            topn.tofile(gen_root / names["post_aqb_path"])
            np.savez(gen_root / names["prob_path"], prob=np.zeros(SHAPE),
                     target_porosity=target, seed=seed)
            stem = f"{crop_id}_prob_tp_aqb"
            tpaqb.tofile(group_dir / f"{stem}.raw")
            metadata = json.dumps({"segment": {"method": "TP-AQB"}, "repair": {}})
            np.savez(group_dir / f"{stem}.npz", seg=tpaqb,
                     target_porosity=target, achieved_porosity=float(tpaqb.mean()),
                     seed=seed, metadata_json=np.asarray(metadata))
            rows.append({
                "crop_id": crop_id,
                "group": group,
                "target_phi": repr(target),
                "phi_real": repr(float(real.mean())),
                "phi_pre_aqb": repr(float(fixed.mean())),
                "phi_post_aqb": repr(float(topn.mean())),
                "seed": str(seed),
                **names,
            })
            index += 1

    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(reversed(rows))
    return manifest, gen_root, tpaqb_root, out_root


def test_test95_wrapper_stages_voxel_identical_tpaqb_raw_and_npz(tmp_path):
    manifest, gen_root, tpaqb_root, out_root = build_fixture(tmp_path)
    augmented, summaries = assemble(
        manifest, gen_root, tpaqb_root, out_root, expected_shape=SHAPE
    )

    assert len(augmented) == 95
    assert len(summaries) == 4
    assert sum(summary["n"] for summary in summaries) == 95
    assert len(list((out_root / "stage" / "tpaqb").rglob("*.raw"))) == 95
    assert len(list((out_root / "stage" / "tpaqb_npz").rglob("*.npz"))) == 95
    assert augmented[0]["crop_id"] == "crop_000"
    assert float(augmented[0]["phi_tpaqb"]) == 0.125
    raw_path = out_root / "stage" / "tpaqb" / "phi0p11" / "crop_000.raw"
    npz_path = out_root / "stage" / "tpaqb_npz" / "phi0p11" / "crop_000.npz"
    assert raw_path.is_file() and npz_path.is_file()
    with np.load(npz_path) as data:
        assert np.array_equal(np.fromfile(raw_path, dtype=np.uint8).reshape(SHAPE), data["seg"])


def test_test95_wrapper_rejects_tpaqb_raw_npz_mismatch(tmp_path):
    manifest, gen_root, tpaqb_root, out_root = build_fixture(tmp_path)
    bad_raw = next(tpaqb_root.rglob("crop_000_prob_tp_aqb.raw"))
    damaged = np.fromfile(bad_raw, dtype=np.uint8)
    damaged[0], damaged[1] = damaged[1], damaged[0]
    damaged.tofile(bad_raw)

    with pytest.raises(ValueError, match="RAW and NPZ seg differ"):
        assemble(manifest, gen_root, tpaqb_root, out_root, expected_shape=SHAPE)


def test_test95_wrapper_rejects_missing_tpaqb_provenance(tmp_path):
    manifest, gen_root, tpaqb_root, out_root = build_fixture(tmp_path)
    npz_path = next(tpaqb_root.rglob("crop_000_prob_tp_aqb.npz"))
    with np.load(npz_path) as data:
        payload = {key: data[key] for key in data.files if key != "metadata_json"}
    np.savez(npz_path, **payload)

    with pytest.raises(KeyError, match="metadata_json"):
        assemble(manifest, gen_root, tpaqb_root, out_root, expected_shape=SHAPE)


def test_test95_wrapper_rejects_duplicate_tpaqb_npz(tmp_path):
    manifest, gen_root, tpaqb_root, out_root = build_fixture(tmp_path)
    source = next(tpaqb_root.rglob("crop_000_prob_tp_aqb.npz"))
    duplicate = tpaqb_root / "duplicate" / "crop_000_tp_aqb.npz"
    duplicate.parent.mkdir()
    duplicate.write_bytes(source.read_bytes())

    with pytest.raises(FileNotFoundError, match="expected one file, found 2"):
        assemble(manifest, gen_root, tpaqb_root, out_root, expected_shape=SHAPE)


def test_test95_wrapper_rejects_duplicate_manifest_crop_id(tmp_path):
    manifest, gen_root, tpaqb_root, out_root = build_fixture(tmp_path)
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows[1]["crop_id"] = rows[0]["crop_id"]
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="Duplicate crop_id"):
        assemble(manifest, gen_root, tpaqb_root, out_root, expected_shape=SHAPE)


def test_test95_wrapper_rejects_missing_branch_file(tmp_path):
    manifest, gen_root, tpaqb_root, out_root = build_fixture(tmp_path)
    (gen_root / "crop_000_off.raw").unlink()

    with pytest.raises(FileNotFoundError, match="crop_000 fixed"):
        assemble(manifest, gen_root, tpaqb_root, out_root, expected_shape=SHAPE)


def test_test95_wrapper_rejects_corrupt_raw_shape(tmp_path):
    manifest, gen_root, tpaqb_root, out_root = build_fixture(tmp_path)
    (gen_root / "crop_000_real.raw").write_bytes(b"\x00\x01")

    with pytest.raises(ValueError, match="expected 8 uint8 voxels"):
        assemble(manifest, gen_root, tpaqb_root, out_root, expected_shape=SHAPE)


def test_test95_wrapper_rejects_wrong_target_pore_count(tmp_path):
    manifest, gen_root, tpaqb_root, out_root = build_fixture(tmp_path)
    raw_path = next(tpaqb_root.rglob("crop_000_prob_tp_aqb.raw"))
    npz_path = next(tpaqb_root.rglob("crop_000_prob_tp_aqb.npz"))
    wrong = np.asarray([1, 1, 0, 0, 0, 0, 0, 0], dtype=np.uint8).reshape(SHAPE)
    wrong.tofile(raw_path)
    metadata = json.dumps({"segment": {"method": "TP-AQB"}, "repair": {}})
    np.savez(
        npz_path,
        seg=wrong,
        target_porosity=0.11,
        achieved_porosity=float(wrong.mean()),
        seed=1000,
        metadata_json=np.asarray(metadata),
    )

    with pytest.raises(ValueError, match="pore count 2 != target 1"):
        assemble(manifest, gen_root, tpaqb_root, out_root, expected_shape=SHAPE)
