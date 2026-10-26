import csv

import numpy as np

from scripts.evaluate_spatial_test95 import EXPECTED_GROUPS, assemble


def test_test95_wrapper_joins_by_crop_id_and_stages_actual_tpaqb(tmp_path):
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
            real = np.asarray([0, 1, 0, 1], dtype=np.uint8)
            fixed = np.asarray([0, 0, 0, 1], dtype=np.uint8)
            topn = np.asarray([0, 1, 0, 0], dtype=np.uint8)
            tpaqb = np.asarray([1, 1, 0, 0], dtype=np.uint8)
            names = {
                "real_path": f"{crop_id}_real.raw",
                "pre_aqb_path": f"{crop_id}_off.raw",
                "post_aqb_path": f"{crop_id}_on.raw",
            }
            real.tofile(gen_root / names["real_path"])
            fixed.tofile(gen_root / names["pre_aqb_path"])
            topn.tofile(gen_root / names["post_aqb_path"])
            tpaqb.tofile(group_dir / f"{crop_id}_prob_tp_aqb.raw")
            rows.append(
                {
                    "crop_id": crop_id,
                    "group": group,
                    "phi_real": "0.5",
                    "phi_post_aqb": "0.25",
                    **names,
                }
            )
            index += 1

    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(reversed(rows))

    augmented, summaries = assemble(manifest, gen_root, tpaqb_root, out_root)

    assert len(augmented) == 95
    assert len(summaries) == 4
    assert augmented[0]["crop_id"] == "crop_000"
    assert float(augmented[0]["phi_tpaqb"]) == 0.5
    assert (out_root / "stage" / "tpaqb" / "phi0p11" / "crop_000.raw").is_file()
    assert (out_root / "stage" / "topn" / "phi0p11" / "crop_000.raw").is_file()
