# spatial_vqaware_v2 — R03 final weights and Test-95 manifests

This folder is a staging copy for the GitHub release. It contains the weights
and manifests read by the final 95-pair Test run (T1). It is **not** a training
run and does **not** contain generated volumes.

## Final sampling weight

- `R03_calibrated/calib_A_scale.pth` — **final sampling weight**.
  Its 68 model tensors are bitwise identical to
  `07_ablation/armB_vq_nocodebook/unet_best_noise.pth`; only `cfg.poro_scale`
  differs: `0.02 -> 0.011804`. No retraining.
- `R03_calibrated/ae_best.pth` — frozen VQ-VAE decoder.
- `R03_calibrated/latent_stats.npz` — latent mean/std.

R03 is a zero-training condition-gain calibration: armB's delivered-porosity
slope is 0.5902; the condition enters through one column of `cond_mlp.0`;
`m = 1 / 0.5902 = 1.69434`, and `poro_scale = 0.02 / m = 0.011804`.

## Optional pre-calibration source

- `optional_precalibration/unet_best_noise.pth` — armB source weights,
  `poro_scale=0.02`. **Not the final sampling weight.** Do not substitute it
  for `calib_A_scale.pth`; the T1 driver asserts `poro_scale == 0.011804`.
- `optional_precalibration/run_cfg.json` — armB run configuration.

## Manifests and provenance

- `R03_calibrated/calibration_definition.json`
- `R03_calibrated/r03_identity_gates.json`
- `R03_calibrated/r03_verdict_20seed.json`
- `test_cohort_v1/test_crop_porosity_all_100.csv` — 100 rows; 95 used, 5
  `target_group="none"` excluded; group sizes 8/29/39/19.
- `final_test_v2/final_test_seed_manifest.csv` — 95 rows.
- `final_test_v2/FINAL_TEST_PAIR_MANIFEST.csv` — 95 rows x 18 columns.
- `spatial_split.json` — original path `03_latent_stats/v1/spatial_split.json`.

## Hash calibers

- `MANIFEST.sha256` uses **file SHA256** (`sha256sum`).
- `calibration_definition.json`'s `source_weight_sha256=0df5e3d8...` is a
  state-dict content hash over sorted keys + concatenated float32 values
  without key names. It is **not** a file SHA256. The armB file SHA256 is
  `f3e68a21...`. The separate `45866df1453ecf4f` digest in `FINAL_REPORT.md`
  uses sorted key names + little-endian float32 tensor bytes.
- After `git lfs pull`, verify with:

```bash
sha256sum -c MANIFEST.sha256
```

Without `git lfs pull`, `.pth` files in a fresh clone are pointer files and the
checksums will not match.

The repository also provides a cross-platform verifier:

```bash
python scripts/verify_spatial_release.py --check-tensors
```

## Not included

- `S1.raw` — the main sandstone 800^3 volume. It is available from the
  published Mendeley Data DOI (`https://doi.org/10.17632/vp2yw9c7jj.1`) and is
  not duplicated in GitHub.
- `final_test_v2/gen/` — the 5.9 GB generated probability/OFF/ON volumes.
- `calib_B_col128.pth` / `calib_B_col128_m2.pth` — equivalence cross-checks,
  not read by the T1 driver.
- Fontainebleau v2 checkpoints — a separate track; select the checkpoint and
  cohort per figure/table before publishing.

## Code

The public, path-configurable Test-95 driver is
`scripts/generate_spatial_test95.py`. It loads this package and calls the frozen
sampling implementation in
`scripts/train_256_vqvae_ddpm_spatial_vqaware_v3_film.py`.

```bash
python scripts/generate_spatial_test95.py \
  --raw-path data/raw/Bei_800x800x800.raw \
  --device cuda
```

The generated `*_prob.npz` files can then be processed with
`scripts/apply_tp_aqb.py`. The driver writes new local manifests under
`outputs/spatial_test95/` and never overwrites the released reference
manifests in this directory.
