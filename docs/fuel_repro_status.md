# Fuel-aligned reproducibility status

Audit implementation date: 2026-10-26. This file reports verification status;
it does not claim that a command was run when only its interface was tested.

## Passed

- Frozen cohort identity: 100 manifest rows, 95 used crop IDs, no duplicate
  used IDs, and group sizes 8/29/39/19.
- Release verification: `scripts/verify_spatial_release.py` passed for all 14
  manifest entries after final metadata edits.
- Git LFS materialization and raw-byte verification: `git lfs pull` followed
  by `sha256sum -c MANIFEST.sha256` passed for all 14 entries.
- Focused naming, manifest, pipeline, TP-AQB, Test-95 wrapper, and release
  tests: 29 passed.
- Full repository test suite: 47 passed.
- Synthetic Test-95 integration: all 95 small binary fixtures were joined by
  `crop_id`; actual TP-AQB and historical top-N branches were staged under
  distinct canonical trees.
- Public-text path scan found no `/home/hipeson` or private Windows user paths
  outside checkpoint/NPZ binaries.

## Failed

- None in the executed code/checksum/test checks above.

## Not run

- Full 95-volume diffusion sampling and downstream S2/PNM/permeability
  evaluation. The approximately 5.9 GB generated volume set was intentionally
  not regenerated during this repository repair.
- End-to-end Fuel figure rendering. Method/branch mapping and commands were
  corrected, but figures were not claimed as regenerated.

## Blocked by missing provenance or artifacts

- Exact VQ-VAE training history. The Fuel manuscript/config states 15 epochs;
  the released checkpoint embeds an 80-epoch configuration and stores best
  epoch 12 (zero-based). The complete original training log is absent.
- Fuel Fig. 5 no-condition ablation from frozen weights: the matching
  no-condition checkpoint is not bundled.
- Fuel Fig. 11 / Table 3 Fontainebleau reproduction: the candidate epoch-75
  checkpoint, EXT3X implementation, complete seed/crop manifests, and exact
  source tables are not bundled. The historical 80/150 compatibility weights
  are not accepted as a substitute.

## Re-run commands

```bash
python scripts/verify_spatial_release.py
git lfs pull
(cd savedmodels/spatial_vqaware_v2 && sha256sum -c MANIFEST.sha256)
pytest -q
python scripts/evaluate_spatial_test95.py --help
```
