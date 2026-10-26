# Fuel final reproduction audit

Audit date: 2026-10-26  
Repository: `gaohui-ni/porosity-controlled-digital-rock-generation`  
Branch: `fix/fuel-reproduction-final-audit`  
Required baseline present: `688d22ac16` (`PASS`)  
Final audited implementation commit: `fa92afbb840e087f606b427636f2aee283e2c201`

This was a code, documentation, and provenance audit. It did not retrain a
model, regenerate the 95 formal `256^3` samples, rerun PNM/permeability, modify
released tensors, alter crop IDs/seeds, or publish internal result CSV files.

## Status summary

| Area | Status | Conclusion |
| --- | --- | --- |
| Test-95 RAW/NPZ assembly | `PASS` | Actual TP-AQB is staged once per crop as voxel-identical RAW and NPZ (`seg`) artifacts; top-N retains its historical identity. |
| Test-95 identity gates | `PASS` | 95 used crop IDs, group sizes 8/29/39/19, unique IDs, target/seed/shape/pore-count checks, and missing/corrupt/duplicate rejection are implemented. |
| Fuel Fig. 8 command | `PASS` | Uses TP-AQB RAW with `--gen_use raw`; the Fuel statistic is the unnormalized `S2,d(r)=<I(x)I(x+r e_d)>`. |
| Fuel Fig. 9 command | `PASS` | PNM reads `stage/tpaqb_npz` and explicitly selects NPZ key `seg`. |
| Fuel Fig. 10 command | `PASS` | Permeability continues to read TP-AQB RAW from `stage/tpaqb`. |
| Fuel documentation alignment | `PASS` | Active documents use Fuel Figures 1–11, identify Fig. 11 as Fontainebleau permeability, and label `plot_all.py` as a synthetic demo. |
| Release metadata | `PASS` | `v0.2.0` is identified as a release candidate; the latest formal GitHub Release is `v0.1.1`. No Release was created. |
| VQ-VAE exact training history | `BLOCKED` | Manuscript/config says 15 epochs; released checkpoint embeds `epochs_vae=80` and selected zero-based epoch 12; no complete original training log was found. |
| Fontainebleau Fig. 11/Table 3 provenance | `BLOCKED` | A local 20+20 permeability table exists, but its companion notes conflict with it and the epoch-75/EXT3X checkpoint and complete manifests are absent. |
| Full GPU Test-95 regeneration | `NOT_RUN` | Prohibited by scope and unnecessary for this repair. |
| Full PNM/permeability recomputation | `NOT_RUN` | Prohibited by scope; only command/data-format contracts were validated. |
| Executed checks | `PASS` | No executed check failed. |

## Modified files and repairs

### Evaluation and tests

- `scripts/evaluate_spatial_test95.py`
  - requires formal default shape `256 x 256 x 256`;
  - validates manifest group counts and duplicate IDs;
  - validates real/fixed/top-N RAW shape, binary values, crop identity, and
    recorded porosity;
  - validates probability-field target and seed metadata;
  - requires an adjacent actual TP-AQB RAW/NPZ pair;
  - requires NPZ `seg`, target, achieved porosity, seed, and TP-AQB
    segment/repair provenance;
  - verifies RAW and NPZ voxel identity and exact target pore count;
  - writes `stage/tpaqb` and `stage/tpaqb_npz` without treating them as two
    statistical samples;
  - preserves `phi_post_aqb` as top-N and adds separate `phi_tpaqb`,
    `tpaqb_path`, and `tpaqb_npz_path` fields.
- `tests/test_evaluate_spatial_test95.py`
  - creates a 95-crop synthetic cohort with group sizes 8/29/39/19;
  - covers successful RAW/NPZ staging, no double count, mismatch, missing
    provenance, duplicate artifacts/IDs, missing input, corrupt shape, and
    incorrect pore count.
- `tests/test_final_release_integrity.py`
  - locks the Fuel Fig. 8/9/10 input-format contract.
- `scripts/evaluate_s2_lineal_edt.py`
  - clarifies that normalization is optional and must be omitted for Fuel
    Fig. 8.
- `scripts/run_pipeline.py`
  - renames the old `fig14_fontainebleau.csv` output to the neutral
    `fontainebleau_compatibility_permeability.csv`.

### Documentation and metadata

- `docs/figure_reproduction.md`, `docs/figure_mapping.md`: corrected TP-AQB
  RAW/NPZ routes, unnormalized S2, and Figures 1–11 mapping.
- `README.md`, `docs/reproduction.md`, `docs/reproduction_guide.md`,
  `docs/reproducibility.md`, `docs/journal_compliance.md`,
  `docs/github_workflow_for_cg.md`: made Fuel the active target, separated
  formal and demo commands, and marked Fontainebleau mode as compatibility.
- `docs/methodological_novelty.md`, `docs/code_availability.md`: removed the
  obsolete current-Fuel Figures 12–14 mapping.
- `docs/model_manifest.md`: corrected the release manifest count from 13 to
  14.
- `CITATION.cff`: distinguishes the `v0.2.0` candidate metadata from formal
  GitHub Release `v0.1.1`.
- `results/source_data/README.md`: uses the neutral Fontainebleau compatibility
  filename and does not claim it is Fuel Fig. 11 source data.
- `docs/spatial_vqaware_workflow.md`, `docs/fuel_repro_status.md`: no longer
  infer the VQ-VAE's actual training length from the current config.
- `docs/fontainebleau_provenance_audit.md`: records the newly observed 20+20
  table while preserving the blocked verdict caused by contradictory notes and
  missing provenance.

## Tests actually executed

| Command/check | Status | Result |
| --- | --- | --- |
| `pytest -q -p no:cacheprovider` | `PASS` | 55 passed in 8.64 s on the final implementation tree. |
| `python scripts/verify_spatial_release.py --check-tensors` | `PASS` | 14 files, 100 cohort rows, 95 used samples; tensor metadata validated. Executed with the installed PyTorch runtime at `D:/python/python.exe`. |
| `git lfs pull` | `PASS` | Completed before tensor/hash verification. |
| `sha256sum -c savedmodels/spatial_vqaware_v2/MANIFEST.sha256` | `PASS` | All 14 entries reported `OK`. |
| `python run_pipeline.py --mode spatial --config configs/main.yaml --dry-run` | `PASS` | Emitted the 15-epoch retraining config, 40-epoch DDPM config, calibrated sampling, and TP-AQB commands without training. |
| Python compilation of modified scripts | `PASS` | `evaluate_spatial_test95.py`, `evaluate_s2_lineal_edt.py`, and `run_pipeline.py` compiled. |
| PNM CLI format check | `PASS` | Installed PoreSpy/OpenPNM environment exposes `--gen-root` and `--gen-npz-key`; docs use `tpaqb_npz`/`seg`. |
| `CITATION.cff` YAML parse | `PASS` | Parsed successfully. |
| Active-document stale-name scan | `PASS` | No active current-Fuel Figures 12–14, old `fig14_fontainebleau.csv`, or 13-entry manifest claim remains. Historical dated audit text was retained as history. |
| `git diff --check` | `PASS` | No whitespace errors. |

The first attempt to run tensor verification with the lightweight Anaconda
Python returned `ModuleNotFoundError: torch`; this was an environment mismatch,
not a repository failure. The required command was then rerun successfully
with the installed PyTorch runtime. The PNM evaluator was not run on volumes;
only its CLI/data-format interface was checked using the installed
`pore_eval` environment.

## VQ-VAE epoch audit

Status: `BLOCKED`.

Verified evidence:

- Fuel manuscript/current retraining config: VQ-VAE 15 epochs, DDPM 40 epochs;
- released `ae_best.pth`: selected zero-based epoch 12 and embedded
  `epochs_vae=80`;
- released DDPM: selected zero-based epoch 34 and embedded 40-epoch setting;
- `optional_precalibration/run_cfg.json`: also contains `epochs_vae=80` and
  `epochs_ddpm=40`;
- no complete original VQ-VAE training log was found in the repository or the
  searched local project/Desktop evidence roots.

Therefore 80 may have been a configured maximum, but the available evidence
does not prove that interpretation or prove that the released VQ-VAE stopped
after 15 epochs. No checkpoint or manuscript value was altered.

## Fontainebleau sample-count and provenance audit

Status: `BLOCKED` for direct reproduction of Fuel Fig. 11/Table 3.

The formal Fuel scope is `0.2041` (training reference), `0.1766`, and `0.1289`;
`0.0829` is exploratory and excluded from the current main result.

| Condition | Local generation/structure record | Local permeability record observed | Confirmed as exact Fuel Fig. 11/Table 3 lineage? |
| --- | --- | --- | --- |
| 0.2041 | 20 REAL + 20 GEN | 20 REAL + 20 GEN, non-null `Kgeom` rows | `BLOCKED` |
| 0.1766 | 20 REAL + 20 GEN | 20 REAL + 20 GEN, non-null `Kgeom` rows | `BLOCKED` |
| 0.1289 | 20 REAL + 20 GEN | 20 REAL + 20 GEN, non-null `Kgeom` rows | `BLOCKED` |

Why the formal mapping remains blocked:

- the local combined permeability table has 20+20 rows per condition, but its
  companion README says external permeability came from an older 5+5 source,
  reports a 70-volume merge, and elsewhere says the 20+20 paired batch was not
  generated;
- the older referenced `EXT3X_REAL5_GEN5_PERMEABILITY.csv` was not found in the
  searched local roots;
- no released hash/manifest ties the local table to Fuel Fig. 11/Table 3;
- the named epoch-75 checkpoint, EXT3X implementation, complete crop/seed
  manifests, and exact figure/table source chain are not bundled;
- the released 80/150 Fontainebleau files are compatibility artifacts and were
  not substituted.

The audit therefore records that a 20+20 permeability table exists, but does
not promote it to the formal manuscript source. No internal raw result CSV was
added to the repository.

## Not-run items

- `NOT_RUN`: 95-volume diffusion generation (approximately 5.9 GB outputs).
- `NOT_RUN`: full S2 evaluation on the 95 formal volumes.
- `NOT_RUN`: full PNM and permeability calculations.
- `NOT_RUN`: end-to-end Fuel figure rendering.
- `NOT_RUN`: any VAE/DDPM training, fine-tuning, or TP-AQB ablation.
- `NOT_RUN`: creation of a GitHub Release.

## Remaining submission issues

- Main-sandstone Test-95 code/release path: no unresolved major repository
  defect found (`PASS`).
- VQ-VAE historical 15-versus-80 evidence: unresolved and explicitly
  disclosed (`BLOCKED`).
- Fontainebleau final checkpoint/sample/source-data chain: unresolved and
  explicitly disclosed (`BLOCKED`).
- Fuel Fig. 5 matching no-condition checkpoint: not bundled (`BLOCKED`).

These blocked provenance items prevent claims of exact numerical reproduction
for the affected results, but they are no longer hidden or replaced by
compatibility artifacts.

## Git diff summary

The final audited implementation commit changes 21 files with 406 insertions
and 167 deletions. It contains evaluation validation/staging logic, synthetic
tests, command-contract tests, neutral compatibility naming, and documentation
alignment. This report is the only additional file in the following audit
commit. The branch is intended for review through a Pull Request and must not
be automatically merged into `main`.
