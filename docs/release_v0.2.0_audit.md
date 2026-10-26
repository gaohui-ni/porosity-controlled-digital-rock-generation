# v0.2.0 pre-release repair and acceptance audit

> **Historical audit notice.** This document records the repository state and
> acceptance checks as of 2026-10-20. It is not the current training-provenance
> authority. See `FINAL_FUEL_REPRO_AUDIT.md` for the current status, including
> the still-unresolved VQ-VAE 15-versus-80 epoch conflict.

Audit date: 2026-10-20. Baseline commit:
`6da2bab4bfa704caf14e9530cc76a8c12941b688`.

## Scope and non-mutation guarantees

This branch repairs documentation, release checks, CI, tests, line-ending
policy, and byte manifests. It does not train a model, generate a rock, alter
a frozen checkpoint, modify a delivered RAW volume, change any of the 95 crop
IDs/seeds/pairs, or change a scientific result.

## Repairs

| Area | Previous problem | Repair | Experimental result affected? |
| --- | --- | --- | --- |
| spatial `MANIFEST.sha256` | three CSV hashes described Windows CRLF bytes and failed on LF checkouts | use repository LF hashes and enforce LF for the three formal CSVs | No |
| data availability | `data/README.md` said the main data were unavailable | cite Mendeley DOI and separate ANU redistribution limits | No |
| Fontainebleau parents | one README said four volumes | document exactly three paper volumes and historical filename labels | No |
| Test-95 entry point | released-weight generation, retraining and compatibility inference were easy to confuse | document three separate commands and exact formal inputs | No |
| delivered schedule | default/history fields could be read as 80/150 | record actual 15/40 schedule, selected epochs and validation/time values | No |
| spatial package status | called a staging copy | identify it as delivered package and document LFS/checksum procedure | No |
| Fig. 3 | implied a real reference segmentation input | identify the generated fixed-threshold reference and forbidden real controls | No |
| Fig. 5 | implied full released-weight ablation reproducibility | disclose missing FiLM-OFF checkpoint and required retraining | No |
| TP-AQB config | named unpublished experiment-side files as if shipped | label historical files and map the formal code to `src/sampling/tp_aqb.py` | No |
| CI/tests | duplicated full dry-run; omitted spatial dry-run and release verification | add LFS, tensor/release check, spatial dry-run and spatial assertions | No |
| version metadata | retained v0.1.1 | prepare v0.2.0 metadata; no tag or GitHub Release is created | No |
| Fontainebleau provenance | public scope was over-broad | disclose missing formal artifacts and the 20/5 conflict | No |

## Changed-file inventory

| Path | Repair |
| --- | --- |
| `.gitattributes` | enforce LF for the three byte-verified release CSVs |
| `.github/workflows/tests.yml` | fetch LFS, verify tensors, add spatial dry-run, remove duplicate full dry-run |
| `CITATION.cff` | prepare v0.2.0/date metadata; no tag or Release created |
| `README.md` | separate Test-95 generation, spatial retraining and compatibility inference; correct model layout |
| `configs/fontainebleau_config.yaml` | document three-rock and provenance boundary without changing values |
| `configs/tp_aqb_v2.yaml` | map the published TP-AQB implementation and label missing historical files |
| `data/README.md`, `data/fontainebleau/README.md` | correct Mendeley/ANU availability and three-parent description |
| `docs/spatial_vqaware_workflow.md` | add delivered 15/40 schedule, selected epochs, losses and run times |
| `docs/figure_mapping.md`, `docs/figure_reproduction.md` | fix Fig. 3 inputs, Fig. 5 scope and Fontainebleau limits |
| `docs/fontainebleau_protocol.md`, `docs/fontainebleau_provenance_audit.md` | disclose 20/5 evidence conflict and missing formal bundle |
| `docs/model_manifest.md`, `docs/code_availability.md`, `docs/journal_compliance.md` | distinguish formal spatial and compatibility weights and state reproducibility limits |
| `savedmodels/spatial_vqaware_v2/README.md` | replace staging wording and add cross-platform LFS verification |
| `savedmodels/spatial_vqaware_v2/MANIFEST.sha256` | update three LF CSV hashes and changed package-README hash |
| three formal release CSVs | line-ending normalization only; rows, fields and values unchanged |
| `tests/test_final_release_integrity.py` | retain compatibility assertions and add spatial 15/40, release-path and hash assertions |

Every change above is documentation, release metadata, CI or testing. None
changes a scientific output or frozen tensor.

## Acceptance status

Completed acceptance checks:

- `git diff --check`: passed;
- `python scripts/verify_spatial_release.py --check-tensors`: passed, 13/13
  files, 100 cohort rows, 95 used samples;
- LFS identity: all four spatial model/statistics objects match the recorded
  SHA256 OIDs (`ae_best.pth`, `calib_A_scale.pth`, `latent_stats.npz`, and
  `unet_best_noise.pth`);
- `python run_pipeline.py --mode spatial --config configs/main.yaml --dry-run`:
  passed and emitted 15 VAE / 40 DDPM epochs, FiLM, calibrated sampling and
  TP-AQB v2 commands without training;
- `python -m pytest tests -q`: 44 passed; only the pre-existing local
  `.pytest_cache` permission warning was emitted;
- documented release paths: checked and present.

## Submission consistency

- Main sandstone: formal second-version weights and 100/95 manifests are
  released and verifiable.
- Training implementation: the public spatial configuration encodes VQ-VAE
  15 epochs, DDPM 40 epochs, FiLM conditioning and TP-AQB v2.
- Fontainebleau: code and three-volume protocol are present, but final figure
  provenance is not fully released or internally unambiguous.
- Code Availability: must state that main-sandstone Test-95 is reproducible
  from released artifacts while final Fontainebleau figures have a disclosed
  artifact/provenance limitation.

## Final state

`BLOCKED` pending resolution of the Fontainebleau scientific-provenance chain.
The code/document/CI repairs may be merged independently, but the repository
must not be described as fully READY for submission while that conflict
remains.
