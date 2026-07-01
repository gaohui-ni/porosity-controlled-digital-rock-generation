# Submission Checklist

This checklist is intended for the final Computers & Geosciences / Energy & Fuels repository audit before manuscript submission.

## Reviewer Entry Points

- [x] Official reproduction command is documented in `README.md`.
- [x] `run_pipeline.py` is the only reviewer-facing reproduction entry point.
- [x] `scripts/run_pipeline.py` is documented as an internal modular implementation.
- [x] `--dry-run` is documented as a pipeline logic check, not a substitute for full training.

## Lightweight Verification

- [x] Synthetic demo can run without raw micro-CT data.
- [x] Notebook tutorials are available under `notebooks/tutorials/`.
- [x] Unit tests cover porosity matching, porosity calculation, S2 output shape, and pipeline dry-run.
- [x] GitHub Actions runs lightweight tests and dry-run checks.

## Data Availability

- [x] Primary laboratory sandstone micro-CT data are referenced by Mendeley Data DOI.
- [x] Synthetic/demo data are distinguished from manuscript-scale raw data.
- [x] Fontainebleau validation data are identified as external data and are not redistributed.
- [x] Data availability details are documented in `docs/data_availability.md`.

## Figure and Metric Reproduction

- [x] Figure-to-code mapping is documented in `docs/figure_mapping.md`.
- [x] Command-level figure reproduction notes are documented in `docs/figure_reproduction.md`.
- [x] `scripts/reproduce_figures.py` provides a reviewer-facing figure reproduction helper.
- [x] Metrics are summarized through `src/metrics/summarize_all.py`.

## Final Checks Before Submission

- [ ] Run the full GPU workflow with the released Mendeley dataset.
- [ ] Inspect generated `results/results_summary.json`, `results/summary.json`, and `results/results_summary.csv`.
- [ ] Confirm final manuscript figure numbers match `docs/figure_mapping.md`.
- [ ] Confirm Mendeley Data version and DOI are final in README, manuscript, and `docs/data_availability.md`.
- [ ] Tag a release after final manuscript/code synchronization.

Note: Precomputed summary files are intentionally not bundled before the final full run. They should be generated from the released data and configuration.
