# Reproduction Guide

This guide gives the shortest reviewer-facing path through the repository.

## Quick check

```bash
python run_demo.py
python run_pipeline.py --mode spatial --config configs/main.yaml --dry-run
```

The demo is synthetic. The dry run prints the five `spatial` steps without
executing them: VQ-VAE training, latent statistics, DDPM training, configured
condition-sweep generation, and TP-AQB.

## A. Train from scratch

```bash
python run_pipeline.py --mode spatial --config configs/main.yaml
```

This requires the published sandstone micro-CT data and a CUDA-capable
environment. It does not invoke the released-weight Test-95 generator or the
complete Figs. 8–10 scientific evaluation.

## B. Generate and stage the released Test-95 cohort

Use `scripts/generate_spatial_test95.py`, then `scripts/apply_tp_aqb.py`, then
`scripts/evaluate_spatial_test95.py`, as documented in `README.md` and
`docs/figure_reproduction.md`. The staging wrapper validates cohort identity
and prepares the real/TP-AQB trees; it does not calculate all final metrics.

## C. Compute metrics and reproduce figures

Run the explicit Fig. 8–10 commands in `docs/figure_reproduction.md`. These
commands calculate the spatial, pore-network, topology, and permeability
outputs from the staged cohort.

After the metric outputs exist, this command creates an inventory:

```bash
python src/metrics/summarize_all.py --root results
```

It writes `results/results_summary.json`, `results/summary.json`, and
`results/results_summary.csv`, but does not compute scientific metrics or
render the manuscript figures.

The historical compatibility validation additionally requires three external
ANU Fontainebleau volumes, which are not redistributed:

```bash
python run_pipeline.py --mode fontainebleau --config configs/main.yaml
```

That compatibility command is not asserted to reproduce Fuel Fig. 11 or
Table 3.
