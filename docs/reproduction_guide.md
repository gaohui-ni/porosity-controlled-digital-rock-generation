# Reproduction Guide

This guide gives the shortest reviewer-facing path through the repository.

## Quick Check

```bash
python run_demo.py
```

This runs the synthetic 64^3 workflow, checks quantile-based porosity control, and writes lightweight summaries under `examples/`.

## Inspect the Second-Version Workflow

```bash
python run_pipeline.py --mode spatial --config configs/main.yaml --dry-run
```

This prints the data preparation, training, generation, evaluation, and summary commands without launching GPU training.

## Main-Sandstone Manuscript Workflow

```bash
python run_pipeline.py --mode spatial --config configs/main.yaml
```

This requires the published sandstone micro-CT data, a CUDA-capable PyTorch installation, and enough memory for 256^3 training.

The historical compatibility validation additionally requires access to three
external ANU Fontainebleau volumes, which are not redistributed. This command
is not asserted to reproduce Fuel Fig. 11 or Table 3:

```bash
python run_pipeline.py --mode fontainebleau --config configs/main.yaml
```

## Result Summary

```bash
python src/metrics/summarize_all.py --root results
```

This writes `results/results_summary.json` and `results/results_summary.csv`.
