# Reproduction Guide

This guide gives the shortest reviewer-facing path through the repository.

## Quick Check

```bash
python run_demo.py
```

This runs the synthetic 64^3 workflow, checks quantile-based porosity control, and writes lightweight summaries under `examples/`.

## Inspect the Full Workflow

```bash
python run_pipeline.py --mode full --config configs/main.yaml --dry-run
```

This prints the data preparation, training, generation, evaluation, and summary commands without launching GPU training.

## Full Manuscript-Scale Workflow

```bash
python run_pipeline.py --mode full --config configs/main.yaml
```

This requires the published sandstone micro-CT data, a CUDA-capable PyTorch installation, and enough memory for 256^3 training.

## Result Summary

```bash
python src/metrics/summarize_all.py --root results
```

This writes `results/results_summary.json` and `results/results_summary.csv`.
