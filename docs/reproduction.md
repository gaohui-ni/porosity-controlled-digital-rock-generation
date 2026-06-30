# Reproduction

Use this short path for review:

```bash
python run_demo.py
python run_pipeline.py --mode full --config configs/main.yaml --dry-run
python scripts/evaluate_all.py --config configs/main.yaml --dry-run
```

Use the full workflow on a CUDA workstation:

```bash
python run_pipeline.py --mode full --config configs/main.yaml
python src/metrics/summary.py --root results
python scripts/plot_all.py
```

The full workflow requires the published sandstone micro-CT data and trained checkpoints generated locally.
