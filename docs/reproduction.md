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

The complete workflow can be reproduced through model retraining using the released code, configurations, and public raw data. Because training and sampling are stochastic, exact generated volumes and numerically identical outputs are not guaranteed.
