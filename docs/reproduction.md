# Reproduction

Use this short path for review:

```bash
python run_demo.py
python run_pipeline.py --mode spatial --config configs/main.yaml --dry-run
python scripts/evaluate_all.py --config configs/main.yaml --dry-run
```

Use the public main-sandstone workflow on a CUDA workstation:

```bash
python run_pipeline.py --mode spatial --config configs/main.yaml
python src/metrics/summarize_all.py --root results
python scripts/plot_all.py
```

The second-version main-sandstone workflow can be reproduced through model
retraining using the released code, configuration, and public Mendeley Data
volume. Exact recreation of the reported 95-pair cohort additionally requires
the unbundled spatial checkpoints and frozen crop manifest. Run
`--mode fontainebleau` separately when its three external ANU volumes are
available. Because training and sampling are stochastic, exact generated
volumes are not guaranteed.
