# Reproduction

Use this short path for review:

```bash
python run_demo.py
python run_pipeline.py --mode full --config configs/main.yaml --dry-run
python scripts/evaluate_all.py --config configs/main.yaml --dry-run
```

Use the public main-sandstone workflow on a CUDA workstation:

```bash
python run_pipeline.py --mode main --config configs/main.yaml
python src/metrics/summarize_all.py --root results
python scripts/plot_all.py
```

The main-sandstone workflow can be reproduced through model retraining using
the released code, configuration, and public Mendeley Data volume. The complete
`full` workflow additionally requires access to the external ANU Fontainebleau
volumes, which are not redistributed. Because training and sampling are
stochastic, exact generated volumes and numerically identical outputs are not
guaranteed.
