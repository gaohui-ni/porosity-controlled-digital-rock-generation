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
```

`scripts/plot_all.py` is a synthetic plotting demo, not a formal Fuel figure
reproduction command.

The second-version main-sandstone workflow can be reproduced through model
retraining using the released code, configuration, public Mendeley Data
volume, and the Git-LFS spatial package. The package includes the calibrated
checkpoint, frozen 100-crop table, and 95-sample seed and pair manifests;
generated volumes are regenerated locally. `--mode fontainebleau` is a
historical compatibility path and is not accepted as direct Fuel Fig. 11 or
Table 3 reproduction. Because training and sampling are stochastic, exact
generated volumes are not guaranteed.
