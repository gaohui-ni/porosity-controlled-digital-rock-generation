# Reproduction

Use this short path for a lightweight review:

```bash
python run_demo.py
python run_pipeline.py --mode spatial --config configs/main.yaml --dry-run
python scripts/evaluate_all.py --config configs/main.yaml --dry-run
```

The manuscript workflow has three distinct phases.

## A. Train from scratch

```bash
python run_pipeline.py --mode spatial --config configs/main.yaml
```

This command retrains the spatial VQ-VAE and latent DDPM, computes latent
statistics, generates the configured condition sweep, and applies TP-AQB. It
does not generate the formal released-weight Test-95 cohort and does not
calculate all Fuel Figs. 8–10 metrics.

## B. Generate and stage Test-95 with released weights

```bash
python scripts/generate_spatial_test95.py \
  --raw-path data/raw/Bei_800x800x800.raw \
  --device cuda
python scripts/apply_tp_aqb.py \
  --input-root outputs/spatial_test95/gen \
  --output-root data/generated_phi_sets_spatial_test95 \
  --config configs/tp_aqb_v2.yaml
python scripts/evaluate_spatial_test95.py \
  --manifest outputs/spatial_test95/FINAL_TEST_PAIR_MANIFEST.csv \
  --gen-root outputs/spatial_test95/gen \
  --tpaqb-root data/generated_phi_sets_spatial_test95 \
  --out-root results/spatial_test95
```

The final command validates and stages the cohort; it is not the full
scientific metric evaluation.

## C. Evaluate metrics and reproduce figures

Run the Fig. 8–10 metric commands in `docs/figure_reproduction.md` on the
staged Test-95 trees. After those commands have produced CSV/JSON outputs,
this optional inventory command summarizes which files already exist:

```bash
python src/metrics/summarize_all.py --root results
```

`summarize_all.py` records paths, row counts, and column names. It does not
compute scientific metrics or reproduce figures. `scripts/plot_all.py` is a
synthetic plotting demo, not a formal Fuel figure command.

The Git-LFS spatial package includes the calibrated checkpoint, frozen
100-crop table, and 95-sample seed and pair manifests; generated volumes are
regenerated locally. `--mode fontainebleau` is a historical compatibility path
and is not accepted as direct Fuel Fig. 11 or Table 3 reproduction. Because
training and sampling are stochastic, exact generated volumes are not
guaranteed.
