# Figure Reproduction Map

This file maps manuscript-style result groups to the commands and output folders used in this repository. Exact figure numbers can be adjusted to match the final accepted manuscript layout.

## Overview

The main configuration file is:

```bash
configs/main.yaml
```

The full manuscript-scale workflow is reproduced through the official top-level entry point:

```bash
python run_pipeline.py --mode full --config configs/main.yaml
```

For only the main laboratory sandstone figures, use:

```bash
python run_pipeline.py --mode main --config configs/main.yaml
```

For only the independent Fontainebleau validation figures, use:

```bash
python run_pipeline.py --mode fontainebleau --config configs/main.yaml
```

For generation and evaluation from final supplied checkpoints without retraining, use:

```bash
python run_pipeline.py --mode final --config configs/main.yaml
```

The full workflow can be inspected without running training:

```bash
python run_pipeline.py --mode full --config configs/main.yaml --dry-run
```

For a lightweight figure-output sanity check:

```bash
python scripts/reproduce_figures.py --config configs/main.yaml
```

This helper validates the output path and can generate a synthetic demo slice/two-point-probability figure. Manuscript-scale figures require the full evaluation outputs listed below.

## Spatial Statistics Figures

Use these outputs for directional two-point probability function `S2`, lineal-path, and EDT pore-size figures.

```bash
python scripts/evaluate_s2_lineal_edt.py \
  --real_root data/real256_sets_from_S1_strict \
  --gen_root data/generated_phi_sets \
  --out_root results/fig_s2 \
  --targets 0.11 0.12 0.13 0.14 0.15 \
  --r_max 128
```

Output folder:

```text
results/fig_s2/
```

## Permeability and Voxel-Metric Figures

Use this output for porosity, connected pore fraction, surface density, EDT summary, and OpenPNM permeability tables.

```bash
python scripts/evaluate_voxel_and_perm.py \
  --input_root data/real256_sets_from_S1_strict \
  --output_csv results/tables/real_voxel_perm.csv \
  --group_name real \
  --shape 256 256 256 \
  --voxel_size 3.5e-6 \
  --recursive \
  --extensions .raw

python scripts/evaluate_voxel_and_perm.py \
  --input_root data/generated_phi_sets \
  --output_csv results/tables/generated_voxel_perm.csv \
  --group_name gen \
  --shape 256 256 256 \
  --voxel_size 3.5e-6 \
  --recursive \
  --extensions .npz

python scripts/compare_metric_tables.py \
  --real-csv results/tables/real_voxel_perm.csv \
  --gen-csv results/tables/generated_voxel_perm.csv \
  --output-csv results/tables/permeability_comparison.csv \
  --source-data-csv results/source_data/fig8_permeability.csv
```

Output files:

```text
results/tables/real_voxel_perm.csv
results/tables/generated_voxel_perm.csv
results/tables/permeability_comparison.csv
results/source_data/fig8_permeability.csv
results/fig_perm/
```

## Topology and Pore-Network Figures

Use these outputs for coordination number, Euler characteristic, and pore-network descriptor figures.

```bash
python scripts/evaluate_coordination_euler.py \
  --real-root data/real256_sets_from_S1_strict \
  --gen-root data/generated_phi_sets \
  --out-root results/fig_pnm
```

The complete Fontainebleau Real-Gen workflow, including all four real volumes,
is reproduced with:

```bash
python run_pipeline.py --mode fontainebleau --config configs/main.yaml
```

For the six-panel pore-network analysis:

```bash
python scripts/evaluate_pore_network_6panel.py --config configs/main.yaml
```

This PNM six-panel step is included in `python run_pipeline.py --mode main --config configs/main.yaml` and therefore also in `--mode full`. Before running the six-panel script directly, check the configuration block at the top of the file and adjust local paths. The manuscript-scale voxel size is read from `configs/main.yaml` as `data.voxel_size_m = 3.5e-6`.

## Fontainebleau Validation Figures

See `docs/fontainebleau_protocol.md` for the validation protocol. The default output folder is:

```text
results/fig_fontainebleau/
```

The Fontainebleau validation is included in:

```bash
python run_pipeline.py --mode fontainebleau --config configs/main.yaml
python run_pipeline.py --mode full --config configs/main.yaml
```

The prepared Fontainebleau raw volume must exist at the path configured in `configs/fontainebleau_config.yaml`.

## Summary Tables

After generating results, write a compact manifest:

```bash
python src/metrics/summarize_all.py --root results
```

This writes:

```text
results/results_summary.json
results/results_summary.csv
```

Precomputed summary files are not bundled before the final full run. They should be generated from the released data and configuration.
