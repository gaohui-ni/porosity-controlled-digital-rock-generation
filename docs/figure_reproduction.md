# Figure Reproduction Map

This file maps manuscript-style result groups to the commands and output folders used in this repository. Exact figure numbers can be adjusted to match the final accepted manuscript layout.

## Overview

The main configuration file is:

```bash
configs/experiment_main.yaml
```

The full workflow can be inspected without running training:

```bash
python scripts/run_pipeline.py --mode full --config configs/experiment_main.yaml --dry-run
```

## Spatial Statistics Figures

Use these outputs for S2, lineal-path, and EDT pore-size figures.

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
  --input_root data/generated_phi_sets \
  --output_csv results/tables/generated_voxel_perm.csv \
  --group_name gen \
  --shape 256 256 256 \
  --recursive
```

Output files:

```text
results/tables/generated_voxel_perm.csv
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

For the six-panel pore-network analysis:

```bash
python scripts/evaluate_pore_network_6panel.py
```

Before running the six-panel script, check the configuration block at the top of the file and adjust local paths.

## Fontainebleau Validation Figures

See `docs/fontainebleau_protocol.md` for the validation protocol. The default output folder is:

```text
results/fig_fontainebleau/
```

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
