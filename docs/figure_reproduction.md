# Figure Reproduction Map

This document follows the current submitted-manuscript numbering for Figures 1–9.

## Overview

The main configuration file is:

```bash
configs/main.yaml
```

The complete two-dataset workflow is run through the official top-level entry
point. It requires the public main-sandstone volume and access to the external
ANU Fontainebleau volumes:

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

This helper runs the synthetic demo by default and validates the output path.
It does not start manuscript-scale training unless `--run-full-pipeline` is
provided explicitly. Use `--skip-pipeline` to plot from existing outputs.
Manuscript-scale figures require the full evaluation outputs listed below.

## Fig. 5: Slice-Wise Porosity

```bash
python scripts/plot_slice_porosity.py \
  --real data/real256_sets_from_S1_strict/phi0p15/REAL_SAMPLE.raw \
  --generated data/generated_phi_sets/phi0p15/GENERATED_SAMPLE.npz \
  --shape 256 256 256 \
  --target 0.15 \
  --output results/figures/fig5_slice_porosity.png
```

Replace the two sample placeholders with the representative files used in the
manuscript. The command writes both the PNG and an adjacent CSV locally.

## Fig. 6 and Supplementary Spatial Statistics

Main-text Fig. 6 contains four panels: the `X`, `Y`, and `Z` directional
two-point correlation functions `S2(r)` as panels (a-c), followed by the `S/V`
comparison as panel (d). The direction-averaged `R = mean(X, Y, Z)` curve is
retained for the Supplementary Material and is not part of main-text Fig. 6.

Use these outputs for directional `S2`, lineal-path, and EDT pore-size figures:

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

Within each porosity folder, `s2_X_*`, `s2_Y_*`, and `s2_Z_*` provide the
main-text directional curves. `s2_R_*` is the supplementary direction-averaged
curve.

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

For the extended six-panel pore-network diagnostic analysis:

This extended diagnostic output is not identical to manuscript Fig. 7.
Manuscript Fig. 7 combines pore radius, throat radius, throat shape factor,
EDT radius, and coordination number from the corresponding evaluation outputs.
Manuscript Fig. 7 uses the `phi0p15` group and excludes throat length,
tortuosity, and Euler-characteristic outputs.

```bash
python scripts/evaluate_pore_network_6panel.py --config configs/main.yaml
```

This PNM six-panel step is included in `python run_pipeline.py --mode main
--config configs/main.yaml` and therefore also in `--mode full`. Verify paths
and target values in `configs/main.yaml`; modification of the Python source
file is not required. The manuscript-scale voxel size is read from
`configs/main.yaml` as `data.voxel_size_m = 3.5e-6`.

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
