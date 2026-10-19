# Figure Reproduction Map

## Workflow selection

The second-version main-sandstone workflow is:

```bash
python run_pipeline.py --mode spatial --config configs/main.yaml
```

Inspect its ordered commands without launching computation:

```bash
python run_pipeline.py --mode spatial --config configs/main.yaml --dry-run
```

The independently trained Fontainebleau experiment is:

```bash
python run_pipeline.py --mode fontainebleau --config configs/main.yaml
```

The `main`, `full`, and `final` modes remain available only for compatibility
with the earlier checkpoint layout. The second-version calibrated spatial
checkpoint and frozen crop, seed, and pair manifests are released under
`savedmodels/spatial_vqaware_v2/`.

## Figures 1–3: method schematics

- Fig. 1 is based on the 3D VQ-VAE implementation in
  `src/models/vqvae3d.py` and the spatial training script.
- Fig. 2 is based on FiLM-conditioned latent diffusion in
  `src/models/unet3d_film.py` and the same spatial training script.
- Fig. 3 represents TP-AQB in `src/sampling/tp_aqb.py` and
  `scripts/apply_tp_aqb.py`. The lightweight top-N quantile demo is not the
  manuscript TP-AQB algorithm.

## Figures 4–7: morphology and porosity control

- Fig. 4 uses representative TP-AQB outputs at targets 0.11–0.14.
- Fig. 5 compares conditional and no-condition sampling at a fixed threshold
  of 0.5; adaptive binarization is intentionally excluded from this ablation.
- Fig. 6 compares 95 paired samples: Real, Gen1 (FiLM plus fixed threshold),
  and Gen2 (FiLM plus TP-AQB). Group sizes are 8, 29, 39, and 19 at targets
  0.11, 0.12, 0.13, and 0.14.
- Fig. 7 uses `scripts/plot_slice_porosity.py` for a representative pair at
  target 0.13.

## Figure 8: spatial statistics and specific surface

Directional two-point correlation is evaluated with:

```bash
python scripts/evaluate_s2_lineal_edt.py \
  --real_root <held-out-real-root> \
  --gen_root data/generated_phi_sets_spatial \
  --out_root results/fig_s2 \
  --targets 0.11 0.12 0.13 0.14
```

Fig. 8a–c shows the X, Y, and Z curves at target 0.11. The other targets are
supplementary. Fig. 8d uses the 6-neighbor face-interface S/V values written by
the voxel-metric evaluation.

## Figure 9: pore-network geometry and topology

```bash
python scripts/evaluate_pore_network_6panel.py --config configs/main.yaml
python scripts/evaluate_coordination_euler.py \
  --real-root <held-out-real-root> \
  --gen-root data/generated_phi_sets_spatial \
  --out-root results/fig_pnm
```

The six main-text panels at target 0.11 are pore radius, throat radius, throat
shape factor, tortuosity, throat length, and coordination number. Results at
targets 0.12–0.14 are supplementary.

## Figure 10: main-sandstone permeability

Run `scripts/evaluate_voxel_and_perm.py` for the real and generated roots,
then compare the resulting tables:

```bash
python scripts/compare_metric_tables.py \
  --real-csv results/tables/real_voxel_perm.csv \
  --gen-csv results/tables/generated_voxel_perm.csv \
  --output-csv results/tables/permeability_comparison.csv \
  --source-data-csv results/source_data/fig10_permeability.csv
```

Fig. 10a reports Kx, Ky, Kz, and Kgeom at target 0.11. Fig. 10b reports Kgeom
across targets 0.11–0.14. The laboratory-sandstone voxel size is 3.5 um.

## Figures 11–14: Fontainebleau extrapolation

Place the three external `480^3` volumes at the paths in
`configs/fontainebleau_config.yaml`, then run `--mode fontainebleau`. Repository
labels 0.2045, 0.1743, and 0.1263 correspond to measured whole-volume
porosities 0.2041, 0.1766, and 0.1289.

- Fig. 11 compares real-crop porosity distributions and identifies the
  training-patch range 0.1926–0.2215.
- Fig. 12 shows directional S2 at measured porosity 0.1766.
- Fig. 13 shows the six pore-network descriptors at 0.1766.
- Fig. 14 shows permeability at 0.1766 and across all three conditions. Its
  source table is `results/source_data/fig14_fontainebleau.csv`.

Each condition compares 20 real subvolumes with 20 generated volumes. Results
for measured porosity 0.1289 are supplementary.

## Summary outputs

```bash
python src/metrics/summarize_all.py --root results
python scripts/plot_all.py
```

See `docs/figure_mapping.md` for the complete figure-to-code registry and
`docs/spatial_vqaware_workflow.md` for spatial-split and TP-AQB provenance.
