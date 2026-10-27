# Fuel Figure Reproduction

## 1. Frozen Test-95 generation and branch assembly

Generate the released probability fields and the two historical binary
baselines:

```bash
python scripts/generate_spatial_test95.py \
  --raw-path data/raw/Bei_800x800x800.raw \
  --device cuda
```

Generate actual TP-AQB v2 outputs:

```bash
python scripts/apply_tp_aqb.py \
  --input-root outputs/spatial_test95/gen \
  --output-root data/generated_phi_sets_spatial_test95 \
  --config configs/tp_aqb_v2.yaml
```

Join all branches by `crop_id`, validate 95 unique crops and group sizes
8/29/39/19, and stage the existing evaluators' directory trees:

```bash
python scripts/evaluate_spatial_test95.py \
  --manifest outputs/spatial_test95/FINAL_TEST_PAIR_MANIFEST.csv \
  --gen-root outputs/spatial_test95/gen \
  --tpaqb-root data/generated_phi_sets_spatial_test95 \
  --out-root results/spatial_test95
```

The historical `phi_post_aqb`/`*_on.raw` data are exact top-N, not TP-AQB.
The wrapper preserves them as `stage/topn` and writes separate
`phi_tpaqb`/`tpaqb_path` fields in its augmented manifest. It stages each
validated TP-AQB volume twice without adding a second statistical sample:
`stage/tpaqb/<group>/<crop_id>.raw` for RAW consumers and
`stage/tpaqb_npz/<group>/<crop_id>.npz` (array key `seg`) for NPZ consumers.

## 2. Fuel Figs. 8–10

Use the final TP-AQB branch consistently:

```bash
python scripts/evaluate_s2_lineal_edt.py \
  --real_root results/spatial_test95/stage/real \
  --gen_root results/spatial_test95/stage/tpaqb \
  --gen_use raw \
  --out_root results/fig_s2 \
  --targets 0.11 0.12 0.13 0.14

python scripts/evaluate_pore_network_6panel.py \
  --config configs/main.yaml \
  --real-root results/spatial_test95/stage/real \
  --gen-root results/spatial_test95/stage/tpaqb_npz \
  --gen-npz-key seg \
  --out-root results/fig_pnm \
  --targets 0.11 0.12 0.13 0.14

python scripts/evaluate_coordination_euler.py \
  --real-root results/spatial_test95/stage/real \
  --gen-root results/spatial_test95/stage/tpaqb \
  --out-root results/fig_pnm \
  --phis phi0p11 phi0p12 phi0p13 phi0p14

python scripts/evaluate_voxel_and_perm.py \
  --input_root results/spatial_test95/stage/real \
  --output_csv results/tables/real_voxel_perm.csv \
  --group_name real --shape 256 256 256 --recursive --extensions .raw

python scripts/evaluate_voxel_and_perm.py \
  --input_root results/spatial_test95/stage/tpaqb \
  --output_csv results/tables/generated_voxel_perm.csv \
  --group_name generated --shape 256 256 256 --recursive --extensions .raw

python scripts/compare_metric_tables.py \
  --real-csv results/tables/real_voxel_perm.csv \
  --gen-csv results/tables/generated_voxel_perm.csv \
  --output-csv results/tables/fig10_permeability_comparison.csv
```

The final command only groups the two permeability CSV files that already
exist locally; it does not rerun PNM/OpenPNM, regenerate digital rocks, or
render the manuscript Figure 10. It groups the unchanged Test-95 cohort by
the four existing target tags (`0.11`, `0.12`, `0.13`, and `0.14`) and writes
per-group `mean`, `std`, and non-null `count` fields. Directional permeability
is reported as `Kx`, `Ky`, and `Kz`, and the geometric mean as `Kgeom`, with
the input table retaining both `_m2` and `_D` units. A failed or
non-percolating direction is `NaN` and therefore contributes zero to that
metric's valid count; `Kgeom` is valid only when all three directions are
valid. The generated comparison CSV is a local derived table and must not be
committed or presented as an automatically reconstructed final Figure 10.

Fuel Fig. 8 uses the unnormalized two-point probability
`S2,d(r) = <I(x) I(x+r e_d)>`. Therefore the command intentionally omits
`--normalize_s2`; adding that option would change the manuscript statistic.

## 3. Figures 4, 6, and 7

### Figure 4: Avizo post-generation visualization

Figure 4 is not a direct output of the Python evaluation pipeline. Its actual
preparation sequence was:

1. generate a `256^3` three-dimensional digital rock with a trained model;
2. obtain the final generated binary volume;
3. visualize that volume in Avizo;
4. export the complete 3D morphology, pore-space structure, and 2D slice
   views separately; and
5. arrange those exports into manuscript Figure 4.

The four displayed porosity conditions remain `0.11`, `0.12`, `0.13`, and
`0.14`. No current Python plotting script is claimed to reproduce the final
Avizo views or panel layout. An Avizo project and exact panel sample IDs are
not fabricated or required here, and the original figure is unchanged.

### Figures 6–7: existing experiment records

Figure 6 compares the existing Real, Gen1, and Gen2 results. Real denotes the
real-rock cohort; Gen1 denotes FiLM-conditioned generation binarized at the
fixed threshold `0.5`; and Gen2 denotes FiLM-conditioned generation followed
by TP-AQB. This is a binarization-strategy ablation/control comparison, not
the Figure 5 FiLM-OFF ablation.

Figure 7 retains the representative real and generated samples at
approximately `phi = 0.13` and their slice-wise porosity curves as shown in
the paper. `scripts/plot_slice_porosity.py` can plot a specified pair, but the
repository does not claim one-click exact reconstruction of the historical
panel when its exact source invocation is unavailable. Do not choose a new
sample or infer a sample ID from directory order to replace the manuscript
panel. No retraining, Test-95 regeneration, original-figure replacement, or
invented source CSV is required for either figure.

## 4. Model identities

- Fig. 1 references the EMA implementation in the spatial training script,
  not legacy `src/models/vqvae3d.py`.
- Fig. 5 requires its own no-condition checkpoint, which is not bundled; its
  exact released-weight reproduction is therefore blocked.
- Historical figure sample IDs must not be inferred from directory ordering.
  Preserve the existing manuscript panels when their exact source invocation
  is unavailable.

## 5. Fontainebleau Fig. 11

The Fuel manuscript uses only 0.2041 (training/in-sample reference), 0.1766,
and 0.1289. The old `fontainebleau_phi0p2045` 80/150 checkpoint is a
compatibility artifact and must not be presented as reproducing Fig. 11.

The candidate final lineage is an epoch-75 checkpoint with an EXT3X sampler,
but those exact artifacts and their source tables are not currently bundled.
Accordingly, the proposed `generate_fontainebleau_extrap.py` entry point is not
published as operational and Fig. 11 remains blocked. The 0.0829 exploratory
condition is outside the current main manuscript.

## 6. Training-schedule limitation

The manuscript states 15 VQ-VAE and 40 DDPM epochs. The released VAE file is
identified by SHA256, stores zero-based best epoch 12, and embeds an 80-epoch
configuration; the released DDPM stores zero-based best epoch 34 and embeds a
40-epoch configuration. Without the complete original training log, exact
15-versus-80 VQ-VAE training-history provenance is unresolved. Sampling from
the frozen weights is reproducible, but their exact training history must not
be claimed as resolved. See
`savedmodels/spatial_vqaware_v2/MANUSCRIPT_PROVENANCE.json`.
