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
```

Fuel Fig. 8 uses the unnormalized two-point probability
`S2,d(r) = <I(x) I(x+r e_d)>`. Therefore the command intentionally omits
`--normalize_s2`; adding that option would change the manuscript statistic.

## 3. Model and figure identities

- Fig. 1 references the EMA implementation in the spatial training script,
  not legacy `src/models/vqvae3d.py`.
- Fig. 5 requires its own no-condition checkpoint, which is not bundled; its
  exact released-weight reproduction is therefore blocked.
- Figs. 4 and 7 require explicit panel/sample IDs. Do not select a crop by
  directory ordering.

## 4. Fontainebleau Fig. 11

The Fuel manuscript uses only 0.2041 (training/in-sample reference), 0.1766,
and 0.1289. The old `fontainebleau_phi0p2045` 80/150 checkpoint is a
compatibility artifact and must not be presented as reproducing Fig. 11.

The candidate final lineage is an epoch-75 checkpoint with an EXT3X sampler,
but those exact artifacts and their source tables are not currently bundled.
Accordingly, the proposed `generate_fontainebleau_extrap.py` entry point is not
published as operational and Fig. 11 remains blocked. The 0.0829 exploratory
condition is outside the current main manuscript.

## 5. Training-schedule limitation

The manuscript states 15 VQ-VAE and 40 DDPM epochs. The released VAE file is
identified by SHA256, stores zero-based best epoch 12, and embeds an 80-epoch
configuration; the released DDPM stores zero-based best epoch 34 and embeds a
40-epoch configuration. Without the complete original training log, exact
15-versus-80 VQ-VAE training-history provenance is unresolved. Sampling from
the frozen weights is reproducible, but their exact training history must not
be claimed as resolved. See
`savedmodels/spatial_vqaware_v2/MANUSCRIPT_PROVENANCE.json`.
