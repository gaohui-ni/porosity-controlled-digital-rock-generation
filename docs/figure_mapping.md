# Fuel Manuscript Figure Registry

This registry follows the current Fuel manuscript, whose main text contains
**Figures 1–11**. Older repository references to main-text Figs. 12–14 are
historical/supplementary mappings and are not part of this registry.

| Figure | Fuel manuscript content | Reproducibility source |
| --- | --- | --- |
| Fig. 1 | 3D EMA VQ-VAE | `VectorQuantizerEMA` and the encoder/decoder in `scripts/train_256_vqvae_ddpm_spatial_vqaware_v3_film.py` |
| Fig. 2 | FiLM-conditioned latent DDPM | The same spatial training script and the released calibrated checkpoint/configuration |
| Fig. 3 | TP-AQB | `src/sampling/tp_aqb.py`, `scripts/apply_tp_aqb.py`, and `configs/tp_aqb_v2.yaml` |
| Fig. 4 | Representative morphology, pore space, and slices at 0.11–0.14 | Frozen Test-95 probability fields followed by TP-AQB; panel volume IDs must be recorded with the figure source data |
| Fig. 5 | FiLM ablation with fixed threshold 0.5 | The ablation's own conditional/no-condition checkpoints, targets, and seeds; TP-AQB is excluded |
| Fig. 6 | Real vs Gen1 fixed threshold vs Gen2 TP-AQB, 95 paired crops | `generate_spatial_test95.py`, `apply_tp_aqb.py`, then `evaluate_spatial_test95.py`; join by `crop_id` |
| Fig. 7 | Slice-wise porosity for a representative target-0.13 pair | `scripts/plot_slice_porosity.py`; the selected `crop_id` and branch must accompany source data |
| Fig. 8 | Spatial correlation/lineal-path/specific-surface evaluation | Existing metric scripts run on `results/spatial_test95/stage/real` and RAW `stage/tpaqb`; Fuel uses unnormalized `S2` |
| Fig. 9 | Pore-network geometry and topology | `evaluate_pore_network_6panel.py` reads `seg` from `stage/tpaqb_npz`; `evaluate_coordination_euler.py` uses the staged real/TP-AQB tree appropriate to its CLI |
| Fig. 10 | Main-sandstone directional and geometric-mean permeability | `evaluate_voxel_and_perm.py` on the same staged real/TP-AQB cohort |
| Fig. 11 | Fontainebleau permeability at 0.2041, 0.1766, and 0.1289 | Manuscript-specific epoch-75/EXT3X lineage remains blocked until the exact checkpoint, sampler, seeds, and source tables are available |

## Critical branch labels

- `*_off.raw` / `stage/fixed`: FiLM plus threshold 0.5 (**Gen1**).
- `*_on.raw` / `stage/topn`: historical exact top-N diagnostic. The legacy
  manifest column `phi_post_aqb` retains this meaning and is not TP-AQB.
- `stage/tpaqb`: actual TP-AQB v2 output (**Gen2** and the final generated
  branch used by Figs. 4 and 6–10 unless a caption explicitly says otherwise).
- `stage/tpaqb_npz`: the same validated TP-AQB voxels packaged once per crop
  with NPZ key `seg` for the PNM evaluator; it is not an additional cohort.

`src/models/vqvae3d.py` is a legacy dual-loss compatibility implementation; it
is not the EMA codebook implementation represented by Fuel Fig. 1.

The 95 crops are overlapping subvolumes from one spatially held-out region,
not 95 independent parent specimens. Group sizes are fixed at 8/29/39/19 for
nominal porosities 0.11/0.12/0.13/0.14.

See `docs/figure_reproduction.md` for commands and
`docs/fuel_repro_status.md` for passed, failed, not-run, and blocked items.
