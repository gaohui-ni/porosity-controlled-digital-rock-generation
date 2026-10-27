# Fuel Manuscript Figure Registry

This registry follows the current Fuel manuscript, whose main text contains
**Figures 1–11**. Older repository references to main-text Figs. 12–14 are
historical/supplementary mappings and are not part of this registry.

| Figure | Fuel manuscript content | Reproducibility source |
| --- | --- | --- |
| Fig. 1 | 3D EMA VQ-VAE | `VectorQuantizerEMA` and the encoder/decoder in `scripts/train_256_vqvae_ddpm_spatial_vqaware_v3_film.py` |
| Fig. 2 | FiLM-conditioned latent DDPM | The same spatial training script and the released calibrated checkpoint/configuration |
| Fig. 3 | TP-AQB | `src/sampling/tp_aqb.py`, `scripts/apply_tp_aqb.py`, and `configs/tp_aqb_v2.yaml` |
| Fig. 4 | Representative morphology, pore space, and slices at 0.11–0.14 | Post-generation visualization prepared in Avizo from final generated binary volumes; it is not directly rendered by the Python evaluation pipeline |
| Fig. 5 | FiLM ablation with fixed threshold 0.5 | The ablation's own conditional/no-condition checkpoints, targets, and seeds; TP-AQB is excluded |
| Fig. 6 | Real vs Gen1 fixed threshold vs Gen2 TP-AQB porosity distributions for 95 paired crops | Existing binarization-strategy ablation/control results: Real rocks, FiLM-conditioned generation with fixed threshold 0.5 (Gen1), and FiLM-conditioned generation with TP-AQB (Gen2). This is not the Fig. 5 FiLM-OFF ablation |
| Fig. 7 | Slice-wise porosity for the existing representative real/generated samples at approximately 0.13 | Existing experiment record. Preserve the samples and curves shown in the manuscript; `scripts/plot_slice_porosity.py` is a plotting utility, not evidence that the historical panel can be reselected or exactly redrawn without its original sample record |
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

## Figure 4 visualization boundary

Figure 4 was assembled after generation as follows: a trained model generated
the `256^3` digital rocks; the final binary volumes were opened in Avizo; the
complete three-dimensional morphology, pore-space structure, and
two-dimensional slices were exported separately; and those exports were
arranged into the manuscript panel. The four displayed porosity conditions are
`0.11`, `0.12`, `0.13`, and `0.14`.

**Figure 4 is a post-generation visualization prepared using Avizo, rather
than a figure directly rendered by the Python evaluation pipeline.** The
repository does not claim that an existing Python plotting command recreates
the final Figure 4 layout. No Avizo project or unverified panel sample IDs are
provided.

## Figures 6–7 historical-source boundary

Figures 6 and 7 come from the experiments already used for the manuscript;
they were not newly generated for this repository documentation repair.
Figure 6 is specifically the Real/Gen1/Gen2 binarization comparison described
above, whereas Figure 5 is the separate FiLM conditioning ablation. Figure 7
uses the representative real and generated samples at approximately
`phi = 0.13` already shown in the paper. Existing conclusions and numerical
records are retained. If an exact historical plotting script or source table
is unavailable, this registry does not substitute invented CSV rows, sample
IDs, or a newly selected representative sample.

See `docs/figure_reproduction.md` for commands and
`docs/fuel_repro_status.md` for passed, failed, not-run, and blocked items.
