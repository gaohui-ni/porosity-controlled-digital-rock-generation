# Fontainebleau scientific-provenance audit

Audit date: 2026-10-20. Baseline: `6da2bab`. This is a provenance audit, not a
new experiment. No checkpoint, RAW volume, seed, crop identifier, generated
sample, or manuscript statistic was changed.

## Finding

**Status: BLOCKED—SCIENTIFIC PROVENANCE.** The public repository does not
contain a confirmed formal-v2 Fontainebleau checkpoint plus complete frozen
sample, seed, crop, TP-AQB, metric, and figure-source manifests. Local records
also mix a 20-window TP-AQB-v2 structural evaluation with 5+5 permeability
inputs for two external conditions. An author decision and missing artifacts
are required before the release can claim direct numerical reproduction of
Figs. 11–14 and Table 3.

## 20-sample versus 5-sample evidence

| Item | 20-sample record | 5-sample record |
| --- | --- | --- |
| Actual REAL count | 20 windows per condition are declared in `BY_POROSITY2_SUMMARY.csv`; windows from each parent overlap and are not independent rocks | 5 external REAL permeability rows per condition are described by `EXT3X_REAL5_GEN5_PERMEABILITY.csv` provenance |
| Actual GEN count | 20 TP-AQB-v2 seeds per condition are declared; external seeds `2031101400..419` are described | 5 external generated permeability rows per condition |
| Training checkpoint | local notes name `cont80 / ep075`; no confirmed formal-v2 checkpoint is present in this repository | same unresolved local lineage; public `savedmodels/fontainebleau_phi0p2045/` is compatibility-only |
| Sampling configuration | later note says 15 new seeds were added to the original 5 | original five-seed external sampling |
| TP-AQB version | local 2026-10-15 note says the expanded set used TP-AQB v2 | older notes include both TP-AQB-v1 history and a five-sample v2 subset; no complete public manifest resolves every artifact |
| Seed manifest | ranges are described in local notes, but no complete formal manifest is published | not published as a formal release manifest |
| S2 / PNM data | plotted/cached 20-window products and summaries exist locally | older five-window products are referenced locally |
| Permeability data | training condition 0.2041 uses 20 REAL and 20 GEN volumes | local plotting README explicitly says external conditions read `EXT3X_REAL5_GEN5_PERMEABILITY.csv` |
| Kgeom statistics | 0.2041 median: REAL 2.2803 D, GEN 2.4833 D | 0.1766: REAL 1.93642 D, GEN 1.4175 D; 0.1289: REAL 0.4473 D, GEN 0.3789 D |
| Paper mapping | manuscript text declares 20 REAL and 20 GEN per condition | Table 3 external-condition values trace to the documented 5+5 permeability source, while the training condition traces to 20+20 |
| Complete reproducibility | No: missing public formal checkpoint and end-to-end manifests | No: incomplete public seed/crop/metric provenance |

The six Table 3 values are medians of positive per-volume `Kgeom =
(Kx*Ky*Kz)^(1/3)` values under the documented plotting convention. The
available evidence does **not** support describing all three permeability
conditions as 20-real/20-generated measurements.

## Public checkpoint audit

The repository currently publishes:

- a formal main-sandstone package at `savedmodels/spatial_vqaware_v2/`;
- historical compatibility Fontainebleau files at
  `savedmodels/fontainebleau_phi0p2045/`.

It does not publish an artifact set that proves the compatibility files are the
final Fig. 11–14 checkpoint. Therefore they must not be relabelled or silently
substituted.

## Required resolution

Choose one author-confirmed provenance chain and publish, without altering its
scientific content:

1. the exact final Fontainebleau VQ-VAE/DDPM checkpoint and latent statistics;
2. the three parent-volume identities and measured-porosity mapping;
3. complete REAL crop coordinates and GEN seed manifests;
4. TP-AQB version/configuration and per-volume identities;
5. S2, PNM and permeability source tables used by each figure/table;
6. hashes tying Figs. 11–14, Table 3 and Supplementary Figs. S13–S15 to those
   files.

Until then the repository accurately supports code-level retraining and
evaluation, but not direct numerical reproduction of the final Fontainebleau
figures.
