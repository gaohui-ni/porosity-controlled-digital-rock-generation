# Fontainebleau scientific-provenance audit

Audit updated: 2026-10-26. Original baseline: `6da2bab`. This is a provenance audit, not a
new experiment. No checkpoint, RAW volume, seed, crop identifier, generated
sample, or manuscript statistic was changed.

## Finding

**Status: BLOCKED—SCIENTIFIC PROVENANCE.** The public repository does not
contain a confirmed formal-v2 Fontainebleau checkpoint plus complete frozen
sample, seed, crop, TP-AQB, metric, and figure-source manifests. Local records
include a 20-real/20-generated permeability table for each of four conditions.
They also state that 15 samples per arm were added to an original five-sample
batch, so the 5+5 records are treated here as historical precursor records and
the final comparison count is 20 REAL + 20 GEN per condition. The missing
final checkpoint, manifests, and figure-source hashes still prevent a
self-contained public provenance chain. Those artifacts are required before
the release can claim direct numerical reproduction of Fuel Fig. 11 and
Table 3.

## 20-sample versus 5-sample evidence

| Item | 20-sample record | 5-sample record |
| --- | --- | --- |
| Actual REAL count | Local `BY_POROSITY2_SUMMARY.csv` and `BY_POROSITY2_PERM.csv` copies contain 20 rows per arm and condition; windows from each parent overlap and are not independent rocks | Companion notes describe 5 external REAL permeability rows per condition from an older source |
| Actual GEN count | The same local tables contain 20 TP-AQB-v2 rows per arm and condition; external seeds `2031101400..419` are described | Companion notes describe 5 external generated permeability rows per condition |
| Training checkpoint | local notes name `cont80 / ep075`; no confirmed formal-v2 checkpoint is present in this repository | same unresolved local lineage; public `savedmodels/fontainebleau_phi0p2045/` is compatibility-only |
| Sampling configuration | later note says 15 new seeds were added to the original 5 | original five-seed external sampling |
| TP-AQB version | local 2026-10-15 note says the expanded set used TP-AQB v2 | older notes include both TP-AQB-v1 history and a five-sample v2 subset; no complete public manifest resolves every artifact |
| Seed manifest | ranges are described in local notes, but no complete formal manifest is published | not published as a formal release manifest |
| S2 / PNM data | plotted/cached 20-window products and summaries exist locally | older five-window products are referenced locally |
| Permeability data | The inspected local combined table has non-null `Kgeom` for 20 REAL and 20 GEN rows in each condition | The companion README points to `EXT3X_REAL5_GEN5_PERMEABILITY.csv` for the earlier five-sample stage; that historical CSV was not found in the searched local roots |
| Kgeom statistics | 0.2041 median: REAL 2.2803 D, GEN 2.4833 D | 0.1766: REAL 1.93642 D, GEN 1.4175 D; 0.1289: REAL 0.4473 D, GEN 0.3789 D |
| Paper mapping | Manuscript text declares 20 REAL and 20 GEN per condition | No released hash or manifest ties either local permeability lineage conclusively to Fuel Fig. 11/Table 3 |
| Complete reproducibility | No: missing public formal checkpoint and end-to-end manifests | No: incomplete public seed/crop/metric provenance |

The local tables use per-volume `Kgeom = (Kx*Ky*Kz)^(1/3)` under the documented
plotting convention. Their row counts and the recorded 15-sample expansion
establish the final 20+20 comparison count and identify 5+5 as a historical
precursor. They do **not** establish, without the missing hashes and manifests,
that the inspected 20+20 table is the exact source rendered in Fuel Fig. 11 or
reported in Table 3.

## Public checkpoint audit

The repository currently publishes:

- a formal main-sandstone package at `savedmodels/spatial_vqaware_v2/`;
- historical compatibility Fontainebleau files at
  `savedmodels/fontainebleau_phi0p2045/`.

It does not publish an artifact set that proves the compatibility files are the
Fuel Fig. 11 checkpoint. Therefore they must not be relabelled or silently
substituted.

## Required resolution

Choose one author-confirmed provenance chain and publish, without altering its
scientific content:

1. the exact final Fontainebleau VQ-VAE/DDPM checkpoint and latent statistics;
2. the three parent-volume identities and measured-porosity mapping;
3. complete REAL crop coordinates and GEN seed manifests;
4. TP-AQB version/configuration and per-volume identities;
5. S2, PNM and permeability source tables used by each figure/table;
6. hashes tying Fuel Fig. 11, Table 3 and any supplementary analyses to those
   files.

Until then the repository accurately supports code-level retraining and
evaluation, but not direct numerical reproduction of the final Fontainebleau
figures.
