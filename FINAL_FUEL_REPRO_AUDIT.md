# Fuel reproduction status

Audit updated: 2026-10-26
Repository: `gaohui-ni/porosity-controlled-digital-rock-generation`  
Audited baseline: `2d603a89a34485d7daab0a9b5fbaa4fb7ccdd5a4`

This is a public code, documentation, and provenance status statement. The
audit did not retrain a model, regenerate the formal 95-sample cohort, rerun
PNM/permeability, alter released tensors or configurations, change sample IDs
or seeds, or publish internal result CSV files.

## Verified public workflow

| Area | Status | Public conclusion |
| --- | --- | --- |
| Test-95 identity and staging | `PASS` | The released manifests define 95 used crop IDs in groups 8/29/39/19. Evaluation validates identities, shapes, binary values, targets, seeds, and TP-AQB RAW/NPZ agreement. |
| Fuel Figs. 8–10 input contracts | `PASS` | Fig. 8 uses TP-AQB RAW and unnormalized directional `S2`; Fig. 9 reads TP-AQB NPZ key `seg`; Fig. 10 reads TP-AQB RAW. |
| Release metadata and checksums | `PASS` | The spatial release contains 14 manifest entries and distinguishes the `v0.2.0` release candidate from formal GitHub Release `v0.1.1`. |
| TP-AQB description | `PASS` | The public wording describes constrained voxel adjustment, topological consistency checking, and safeguards intended to reduce alterations to critical pore connections; it does not claim guaranteed topology preservation. |
| Fontainebleau sample count | `PASS` | The final recorded comparison contains 20 REAL + 20 GEN samples per condition. The 5+5 records describe a historical precursor batch. |
| VQ-VAE exact training history | `BLOCKED` | The manuscript/current config says 15 epochs, while the released checkpoint embeds `epochs_vae=80` and stores selected zero-based epoch 12. The complete original training log is unavailable. |
| Fontainebleau Fig. 11/Table 3 source chain | `BLOCKED` | The final count is 20+20, but the formal checkpoint, complete crop/seed manifests, and hashes tying the inspected table to the published figure/table are not released. |
| Fuel Fig. 5 no-condition frozen-weight reproduction | `BLOCKED` | The matching no-condition checkpoint is not bundled. |

## What was executed

The repository audit verified the release manifest, Git LFS materialization,
all 14 recorded hashes, tensor metadata, the spatial dry-run command, Python
syntax for the modified workflow scripts, and the full lightweight test suite.
The current documentation-only follow-up is checked with:

```bash
pytest -q
git diff --check
```

These checks validate repository behavior and document consistency. They are
not substitutes for the manuscript-scale computations listed below.

## Not rerun

- formal 95-volume diffusion generation (approximately 5.9 GB of outputs);
- full S2, PNM, and permeability evaluation on that cohort;
- end-to-end Fuel figure rendering;
- any VAE/DDPM training, fine-tuning, or TP-AQB ablation;
- creation of a GitHub Release.

## Fontainebleau scope and limitation

The final recorded Fontainebleau comparison uses 20 REAL and 20 GEN
subvolumes per condition for `0.2041`, `0.1766`, and `0.1289`; `0.0829` is
exploratory and outside the current main result. Each set of real subvolumes
comes from one parent rock and therefore does not represent 20 independent
rock specimens.

Local records state that 15 samples per arm were added to an original
five-sample batch. This supports treating 5+5 as historical and 20+20 as the
final sample-count convention. It does not resolve the separate provenance
question: the public repository still lacks the formal checkpoint, complete
crop/seed and TP-AQB manifests, and source hashes needed to prove that the
inspected 20+20 permeability table is exactly the one used for Fuel Fig. 11
and Table 3. Direct numerical reproduction of those outputs therefore remains
`BLOCKED`; compatibility artifacts must not be substituted silently.

## Reproduction boundary

The repository supports code-level retraining, released main-sandstone
Test-95 sampling, and evaluation under the documented environment. Exact
historical recovery remains limited where training logs, ablation weights, or
Fontainebleau source manifests are missing. These limitations are substantive
and should accompany any reproducibility claim.
