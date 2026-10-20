# Fontainebleau Validation Protocol

This protocol describes the separate Fontainebleau sandstone experiment used
to test cross-sample porosity extrapolation within the same lithology. It is
not a zero-shot application of the main-sandstone model and is not evidence of
cross-lithology transfer.

## Dataset and labels

Three `480 x 480 x 480` digital rocks at `5.68 um/voxel` are used. The rock
with measured whole-volume porosity `0.2041` is the training rock; the rocks
at `0.1766` and `0.1289` are lower-porosity external validation rocks. Their
subvolume porosity distributions lie outside the observed training-patch range
of `0.1926–0.2215`.

For compatibility with existing filenames and scripts, the repository uses
the nominal labels `0.2045`, `0.1743`, and `0.1263`. The mapping is explicit in
`configs/fontainebleau_config.yaml`:

| Repository label | Manuscript measured whole-volume porosity | Role |
| ---: | ---: | --- |
| 0.2045 | 0.2041 | training condition |
| 0.1743 | 0.1766 | unseen validation condition |
| 0.1263 | 0.1289 | unseen validation condition |

The volumes are third-party ANU data and are not redistributed. Users must
obtain them from the original provider, standardize them to binary `uint8`
with `0 = solid` and `1 = pore`, and declare their local paths under
`real_volumes` in `configs/fontainebleau_config.yaml`.

The currently configured VQ-VAE and latent-DDPM training schedules are retained
unchanged pending separate confirmation; this documentation update does not
resolve or revise the epoch-count question.

- VQ-VAE training: 80 epochs
- Latent DDPM training: 150 epochs

## Evaluation design

For each condition, the workflow compares 20 generated `256^3` volumes with
20 real `256^3` subvolumes. The real crops are subvolumes of one parent rock
per condition, not 20 independent specimens. Distribution bands therefore
describe variation among subvolumes and must not be interpreted as uncertainty
from 20 independent rocks.

The evaluation covers directional two-point correlation, pore/throat geometry,
tortuosity, throat length, coordination number, and directional plus
geometric-mean permeability. The `0.1766` condition is the representative main
text case; results for `0.1289` are reported in the Supplementary Material.

## Command

```bash
python run_pipeline.py --mode fontainebleau --config configs/main.yaml
```

The workflow:

1. extracts 20 porosity-matched real crops from each configured volume;
2. generates 20 samples at each corresponding nominal target;
3. runs the voxel, spatial-statistics, pore-network, and permeability analyses;
4. writes outputs under `results/fig_fontainebleau/`, tables under
   `results/tables/`, and Fig. 14 source data under
   `results/source_data/fig14_fontainebleau.csv`.

The currently released Fontainebleau files under `savedmodels/` belong to the
earlier compatibility workflow. Their role and checksums are recorded in
`docs/model_manifest.md`.

## Public reproducibility boundary

The repository does not currently distribute a confirmed formal-v2
Fontainebleau checkpoint, the complete 20-real/20-generated crop and seed
manifests, or the frozen source-data bundle used for Figs. 11–14 and Table 3.
Local records contain both a later TP-AQB-v2 20-window expansion and a
five-sample permeability source path. Until the authors select and publish one
complete provenance chain, the declared 20/20 design must not be interpreted
as directly reproducible from the public files. No compatibility checkpoint
may be substituted silently. See `docs/fontainebleau_provenance_audit.md`.
