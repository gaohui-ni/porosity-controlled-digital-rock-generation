# Batch generation and Fontainebleau validation workflow

This page separates complete pipeline execution from generation-only commands.
Choose one route for a given output directory. Do not run a complete pipeline
and then repeat its generation step unless a deliberate regeneration is
required.

## 1. Main sandstone: spatial training route

To retrain the spatial VQ-aware workflow from scratch, run:

```bash
python run_pipeline.py --mode spatial --config configs/main.yaml
```

This route trains the spatial VQ-VAE and latent DDPM, estimates latent
statistics, generates the configured condition-sweep probability volumes, and
applies TP-AQB. It does **not** create the frozen formal Test-95 cohort and does
not calculate all scientific metrics used in Figures 8–10. The formal Test-95
generation, staging, and metric commands are documented separately in
[`reproduction.md`](reproduction.md) and
[`figure_reproduction.md`](figure_reproduction.md).

## 2. Main sandstone: independent compatibility generation

Use this generation-only route when compatible checkpoints already exist and
you do not want to execute the full training workflow. For a newly trained
compatibility checkpoint directory:

```bash
python scripts/generate_batch.py \
  --ckpt_dir outputs/main_sandstone \
  --out_root data/generated_phi_sets \
  --targets 0.11 0.12 0.13 0.14 0.15 \
  --n_per_target 100 \
  --poro_center 0.13 \
  --device cuda
```

For the supplied compatibility checkpoint package, replace the checkpoint
argument with:

```text
--ckpt_dir savedmodels/main_sandstone
```

This historical compatibility generator is not the formal Test-95 protocol.
Its output is organized as:

```text
data/generated_phi_sets/
|-- phi0p11/
|-- phi0p12/
|-- phi0p13/
|-- phi0p14/
`-- phi0p15/
```

Each target folder contains generated volumes and associated metadata. Running
this command after a complete pipeline against the same `--out_root` repeats
the generation step.

## 3. Prepare local Fontainebleau inputs

The Fontainebleau volumes are third-party data and are not redistributed
because the authors do not hold redistribution rights. Users must obtain them
from the original provider. Before analysis, standardize the locally obtained
volumes to binary `uint8` format with `0 = solid`, `1 = pore`, and a common
size of `480 x 480 x 480` voxels. Original file packaging and phase-label
conventions may vary by provider version. Declare the standardized local paths
in `configs/fontainebleau_config.yaml`.

## 4. Fontainebleau route A: complete compatibility workflow

The following single command prepares matched real validation patches, trains
the Fontainebleau model, generates the configured validation samples, evaluates
spatial statistics, voxel/permeability metrics, topology, and PNM descriptors,
and finally inventories the result files:

```bash
python run_pipeline.py --mode fontainebleau --config configs/main.yaml
```

This is the complete compatibility route. Do not additionally run the
generation-only command in the next section against the same output directory.

## 5. Fontainebleau route B: independent generation only

If a compatible checkpoint already exists and only generated samples are
needed, run `generate_batch.py` directly instead of the complete route:

```bash
python scripts/generate_batch.py \
  --ckpt_dir outputs/fontainebleau_phi0p2045 \
  --out_root data/generated_fontainebleau_sets \
  --targets 0.2045 0.1743 0.1263 \
  --n_per_target 20 \
  --poro_center 0.13 \
  --device cuda
```

For the supplied compatibility checkpoint package, use:

```text
--ckpt_dir savedmodels/fontainebleau_phi0p2045
```

This route only generates samples. It does not prepare the real validation
patches or run the Fontainebleau scientific evaluations. Those downstream
steps must be invoked explicitly, or the complete route must be run in a fresh
output location. Current provenance limitations for the published
Fontainebleau comparison remain documented in
[`fuel_repro_status.md`](fuel_repro_status.md) and the repository-root
`FINAL_FUEL_REPRO_AUDIT.md`.
