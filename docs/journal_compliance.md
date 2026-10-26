# Journal Compliance Map

This document maps the repository contents to the code, data, and
reproducibility expectations for the current **Fuel** submission.

## Scope Fit

The repository supports a manuscript at the interface of computing and geosciences:

- computational contribution: spatially separated 3D VQ-VAE,
  FiLM-conditioned latent diffusion, and TP-AQB with constrained voxel
  adjustment and topological consistency checking intended to reduce
  alterations to critical pore connections;
- geoscientific contribution: porosity-controllable reconstruction and validation of three-dimensional porous geomaterials from sandstone micro-CT data.

## Repository Requirements

| Journal expectation | Repository location |
| --- | --- |
| Public repository | `https://github.com/gaohui-ni/porosity-controlled-digital-rock-generation` |
| Clear license | `LICENSE` |
| English README with installation and usage | `README.md` |
| Dependencies and computational requirements | `requirements.txt`, `requirements_optional.txt`, `environment.yml`, `README.md`, `docs/dependency_versions.md` |
| Code for the scientific workflow | `src/`, `scripts/`, `pipelines/` |
| Official reproduction entry point | `run_pipeline.py` |
| Lightweight reproducible example | `run_demo.py`, `scripts/demo_quantile_binarization.py`, `examples/` |
| Synthetic test case when full raw experimental data are not bundled | `examples/`, `notebooks/tutorials/0001-basic-usage.ipynb` |
| Tutorials / how-to files | `notebooks/tutorials/`, `docs/user_guide.md`, `docs/reproducibility.md` |
| User guide with inputs and outputs | `docs/user_guide.md` |
| Data availability and redistribution limits | `docs/data_availability.md` |
| Checkpoints | Formal second-version main-sandstone and historical compatibility files are distributed through Git LFS; layout, scope, and checksums are documented in `docs/checkpoints.md` and `docs/model_manifest.md` |
| Figure/result reproduction map | `docs/figure_mapping.md`, `docs/figure_reproduction.md` |
| Automated lightweight checks | `.github/workflows/tests.yml`, `tests/` |

## Reproducibility Levels

1. **Lightweight check without raw data**

```bash
python run_demo.py
```

2. **Pipeline logic check without training**

```bash
python run_pipeline.py --mode spatial --config configs/main.yaml --dry-run
```

3. **A. Train the main-sandstone model from scratch**

```bash
python run_pipeline.py --mode spatial --config configs/main.yaml
```

This command performs training, condition-sweep generation, and TP-AQB. It
does not generate the formal released-weight Test-95 cohort or calculate all
scientific metrics for Figs. 8–10.

4. **B. Generate and stage Test-95 from released weights**

Run `scripts/generate_spatial_test95.py`, `scripts/apply_tp_aqb.py`, and
`scripts/evaluate_spatial_test95.py` in that order, using the commands in
`docs/figure_reproduction.md`. The third command validates and stages branches;
it is not the complete scientific evaluation.

5. **C. Compute scientific metrics and reproduce figures**

Run the metric-specific Fig. 8–10 commands in
`docs/figure_reproduction.md`. `src/metrics/summarize_all.py` may then inventory
the CSV/JSON files already produced; it does not calculate metrics or figures.

The separate historical `fontainebleau` compatibility workflow additionally
requires three external ANU Fontainebleau volumes, which are not redistributed
in this repository; it is not the formal Fuel Fig. 11 reproduction route.

## Data and Model Limitations

The main sandstone micro-CT volume is referenced through Mendeley Data rather
than stored in GitHub. The external ANU Fontainebleau volumes are not
redistributed. Compatibility checkpoints and the second-version spatial
final-sampling package are distributed through Git LFS. The package includes
the frozen crop, seed, and pair manifests but omits the 5.9 GB generated
volumes. The repository provides synthetic examples, configurations, and
training/generation/evaluation scripts so reviewers can inspect the workflow
and rerun the portions for which they have the required data and compute
environment.

The formal main-sandstone Test-95 package is directly verifiable and can be
regenerated from the released Mendeley volume. Fontainebleau is more limited:
the repository exposes retraining/evaluation code and historical compatibility
weights, but not a confirmed formal-v2 checkpoint plus complete Fuel Fig. 11
provenance bundle. This limitation is disclosed rather than treating the
compatibility checkpoint as the paper's final extrapolation model.

## Computational Requirements

- Lightweight tests and tutorials can be run on CPU.
- Full 256^3 model training and batch generation are intended for a CUDA-capable workstation/server.
- Pore-network and permeability analysis require optional dependencies listed in `requirements_optional.txt`.
- Default manuscript-scale raw data path and shape are defined in `configs/main.yaml`.

## Release Record

- The public workflow configuration is `configs/main.yaml`.
- Manuscript figure numbering is mapped in `docs/figure_mapping.md`.
- Released spatial and compatibility checkpoints are distributed through Git LFS as documented consistently in
  `README.md`, `docs/checkpoints.md`, and `docs/model_manifest.md`.
- Core direct dependency versions from the experiment environment and
  reconstruction instructions are recorded under `environment/`.
- `results/results_summary.json`, `results/summary.json`, and
  `results/results_summary.csv` are local inventories of existing result
  files. They are not distributed and are not themselves scientific metric
  outputs.
