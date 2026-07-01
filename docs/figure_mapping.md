# Figure Mapping

This file is the short manuscript-to-code map for reviewers. Adjust final figure numbers after the manuscript layout is fixed.

The official full reproduction entry point is:

```bash
python run_pipeline.py --mode full --config configs/main.yaml
```

For figure-oriented reproduction after configuring the data paths, use:

```bash
python scripts/reproduce_figures.py --config configs/main.yaml
```

| Manuscript result group | Code entry point | Output folder |
| --- | --- | --- |
| Porosity-controlled generation demo | `python run_demo.py` | `examples/` |
| S2, lineal-path, and EDT curves | `python scripts/evaluate_s2_lineal_edt.py ...` | `results/fig_s2/` |
| Permeability and voxel metrics | `python scripts/evaluate_voxel_and_perm.py ...` | `results/tables/`, `results/fig_perm/` |
| Topology and pore-network metrics | `python scripts/evaluate_coordination_euler.py ...` | `results/fig_pnm/` |
| Fontainebleau validation | `docs/fontainebleau_protocol.md` | `results/fig_fontainebleau/` |
| Full command list | `python run_pipeline.py --mode full --config configs/main.yaml --dry-run` | `results/pipeline_full_manifest.json` |
| Figure reproduction helper | `python scripts/reproduce_figures.py --config configs/main.yaml` | `results/figures/` |

For detailed commands, see `docs/figure_reproduction.md`.
