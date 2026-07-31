# Computers & Geosciences Reproduction Entry Point

Use the official top-level command:

```bash
python run_pipeline.py --mode full --config configs/main.yaml
```

The `full` mode requires both the publicly released main-sandstone volume and
access to the external ANU Fontainebleau volumes, which are not redistributed
in this repository. To reproduce only the public main-sandstone workflow, use:

```bash
python run_pipeline.py --mode main --config configs/main.yaml
```

For detailed instructions, see:

- `docs/reproducibility.md`
- `docs/user_guide.md`
- `docs/figure_reproduction.md`
- `docs/data_availability.md`
