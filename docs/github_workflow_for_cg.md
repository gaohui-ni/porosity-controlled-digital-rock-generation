# Computers & Geosciences Reproduction Entry Point

Use the second-version main-sandstone command:

```bash
python run_pipeline.py --mode spatial --config configs/main.yaml
```

The separate Fontainebleau validation requires three external ANU volumes,
which are not redistributed:

```bash
python run_pipeline.py --mode fontainebleau --config configs/main.yaml
```

For detailed instructions, see:

- `docs/reproducibility.md`
- `docs/user_guide.md`
- `docs/figure_reproduction.md`
- `docs/data_availability.md`

The `main`, `full`, and `final` modes remain available for earlier-workflow
checkpoint compatibility.
