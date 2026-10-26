# Fuel Reproduction Entry Point

This filename is retained for historical links. The active submission target
is **Fuel**, not Computers & Geosciences.

Use the second-version main-sandstone command:

```bash
python run_pipeline.py --mode spatial --config configs/main.yaml
```

The historical Fontainebleau compatibility workflow requires three external
ANU volumes, which are not redistributed. It must not be presented as direct
reproduction of Fuel Fig. 11 or Table 3 until the final checkpoint and sample
provenance are available:

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
