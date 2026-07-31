# Fontainebleau Validation Protocol

This protocol defines the independent Fontainebleau sandstone validation and
porosity-extrapolation workflow.

## Dataset

The validation data are four Fontainebleau sandstone digital rock samples from
a previously reported Australian National University dataset. They are not
redistributed in this repository.

- Volume size: `480 x 480 x 480` voxels
- Voxel resolution: `5.68 um/voxel`
- Porosities: `0.2045`, `0.1743`, `0.1263`, and `0.0853`

Place the four binary `uint8` volumes at the paths declared under
`real_volumes` in `configs/fontainebleau_config.yaml`:

```text
data/fontainebleau/
|-- phi0p2045.raw
|-- phi0p1743.raw
|-- phi0p1263.raw
`-- phi0p0853.raw
```

## Training And Conditioning

The model is trained on the `0.2045` volume and evaluated at the training
porosity and three unseen porosities.

- VQ-VAE training: 80 epochs
- Latent DDPM training: 150 epochs
- Training volume porosity: `0.2045`
- FiLM normalization: `(phi - 0.13) / 0.02`
- Generated samples per target: 50
- Comparison patch size: `256 x 256 x 256` voxels

For each `480 x 480 x 480` parent volume, the workflow extracts 50 overlapping
`256 x 256 x 256` subvolumes using a 32-voxel stride and a 32-voxel minimum
origin separation. These are overlapping subvolumes from one parent rock
volume, not 50 independent rock specimens. Shaded bands therefore describe
dispersion among overlapping subvolumes and must not be interpreted as
statistical uncertainty estimated from 50 independent specimens.

The training porosity and FiLM normalization center are different quantities
and must not be substituted for one another.

## Official Commands

Run the independent validation workflow:

```bash
python run_pipeline.py --mode fontainebleau --config configs/main.yaml
```

Use the released final checkpoints without retraining:

```bash
python run_pipeline.py --mode final --config configs/main.yaml
```

The workflow:

1. extracts porosity-matched real 256-cubed patches from all four volumes;
2. generates samples at all four target porosities;
3. compares directional `S2`, lineal path, and EDT pore-size statistics;
4. compares voxel connectivity and OpenPNM permeability;
5. compares coordination, Euler, and PNM six-panel descriptors;
6. writes Fontainebleau outputs under `results/fig_fontainebleau/` and
   comparison tables under `results/tables/`.
