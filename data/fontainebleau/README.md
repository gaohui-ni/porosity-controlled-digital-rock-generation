# data/fontainebleau

The four ANU Fontainebleau volumes are third-party data and are not
redistributed because the authors do not hold redistribution rights. Users
must obtain them from the original data provider.

Before analysis, standardize each locally obtained volume to binary `uint8`
format with `0 = solid`, `1 = pore`, and a common shape of
`480 x 480 x 480` voxels. Original packaging and phase-label conventions may
vary by provider version. Declare the standardized local paths in
`configs/fontainebleau_config.yaml`.
