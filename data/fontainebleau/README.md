# data/fontainebleau

The three ANU Fontainebleau volumes used by the second-version manuscript are
third-party data and are not
redistributed because the authors do not hold redistribution rights. Users
must obtain them from the original data provider.

Before analysis, standardize each locally obtained volume to binary `uint8`
format with `0 = solid`, `1 = pore`, and a common shape of
`480 x 480 x 480` voxels. Original packaging and phase-label conventions may
vary by provider version. Declare the standardized local paths in
`configs/fontainebleau_config.yaml`.

The measured whole-volume porosities and roles are:

- `0.2041`: training parent rock (historical filename label `0.2045`);
- `0.1766`: extrapolation rock (historical filename label `0.1743`);
- `0.1289`: extrapolation rock (historical filename label `0.1263`).

Historical filenames are retained for compatibility and must not be read as
the current measured porosities. A fourth historical rock is not part of the
three-condition experiment reported in the current manuscript.
