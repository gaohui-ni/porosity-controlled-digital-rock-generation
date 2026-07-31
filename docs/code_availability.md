# Computer Code Availability

**Name of code:** Porosity-Controlled 3D Digital Rock Generation.

**Developer and technical contact:** Gaohui Ni, College of Science, China
University of Petroleum (East China), No. 66 West Changjiang Road, Huangdao
District, Qingdao, Shandong 266580, China; telephone: +86-151-9298-7839;
email: 1946978288@qq.com.

**Year first available:** 2026.

**Hardware requirements:** A standard CPU is sufficient for the lightweight
demonstration and unit tests. A CUDA-capable NVIDIA GPU is recommended for
model training and manuscript-scale generation. The experiments reported in
this study were conducted using an NVIDIA A100 GPU with 40 GB memory.

**Software requirements:** Python 3.10, PyTorch 2.5.1, NumPy 2.2.6, SciPy
1.15.3, PoreSpy 3.0.2, OpenPNM 3.5.2, and the additional dependencies
documented in the repository.

**Programming language:** Python.

**Program size:** Approximately 2.73 MB of uncompressed Git-tracked release
content, excluding pretrained checkpoints, raw data, generated outputs, and
Git metadata. This value was calculated from 125 tracked files after excluding
`savedmodels/`, `data/`, `results/`, and `outputs/`.

**Source code availability:** The source code is publicly available in the
GitHub repository *Porosity-Controlled 3D Digital Rock Generation* under the
MIT License:
https://github.com/gaohui-ni/porosity-controlled-digital-rock-generation. The
pretrained checkpoints are distributed through Git LFS, with file sizes and
SHA-256 checksums documented in `docs/model_manifest.md`.

The released checkpoints support model inference and analysis. Complete
Fontainebleau validation additionally requires access to the external ANU
volumes, which are not redistributed in this repository. Due to stochastic
sampling and hardware- or software-dependent numerical differences, generated
volumes may not be bitwise identical across computational environments.
