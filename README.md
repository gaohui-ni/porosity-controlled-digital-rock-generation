# 孔隙度可控的三维多孔介质生成

本仓库是论文 **A Porosity-Controllable Generative Framework for
Three-Dimensional Porous Media Based on Discrete Latent-Space Diffusion** 的
核心代码实现。

## 作者

Gaohui Ni, Yanyan Ma, Shaowei Ma, Yuxin Yang, Xuefeng Liu, and Hao Ni

## 方法简介

第二版论文采用“离散潜空间生成 + 双重孔隙度控制”的三维数字岩心生成框架：

1. 使用三维 VQ-VAE 将 `256^3` 微米 CT 子体压缩为
   `32 x 64 x 64 x 64` 离散潜变量；
2. 使用 FiLM 将目标孔隙度注入潜空间 DDPM 的去噪过程；
3. 使用拓扑保持自适应分位数二值化（TP-AQB）精确匹配最终孔隙度，同时
   尽量保护骨架、喉道邻域、连通性和比表面积。

主砂岩实验使用空间上互不重叠的训练、验证和测试区域。最终比较包括测试区
95 个真实样本和 95 个对应条件下的生成样本。目标孔隙度为 `0.11`、`0.12`、
`0.13` 和 `0.14`，四组分别包含 8、29、39 和 19 对样本。

Fontainebleau 实验在同一岩性内验证跨样本孔隙度外推能力：使用一块
`480^3` 岩心训练，并在两块更低孔隙度岩心上验证。该实验单独训练模型，
并不是主砂岩模型的零样本迁移，也不表示跨岩性泛化。

更详细的方法、代码位置和证据范围见
[docs/methodological_novelty.md](docs/methodological_novelty.md)。

## 仓库内容

- 三维 VQ-VAE 与离散潜变量表示学习；
- FiLM 条件潜空间 DDPM；
- TP-AQB v2：在满足目标孔隙体素数的同时保护关键拓扑结构；
- 空间隔离的训练、验证和测试流程；
- EMA 向量量化、验证集选模、FiLM 消融和条件校准采样；
- 训练、采样、批量生成和 Fontainebleau 验证脚本；
- 孔隙度、定向两点相关函数 `S2`、比表面积、孔喉网络、拓扑和渗透率评估；
- 无需受限原始数据即可运行的轻量合成示例。

仓库仍保留轻量 top-N 分位数二值化代码，用于演示和旧权重兼容。论文第二版
所述的主要后处理方法是 TP-AQB，二者不能混同。

## 目录结构

```text
configs/      YAML 配置文件
src/          模型、数据、训练、采样、指标和工具代码
scripts/      训练、生成和评估命令行脚本
examples/     小型合成数据和演示输出
docs/         安装、使用、复现和数据/代码可用性说明
data/         用户提供原始数据的占位目录
outputs/      新训练权重和中间输出目录
results/      图、曲线、表格和汇总结果目录
savedmodels/  通过 Git LFS 发布的旧流程兼容权重
environment/  论文实验环境的依赖版本
```

## 安装

推荐使用 Conda：

```bash
conda env create -f environment/environment-lock.yml
conda activate vq256_cuda
pip install -e .
```

也可以直接安装依赖：

```bash
pip install -r requirements.txt
pip install -r requirements_optional.txt
pip install -e .
```

轻量演示不需要 `porespy` 和 `openpnm`；孔喉网络与渗透率计算需要安装
`requirements_optional.txt`。完整环境说明见
[docs/dependency_versions.md](docs/dependency_versions.md)。

## 计算与数据要求

- 轻量演示、Notebook 和单元测试可仅使用 CPU；
- `256^3` 尺度的训练和批量生成建议使用支持 CUDA 的 NVIDIA GPU；
- 孔喉网络和渗透率计算对内存和时间要求较高；
- 主砂岩原始数据默认路径为 `data/raw/Bei_800x800x800.raw`，形状为
  `800 x 800 x 800`；
- 主砂岩体素尺寸为 `3.5e-6` m/voxel，在 `configs/main.yaml` 中配置；
- Fontainebleau 实验需要用户自行取得三块外部 ANU 岩心数据。

## 最小演示

```bash
python scripts/demo_quantile_binarization.py
```

该命令会：

- 写入 `examples/demo_input.npy`；
- 写入 `examples/demo_output/demo_seg.npy`；
- 输出目标孔隙度、实际孔隙度和自适应阈值。

此演示只检查轻量分位数匹配代码，不需要原始微米 CT 数据或模型权重。相关
Notebook 位于：

```text
notebooks/tutorials/0001-basic-usage.ipynb
notebooks/tutorials/0001-basic-usage-synthetic-rock.ipynb
notebooks/tutorials/0002-porosity-control.ipynb
notebooks/tutorials/0003-fontainebleau-validation.ipynb
```

## 论文第二版复现入口

仓库顶层的 `run_pipeline.py` 是统一入口；`scripts/run_pipeline.py` 是其调用的
内部实现。

主砂岩第二版流程：

```bash
python run_pipeline.py --mode spatial --config configs/main.yaml
```

Fontainebleau 三岩心实验需要在外部数据就绪后单独运行：

```bash
python run_pipeline.py --mode fontainebleau --config configs/main.yaml
```

可用模式：

- `demo`：轻量合成孔隙度控制检查；
- `spatial`：论文第二版的空间隔离 EMA-VQ/DDPM、FiLM 校准采样和 TP-AQB；
- `fontainebleau`：单独训练的 Fontainebleau 跨样本孔隙度外推实验；
- `main`：旧版主砂岩权重兼容流程；
- `full`：旧版 `main` 后接 `fontainebleau` 的兼容流程；
- `final`：使用 `savedmodels/` 中兼容权重进行推理和评估，不重新训练。

`spatial` 是与论文第二版对应的主流程。空间划分、实现细节和来源见
[docs/spatial_vqaware_workflow.md](docs/spatial_vqaware_workflow.md)。

当前 `savedmodels/` 中的文件属于旧版兼容流程。论文第二版使用的空间模型
权重以及冻结的 95 样本坐标清单尚未包含在仓库中。因此，当前代码可以重新
训练和执行完整方法，但精确复现论文中的 95 对样本还需要补充这些文件。

## 检查流程但不启动训练

```bash
python run_pipeline.py --mode spatial --config configs/main.yaml --dry-run
```

`--dry-run` 只生成有序命令和流程清单，不会训练模型、生成样本或执行评估。
输出清单为 `results/pipeline_spatial_manifest.json`。

轻量审稿检查：

```bash
python run_demo.py
```

## 输出位置

- 当前空间流程权重和中间结果：`outputs/spatial_vqaware/`；
- TP-AQB 生成体：`data/generated_phi_sets_spatial/`；
- 空间统计结果：`results/fig_s2/`；
- 孔喉网络结果：`results/fig_pnm/`；
- 渗透率结果：`results/fig_perm/`；
- Fontainebleau 结果：`results/fig_fontainebleau/`；
- 表格：`results/tables/`；
- 图源数据：`results/source_data/`；
- 汇总：`results/results_summary.json`、`results/summary.json` 和
  `results/results_summary.csv`。

## 论文图件对应关系

论文第二版包括 Fig. 1–14。完整的图件—代码—输入—输出对应关系见：

- [docs/figure_mapping.md](docs/figure_mapping.md)
- [docs/figure_reproduction.md](docs/figure_reproduction.md)

其中主要数值图包括：

- Fig. 5：FiLM 条件消融；
- Fig. 6：95 对样本的 Real、Gen1 和 Gen2 孔隙度比较；
- Fig. 8：定向 `S2` 和比表面积；
- Fig. 9：孔半径、喉道半径、形状因子、曲折度、喉道长度和配位数；
- Fig. 10：主砂岩方向渗透率和几何平均渗透率；
- Fig. 11–14：Fontainebleau 孔隙度分布、空间统计、孔喉网络和渗透率。

## Fontainebleau 数据口径

论文报告的整块岩心实测孔隙度与仓库历史文件标签不同：

| 仓库标签 | 论文实测整块孔隙度 | 用途 |
| ---: | ---: | --- |
| 0.2045 | 0.2041 | 训练岩心 |
| 0.1743 | 0.1766 | 未见验证岩心 |
| 0.1263 | 0.1289 | 未见验证岩心 |

每个条件比较 20 个真实 `256^3` 子体和 20 个生成体。真实子体来自对应的一块
母岩心，不能视为 20 个相互独立的岩石样本。详细协议见
[docs/fontainebleau_protocol.md](docs/fontainebleau_protocol.md)。

训练轮数配置在本次文档更新中保持不变，等待后续单独确认。

## 测试

```bash
pip install -r requirements_dev.txt
pytest tests/
```

测试覆盖分位数匹配、TP-AQB 目标孔隙体素数与拓扑保护、孔隙度、定向两点
相关函数、流程 dry-run、图号注册表和 Fontainebleau 配置一致性。

## 结果汇总

```bash
python src/metrics/summarize_all.py --root results
python scripts/plot_all.py
```

轻量图件路径检查：

```bash
python scripts/reproduce_figures.py --config configs/main.yaml
```

该脚本默认只运行合成演示并检查图件输出路径；只有显式添加
`--run-full-pipeline` 时才会启动论文尺度流程。

## 数据可用性

主砂岩微米 CT 数据及元数据发布于 Mendeley Data：

https://doi.org/10.17632/vp2yw9c7jj.1

Fontainebleau 岩心属于第三方 ANU 数据，本仓库不重新分发。用户需要从原始
提供方获取并按照 `0 = solid`、`1 = pore` 的 `uint8` 约定进行标准化。详见
[docs/data_availability.md](docs/data_availability.md)。

## 兼容模型权重

当前 Git LFS 中的权重用于旧版兼容流程，不是论文第二版 `spatial` 流程的
`ae_best.pth` 和 `unet_best_noise.pth`。

```bash
git lfs install
git clone https://github.com/gaohui-ni/porosity-controlled-digital-rock-generation.git
cd porosity-controlled-digital-rock-generation
git lfs pull
python run_pipeline.py --mode final --config configs/main.yaml
```

权重布局：

```text
savedmodels/
|-- main_sandstone/
|   |-- vqvae_final.pth
|   |-- unet_final.pth
|   `-- latent_stats.npz
`-- fontainebleau_phi0p2045/
    |-- vqvae_final.pth
    |-- unet_final.pth
    `-- latent_stats.npz
```

文件大小、SHA256 和适用范围见：

- [docs/model_manifest.md](docs/model_manifest.md)
- [docs/checkpoints.md](docs/checkpoints.md)

扩散采样具有随机性，且数值结果可能受到硬件和软件环境影响。独立重训练应复现
方法和统计趋势，而不保证生成完全相同的体数据。

## 引用

如果使用本仓库，请引用相关论文。机器可读引用信息见
[CITATION.cff](CITATION.cff)。

## 许可证

本项目使用 MIT License，详见 [LICENSE](LICENSE)。
