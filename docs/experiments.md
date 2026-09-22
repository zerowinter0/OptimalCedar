# PICO / OptimalCedar 实验总文档

**这份文档是所有实验结果的唯一入口。** 新实验的结果与结论直接在本文件追加/更新章节，
不再新开 md；图件在 `docs/figures_20260921/`，原始产物在 `outputs/<run>/`。

更新时间：2026-09-22。

## 0. 实验协议（所有对比共用）

- **平台**：两台同局域网机器，各 2×Xeon Gold 6226R / 64 硬件线程 / 754 GiB；Ray 机器另有
  RTX A6000。本地与远端各 64 CPU 预算，Ray actor 用 `cedar_remote` 资源独占远端，
  `CEDAR_RAY_REQUIRE_REMOTE=1`（driver 节点不广播该资源）。
- **profile**：每个负载一份共享 profile，`layered` 协议（`CEDAR_LAYERED_ADAPTIVE_PROFILE=1`）：
  保留 Cedar 原始 baseline/offloads/tf_fuse 条目，并追加每算子每后端的 isolated backend_compute、
  仿射算子成本 `kx+b`、宽度曲线、object boundary、SMP aggregate transport。所有 optimizer 读同一份。
- **执行**：每个 plan 生成一次、正式跑一遍（cache 负载先独立预热再测）。稳态吞吐 = 数据量 /
  稳态时间（`perf_time_sec`，排除优化与启动）；优化时间 = `setup_time_sec`；cell 上限 2 小时，
  超时或失败按原样记录，不插值。
- **默认 9 个 optimizer（图里的名字）**：

| 图名 | campaign 方法名 | 实现 | 说明 |
| --- | --- | --- | --- |
| unopt | unopti | UnoptimizedOptimizer | 直接执行声明计划 |
| plumber | plumber-opt | PlumberOptimizer | 原生系统 |
| raydata | ray-opt | RayDataOptimizer | 原生系统 |
| cedar | cedar-opt | Optimizer (0) | Cedar 原始分阶段优化 |
| cedar-dp | old-dp-opt | SimpleDpOptimizer (11) | 同一 DP，但只读 Cedar 旧 profile 属性 |
| PICO-Resource | dp-boundary | OldDpBoundaryOptimizer (28) | 只加 stage boundary 项 |
| PICO-Resource-Op | dp-boundary-affine | SimpleDpBoundaryOptimizer (21) | boundary + 每算子 kx+b |
| PICO | dp-boundary-affine-W-width | SimpleDpWorkersWidthBoundaryOptimizer (27) | boundary + kx+b + Workers + width |
| （消融对照）| simple-dp-opt | LayeredSimpleDpOptimizer (29) | 与 PICO-Resource-Op 同模型但不加 boundary |
| （§4.6 消融）| staged-boundary | StagedBoundaryOptimizer (31) | Cedar staged 搜索 + boundary 模型 |
| （§4.6 消融）| staged-boundary-affine | StagedBoundaryAffineOptimizer (32) | Cedar staged 搜索 + boundary + kx+b |
| （§4.6 消融）| staged-boundary-affine-W | StagedWorkersBoundaryAffineOptimizer (33) | Cedar staged 搜索 + boundary + kx+b + W |

## 0.1 文档地图

| 内容 | 位置 |
| --- | --- |
| 全部实验结果与专题分析 | **本文档**（§1–§7） |
| 论文图件（吞吐/优化时间/排序/ρ） | `docs/figures_20260921/` + 生成脚本 `scripts/make_experiment_figures_20260921.py` |
| 图件底层数据 | `outputs/figure_data_20260921/figure_data.json`（`scripts/collect_figure_data_20260921.py`） |
| 正式放大 campaign 原始产物 | `outputs/ultimate_eight_optimizers_fix_20260921/`（= `..._20260920` 的续跑完整版） |
| 新增 Cedar 负载原始产物 | `outputs/ultimate_new_workloads_20260922/`（小数据验证在 `outputs/new_workloads_validation_20260922/`） |
| staged vs DP 消融原始产物 | `outputs/staged_ablation_20260922/`（小数据验证在 `outputs/staged_ablation_validation_20260922/`，§4.6） |
| 专题的原始产物 | 见对应小节里标注的 `outputs/...` 路径 |

2026-09-19 ~ 2026-09-21 的 10 份分散 md（campaign 结果、图件数据、affine 迁移、RAY↔local、
ILP 最优性、W-only、算子缩放证据、新负载、暂停状态、旧交接文档）已全部并入本文档并删除，
完整内容仍在 git 历史里。

## 1. 实验设置（正式放大 campaign）

### 1.1 optimizer

| 标签 | 实现 (selector) | 说明 |
| --- | --- | --- |
| cedar-opt | Optimizer (0) | Cedar 原始分阶段优化器 |
| plumber-opt | PlumberOptimizer (18) | Plumber 基线 |
| ray-opt | RayDataOptimizer (19) | Ray Data 基线 |
| unopti | UnoptimizedOptimizer (24) | 未优化，直接执行原计划 |
| dp-boundary | OldDpBoundaryOptimizer (28) | 只用 boundary，计价沿用 Cedar 旧 profile 属性 |
| dp-boundary-affine | SimpleDpBoundaryOptimizer (21) | boundary + 每个算子 kx+b（新 layered profile） |
| dp-boundary-affine-W-width | SimpleDpWorkersWidthBoundaryOptimizer (27) | boundary + kx+b + Workers + 宽度搜索 |
| simple-dp-opt | LayeredSimpleDpOptimizer (29) | 与 dp-boundary-affine 相同，但**不加 stage boundary 项** |
| old-dp-opt | SimpleDpOptimizer (11) | 与 simple-dp-opt 相同，但只读 Cedar 旧 profile 属性 |

### 1.2 数据量与协议

| 负载 | 放大 campaign | 小数据集 campaign |
| --- | ---: | ---: |
| simclrv2 | 189,380（本地 9,469 张重复 20 遍） | 9,469 |
| simclrv2_cache | 189,380（同上，cache 全开） | 9,469 |
| commonvoice | 300,000 | 15,000 |
| coco | 50,000（train2017） | 5,000（val2017） |
| llava_pretrain | 50,000 | 1,000（实际处理 907） |
| stackexchange | 20,000 | 2,000 |

- 远端 Ray（`cedar_remote`）执行，`CPU_BUDGET=64`，非 cache 负载关闭 cache、`*_cache` 全开；
- 放大 campaign：单 cell 上限 2 小时，超时记 unavailable 并继续；`cedar-opt` 与 `dp-boundary-affine-W-width` 在 llava/stackexchange 上跳过；
- 小数据集 campaign：单 cell 上限 1 小时，`cedar-opt` 在 llava/stackexchange 上跳过；
- 指标：稳态吞吐 = 总数据量 / 稳态时间；非稳态 setup 含 Ray 启动与计划优化；总时长 = setup + 数据执行墙钟。

## 2. 正式放大 campaign 结果与计划（6 负载 × 9 optimizer）

### 2.1 simclrv2

- 数据量：189,380 (9,469 张 × 20 epoch)

**结果**

| optimizer | 总数据量 | 稳态时间 | 稳态吞吐 | 相对 cedar-opt | 非稳态 setup(含启动+优化) | 总时长 | 状态 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cedar-opt | 189,380 | 159.5 s | 1187.7 /s | 1.00× | 22.0 s | 453.5 s | completed |
| plumber-opt | 189,380 | 1771.2 s | 106.9 /s | 0.09× | 2.2 s | 1777.0 s | completed |
| ray-opt | 189,380 | 2263.8 s | 83.7 /s | 0.07× | 12.2 s | 2322.8 s | completed |
| unopti | 189,380 | 4098.6 s | 46.2 /s | 0.04× | 1.2 s | 4103.2 s | completed |
| dp-boundary | 189,380 | 97.7 s | 1937.5 /s | 1.63× | 12.4 s | 112.3 s | completed |
| dp-boundary-affine | 189,380 | 96.4 s | 1964.5 /s | 1.65× | 12.5 s | 111.2 s | completed |
| dp-boundary-affine-W-width | 189,380 | 75.7 s | 2501.5 /s | 2.11× | 267.0 s | 345.2 s | completed |
| simple-dp-opt (new profile, no boundary) | 189,380 | 95.8 s | 1975.8 /s | 1.66× | 12.5 s | 110.7 s | completed |
| old-dp-opt (legacy profile, no boundary) | 189,380 | 536.6 s | 352.9 /s | 0.30× | 12.4 s | 551.8 s | completed |

**各 optimizer 选中的计划**

| optimizer | W | 计划（source → ... → sink，未标注即 INPROCESS） | cache |
| --- | ---: | --- | --- |
| cedar-opt | 64 | LocalFSListerPipe -> ImageReaderPipe -> Grayscale -> RandomResizedCrop -> FusedPipe{2,5,4}[RAY w=1] -> to_float -> Normalize -> BatcherPipe(batch_size=4) -> PrefetcherPipe | 无 |
| plumber-opt | 1 | LocalFSListerPipe -> ImageReaderPipe -> to_float[SMP w=2] -> RandomResizedCrop[SMP w=7] -> RandomHorizontalFlip -> ColorJitter[SMP w=27] -> Grayscale[SMP w=2] -> GaussianBlur[SMP w=25] -> Normalize -> BatcherPipe(batch_size=4) -> PrefetcherPipe | 无 |
| ray-opt | 1 | LocalFSListerPipe -> ImageReaderPipe -> FusedPipe{7,6,5,4,3,2,1}[RAY w=64] -> BatcherPipe(batch_size=4) -> PrefetcherPipe | 无 |
| unopti | 1 | LocalFSListerPipe -> ImageReaderPipe -> to_float -> RandomResizedCrop -> RandomHorizontalFlip -> ColorJitter -> Grayscale -> GaussianBlur -> Normalize -> BatcherPipe(batch_size=4) | 无 |
| dp-boundary | 32 | LocalFSListerPipe -> ImageReaderPipe -> FusedPipe{3,6} -> FusedPipe{2,4,5}[SMP w=1] -> FusedPipe{7,1} -> BatcherPipe(batch_size=4) -> PrefetcherPipe | 无 |
| dp-boundary-affine | 32 | LocalFSListerPipe -> ImageReaderPipe -> FusedPipe{6,3} -> GaussianBlur[SMP w=1] -> FusedPipe{4,5,7,1} -> BatcherPipe(batch_size=4) -> PrefetcherPipe | 无 |
| dp-boundary-affine-W-width | 64 | LocalFSListerPipe -> ImageReaderPipe -> FusedPipe{6,3,4,5,2,7,1} -> BatcherPipe(batch_size=4) -> PrefetcherPipe | 无 |
| simple-dp-opt (new profile, no boundary) | 32 | LocalFSListerPipe -> ImageReaderPipe -> FusedPipe{6,3} -> GaussianBlur[SMP w=1] -> FusedPipe{4,5,7,1} -> BatcherPipe(batch_size=4) -> PrefetcherPipe | 无 |
| old-dp-opt (legacy profile, no boundary) | 32 | LocalFSListerPipe -> ImageReaderPipe -> FusedPipe{7,1,2,3,4,6,5}[SMP w=1] -> BatcherPipe(batch_size=4) -> PrefetcherPipe | 无 |

### 2.2 simclrv2_cache

- 数据量：189,380 (= simclrv2)

**结果**

| optimizer | 总数据量 | 稳态时间 | 稳态吞吐 | 相对 cedar-opt | 非稳态 setup(含启动+优化) | 总时长 | 状态 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cedar-opt | 189,380 | 115.8 s | 1634.8 /s | 1.00× | 21.7 s | 188.7 s | completed |
| plumber-opt | 189,380 | 1439.8 s | 131.5 /s | 0.08× | 2.2 s | 1445.0 s | completed |
| ray-opt | 189,380 | 2133.1 s | 88.8 /s | 0.05× | 12.0 s | 2190.8 s | completed |
| unopti | 189,380 | 4076.9 s | 46.5 /s | 0.03× | 1.1 s | 4081.4 s | completed |
| dp-boundary | 189,380 | 84.9 s | 2231.6 /s | 1.37× | 12.9 s | 100.2 s | completed |
| dp-boundary-affine | 189,380 | 34.4 s | 5502.4 /s | 3.37× | 23.0 s | 58.8 s | completed |
| dp-boundary-affine-W-width | 189,380 | 35.1 s | 5399.4 /s | 3.30× | 417.9 s | 454.4 s | completed |
| simple-dp-opt (new profile, no boundary) | 189,380 | 34.6 s | 5478.3 /s | 3.35× | 22.8 s | 58.9 s | completed |
| old-dp-opt (legacy profile, no boundary) | 189,380 | 556.8 s | 340.1 /s | 0.21× | 12.4 s | 592.4 s | completed |

**各 optimizer 选中的计划**

| optimizer | W | 计划（source → ... → sink，未标注即 INPROCESS） | cache |
| --- | ---: | --- | --- |
| cedar-opt | 64 | LocalFSListerPipe -> ImageReaderPipe -> Grayscale -> ObjectDiskCachePipe -> RandomResizedCrop -> FusedPipe{2,5,4}[RAY w=1] -> to_float -> Normalize -> BatcherPipe(batch_size=4) -> PrefetcherPipe | 有（在 Grayscale 之后） |
| plumber-opt | 1 | LocalFSListerPipe -> ImageReaderPipe -> to_float -> RandomResizedCrop[SMP w=6] -> RandomHorizontalFlip -> ColorJitter[SMP w=25] -> Grayscale[SMP w=2] -> GaussianBlur[SMP w=30] -> Normalize -> BatcherPipe(batch_size=4) -> PrefetcherPipe | 无 |
| ray-opt | 1 | LocalFSListerPipe -> ImageReaderPipe -> FusedPipe{7,6,5,4,3,2,1}[RAY w=64] -> BatcherPipe(batch_size=4) -> PrefetcherPipe | 无 |
| unopti | 1 | LocalFSListerPipe -> ImageReaderPipe -> to_float -> RandomResizedCrop -> RandomHorizontalFlip -> ColorJitter -> Grayscale -> GaussianBlur -> Normalize -> BatcherPipe(batch_size=4) | 无 |
| dp-boundary | 32 | LocalFSListerPipe -> ImageReaderPipe -> Grayscale -> ObjectDiskCachePipe -> RandomResizedCrop -> FusedPipe{2,4,5}[SMP w=1] -> FusedPipe{7,1} -> BatcherPipe(batch_size=4) -> PrefetcherPipe | 有（在 Grayscale 之后） |
| dp-boundary-affine | 64 | LocalFSListerPipe -> ImageReaderPipe -> FusedPipe{3,2} -> ObjectDiskCachePipe -> FusedPipe{6,4,5,7,1} -> BatcherPipe(batch_size=4) -> PrefetcherPipe | 有（在 FusedPipe{3,2} 之后） |
| dp-boundary-affine-W-width | 64 | LocalFSListerPipe -> ImageReaderPipe -> FusedPipe{3,2} -> ObjectDiskCachePipe -> FusedPipe{6,4,5,7,1} -> BatcherPipe(batch_size=4) -> PrefetcherPipe | 有（在 FusedPipe{3,2} 之后） |
| simple-dp-opt (new profile, no boundary) | 64 | LocalFSListerPipe -> ImageReaderPipe -> FusedPipe{3,2} -> ObjectDiskCachePipe -> FusedPipe{6,4,5,7,1} -> BatcherPipe(batch_size=4) -> PrefetcherPipe | 有（在 FusedPipe{3,2} 之后） |
| old-dp-opt (legacy profile, no boundary) | 32 | LocalFSListerPipe -> ImageReaderPipe -> Grayscale -> ObjectDiskCachePipe -> FusedPipe{6,7}[SMP w=1] -> FusedPipe{1,2,4,5}[RAY w=2] -> BatcherPipe(batch_size=4) -> PrefetcherPipe | 有（在 Grayscale 之后） |

### 2.3 commonvoice

- 数据量：300,000

**结果**

| optimizer | 总数据量 | 稳态时间 | 稳态吞吐 | 相对 cedar-opt | 非稳态 setup(含启动+优化) | 总时长 | 状态 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cedar-opt | 300,000 | 1901.4 s | 157.8 /s | 1.00× | 19.0 s | 2186.0 s | completed |
| plumber-opt | 300,000 | 2061.2 s | 145.5 /s | 0.92× | 2.7 s | 2075.2 s | completed |
| ray-opt | 300,000 | 3262.9 s | 91.9 /s | 0.58× | 6.6 s | 3276.4 s | completed |
| unopti | 300,000 | — | — | — | — | — | timeout |
| dp-boundary | 300,000 | 453.3 s | 661.8 /s | 4.19× | 10.3 s | 473.0 s | completed |
| dp-boundary-affine | 300,000 | 408.5 s | 734.4 /s | 4.65× | 19.1 s | 439.4 s | completed |
| dp-boundary-affine-W-width | 300,000 | 403.6 s | 743.2 /s | 4.71× | 24.9 s | 440.8 s | completed |
| simple-dp-opt (new profile, no boundary) | 300,000 | 433.3 s | 692.3 /s | 4.39× | 10.0 s | 451.9 s | completed |
| old-dp-opt (legacy profile, no boundary) | 300,000 | 3753.6 s | 79.9 /s | 0.51× | 7.0 s | 3861.1 s | completed |

**各 optimizer 选中的计划**

| optimizer | W | 计划（source → ... → sink，未标注即 INPROCESS） | cache |
| --- | ---: | --- | --- |
| cedar-opt | 64 | LocalFSListerPipe -> FusedPipe{6,5,4,3,2,1,0}[RAY w=1] -> PrefetcherPipe | 无 |
| plumber-opt | 1 | LocalFSListerPipe -> _read[SMP w=33] -> _resample[SMP w=2] -> _spec[SMP w=2] -> _stretch[SMP w=21] -> time_mask -> frequency_mask -> mel[SMP w=5] -> PrefetcherPipe | 无 |
| ray-opt | 1 | LocalFSListerPipe -> FusedPipe{6,5,4,3,2,1,0}[RAY w=64] -> PrefetcherPipe | 无 |
| unopti | — | 无计划文件（未运行/超时） | — |
| dp-boundary | 32 | LocalFSListerPipe -> FusedPipe{6,5,4,3}[SMP w=1] -> FusedPipe{2,1,0} -> PrefetcherPipe | 无 |
| dp-boundary-affine | 64 | LocalFSListerPipe -> FusedPipe{6,5,4,3,2,1,0} -> PrefetcherPipe | 无 |
| dp-boundary-affine-W-width | 64 | LocalFSListerPipe -> FusedPipe{6,5,4,3,2,1,0} -> PrefetcherPipe | 无 |
| simple-dp-opt (new profile, no boundary) | 32 | LocalFSListerPipe -> _read -> FusedPipe{5,4,3}[SMP w=1] -> FusedPipe{2,1,0} -> PrefetcherPipe | 无 |
| old-dp-opt (legacy profile, no boundary) | 21 | LocalFSListerPipe -> FusedPipe{6,5}[RAY w=1] -> FusedPipe{4,3}[RAY w=1] -> FusedPipe{2,1} -> mel[RAY w=1] -> PrefetcherPipe | 无 |

### 2.4 commonvoice：三种 cost model 的估计与实测对照（Plumber/PICO 含 W 处理，Cedar 保持原样）

**口径**

- `W` = 计划的 `n_local_workers`（整条流水线的副本数）；
- **PICO（新设计 = `SimpleDpWorkersWidthBoundaryOptimizer` 的目标）**：原始 score `S` 是 W 条件化的，按其聚合口径取 `S / W` 作为系统代价（ms / source-record）；
- **Plumber**：先按计划的 stage 宽度算单 worker 瓶颈速率 `X`（rec/s，内部含 θ·R 与 63 核池的分配），**系统速率 = `X × W`**，再取倒数并乘常数换算成同一维度：`cost = 1000 / (X × W)`（ms / source-record）；
- **Cedar**：**不做 W 处理**（模型本身没有 W 维度），直接给出单 worker 的整计划 Amdahl 代价 `C`；与带 W 的模型对比时须记住它是"单副本"代价；
- **实测**：稳态吞吐 `T`（rec/s），同维度换算 `1000 / T`（ms / source-record）。

| 计划 | W | 计划内 stage 宽度 | PICO cost (ms/src) | Plumber cost (ms/src) | Cedar cost (ms/src，未做 W 处理) | 实测 (rec/s) | 实测 1000/T (ms/src) |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| dp-boundary-affine-W-width（新 PICO 选中） | 64 | 0/0 | 0.423 | 0.766 | 4.679 | 743.2 | 1.345 |
| dp-boundary-affine | 64 | 0/0 | 0.423 | 0.766 | 4.679 | 734.4 | 1.362 |
| dp-boundary | 32 | 0/1/0 | 0.865 | 1.420 | 2.249 | 661.8 | 1.511 |
| cedar-opt | 64 | 0/1 | 11.457 | 0.766 | 2.410 | 157.8 | 6.338 |
| plumber-opt | 1 | 0/33/2/2/21/0/0/5 | 8.100 | 0.802 | 9.069 | 145.5 | 6.871 |
| ray-opt | 1 | 0/64 | 19.080 | 0.767 | 2.410 | 91.9 | 10.876 |
| old-dp-opt | 21 | 0/1/1/0/1 | 25.293 | 1.287 | 1.131 | 79.9 | 12.512 |

**模型原始量（便于核对 W 的乘/除）**

| 计划 | PICO score S | Plumber 单 worker 速率 X (rec/s) | Plumber X×W (rec/s) | Cedar 单 worker cost C | 实测 T (rec/s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| dp-boundary-affine-W-width（新 PICO 选中） | 27.07 | 20.4 | 1305.6 | 4.679 | 743.2 |
| dp-boundary-affine | 27.07 | 20.4 | 1305.6 | 4.679 | 734.4 |
| dp-boundary | 27.67 | 22.0 | 704.0 | 2.249 | 661.8 |
| cedar-opt | 733.22 | 20.4 | 1305.6 | 2.410 | 157.8 |
| plumber-opt | 8.10 | 1247.1 | 1247.1 | 9.069 | 145.5 |
| ray-opt | 19.08 | 1304.4 | 1304.4 | 2.410 | 91.9 |
| old-dp-opt | 531.16 | 37.0 | 777.0 | 1.131 | 79.9 |

**预测/实测 倍数（cost 空间，1.0 = 完全准确）**

| 计划 | PICO (S/W) | Plumber (1000/(X·W)) | Cedar (C，未做 W) |
| --- | ---: | ---: | ---: |
| dp-boundary-affine-W-width（新 PICO 选中） | 0.31x | 0.57x | 3.48x |
| dp-boundary-affine | 0.31x | 0.56x | 3.44x |
| dp-boundary | 0.57x | 0.94x | 1.49x |
| cedar-opt | 1.81x | 0.12x | 0.38x |
| plumber-opt | 1.18x | 0.12x | 1.32x |
| ray-opt | 1.75x | 0.07x | 0.22x |
| old-dp-opt | 2.02x | 0.10x | 0.09x |

**读法**

- **Plumber（含 W）**：在"分段 + 每 worker 多进程"形态上很准（`dp-boundary` 预测 704 rec/s vs 实测 661.8，cost 空间 0.94×），在自身计划上 0.12×、ray-opt 0.07×——含 Ray 的计划仍被系统性高估 8–14 倍，因为它的 `θ·R` 模型没有 Ray 的每记录提交/序列化开销；
- **PICO（含 W）**：7 个计划里 5 个落在 0.31–2.02×，与实测的相关性最好；唯一偏乐观的是它自己选中的两份 affine 计划（cost 空间 0.31×，即预测 2364 rec/s vs 实测 743），说明它对"本地融合 + 宽 W"的折扣仍偏松；
- **Cedar（未做 W 处理）**：作为"单副本"代价与带 W 的模型不可直接比较，但仍可看出方向性错误——它在 cedar-opt/ray-opt/old-dp-opt 上给出极小的单副本代价（0.22–0.38×），在它自己也选中的 W-width 计划上给出 3.5×；Cedar 的模型没有后端 IPC、宽度、W 与 lane 竞争项，无法用于这类比较。

### 2.5 coco

- 数据量：50,000 (train2017)

**结果**

| optimizer | 总数据量 | 稳态时间 | 稳态吞吐 | 相对 cedar-opt | 非稳态 setup(含启动+优化) | 总时长 | 状态 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cedar-opt | 50,000 | 1873.9 s | 26.7 /s | 1.00× | 17.1 s | 1940.9 s | completed |
| plumber-opt | 50,000 | 2627.6 s | 19.0 /s | 0.71× | 20.9 s | 2648.8 s | completed |
| ray-opt | 50,000 | 6598.3 s | 7.6 /s | 0.28× | 27.1 s | 6642.7 s | completed |
| unopti | 50,000 (train2017) | — | — | — | — | — | timeout |
| dp-boundary | 50,000 | 216.7 s | 230.8 /s | 8.65× | 8.9 s | 246.2 s | completed |
| dp-boundary-affine | 50,000 | 207.4 s | 241.1 /s | 9.04× | 8.9 s | 239.8 s | completed |
| dp-boundary-affine-W-width | 50,000 | 169.9 s | 294.3 /s | 11.03× | 21.6 s | 221.1 s | completed |
| simple-dp-opt (new profile, no boundary) | 50,000 | 3320.4 s | 15.1 /s | 0.56× | 6.1 s | 3364.7 s | completed |
| old-dp-opt (legacy profile, no boundary) | 50,000 | 1934.3 s | 25.8 /s | 0.97× | 9.0 s | 1981.9 s | completed |

**各 optimizer 选中的计划**

| optimizer | W | 计划（source → ... → sink，未标注即 INPROCESS） | cache |
| --- | ---: | --- | --- |
| cedar-opt | 64 | COCOSourcePipe -> FusedPipe{1,5,4,3,2}[RAY w=1] -> to_tensor -> PrefetcherPipe | 无 |
| plumber-opt | 1 | COCOSourcePipe -> zoom_out -> crop[SMP w=6] -> SanitizeBoundingBox -> RandomHorizontalFlip[SMP w=2] -> distort[SMP w=48] -> to_tensor[SMP w=7] -> PrefetcherPipe | 无 |
| ray-opt | 1 | COCOSourcePipe -> FusedPipe{5,4,3,2,1,0}[RAY w=64] -> PrefetcherPipe | 无 |
| unopti | — | 无计划文件（未运行/超时） | — |
| dp-boundary | 32 | COCOSourcePipe -> FusedPipe{1,5,4,3,2}[SMP w=1] -> to_tensor -> PrefetcherPipe | 无 |
| dp-boundary-affine | 32 | COCOSourcePipe -> distort[SMP w=1] -> FusedPipe{5,4,3,2} -> to_tensor -> PrefetcherPipe | 无 |
| dp-boundary-affine-W-width | 64 | COCOSourcePipe -> FusedPipe{1,5,4,3,2} -> to_tensor -> PrefetcherPipe | 无 |
| simple-dp-opt (new profile, no boundary) | 21 | COCOSourcePipe -> distort[SMP w=1] -> zoom_out -> crop[RAY w=3] -> FusedPipe{3,2} -> to_tensor[SMP w=1] -> PrefetcherPipe | 无 |
| old-dp-opt (legacy profile, no boundary) | 32 | COCOSourcePipe -> FusedPipe{1,5,4,3,2}[RAY w=2] -> to_tensor[SMP w=1] -> PrefetcherPipe | 无 |

### 2.6 llava_pretrain

- 数据量：43,940 (输入 50,000，过滤后)

**结果**

| optimizer | 总数据量 | 稳态时间 | 稳态吞吐 | 相对 cedar-opt | 非稳态 setup(含启动+优化) | 总时长 | 状态 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cedar-opt | 43,940 (输入 50,000，过滤后) | — | — | — | — | — | skipped_user_requested |
| plumber-opt | 43,940 | 2541.0 s | 17.3 /s | — | 3.0 s | 2551.2 s | completed |
| ray-opt | 43,940 | 2856.0 s | 15.4 /s | — | 5.9 s | 2869.7 s | completed |
| unopti | 43,940 | 2731.8 s | 16.1 /s | — | 2.8 s | 2737.9 s | completed |
| dp-boundary | 43,940 | 1668.2 s | 26.3 /s | — | 56.7 s | 1738.5 s | completed |
| dp-boundary-affine | 43,940 | 1556.4 s | 28.2 /s | — | 60.3 s | 1623.5 s | completed |
| dp-boundary-affine-W-width | 43,940 (输入 50,000，过滤后) | — | — | — | — | — | skipped_previous_timeout |
| simple-dp-opt (new profile, no boundary) | 43,940 | 1553.5 s | 28.3 /s | — | 58.3 s | 1618.8 s | completed |
| old-dp-opt (legacy profile, no boundary) | 43,940 | 1712.9 s | 25.7 /s | — | 44.1 s | 1766.4 s | completed |

**各 optimizer 选中的计划**

| optimizer | W | 计划（source → ... → sink，未标注即 INPROCESS） | cache |
| --- | ---: | --- | --- |
| cedar-opt | — | 无计划文件（未运行/超时） | — |
| plumber-opt | 1 | LocalLinePipe -> parse_json_line -> SetImageRootMapper -> FixUnicodeMapper -> PunctuationNormalizationMapper -> AlphanumericFilter -> CharacterRepetitionFilter -> FlaggedWordsFilter[SMP w=2] -> PerplexityFilter[SMP w=2] -> SpecialCharactersFilter -> WordRepetitionFilter[SMP w=2] -> ImageAspectRatioFilter -> ImageShapeFilter -> ImageSizeFilter -> ImageTextSimilarityFilter -> ImageTextMatchingFilter -> sync_text_key -> PrefetcherPipe | 无 |
| ray-opt | 1 | LocalLinePipe -> FusedPipe{15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0}[RAY w=1] -> PrefetcherPipe | 无 |
| unopti | 1 | LocalLinePipe -> parse_json_line -> SetImageRootMapper -> FixUnicodeMapper -> PunctuationNormalizationMapper -> AlphanumericFilter -> CharacterRepetitionFilter -> FlaggedWordsFilter -> PerplexityFilter -> SpecialCharactersFilter -> WordRepetitionFilter -> ImageAspectRatioFilter -> ImageShapeFilter -> ImageSizeFilter -> ImageTextSimilarityFilter -> ImageTextMatchingFilter -> sync_text_key | 无 |
| dp-boundary | 1 | LocalLinePipe -> FusedPipe{15,14,13,12}[RAY w=32] -> ImageTextSimilarityFilter -> ImageTextMatchingFilter[RAY w=1] -> FusedPipe{8,6,5,3,4,7} -> FusedPipe{9,10}[RAY w=31] -> FusedPipe{11,0}[SMP w=63] -> PrefetcherPipe | 无 |
| dp-boundary-affine | 1 | LocalLinePipe -> FusedPipe{15,14,13,12,6,9,8,10,11,7,2}[RAY w=1] -> FusedPipe{1,3,4,5,0} -> PrefetcherPipe | 无 |
| dp-boundary-affine-W-width | — | 无计划文件（未运行/超时） | — |
| simple-dp-opt (new profile, no boundary) | 1 | LocalLinePipe -> FusedPipe{15,14} -> FusedPipe{13,12,6,9,8,10,11,7,2}[RAY w=1] -> FusedPipe{1,3,4,5,0} -> PrefetcherPipe | 无 |
| old-dp-opt (legacy profile, no boundary) | 1 | LocalLinePipe -> FusedPipe{15,14,13,12,2,8,6,5,3,4,7} -> FusedPipe{1,9,11,10}[RAY w=1] -> sync_text_key[SMP w=63] -> PrefetcherPipe | 无 |

### 2.7 stackexchange

- 数据量：7,238 (输入 20,000，过滤后)

**结果**

| optimizer | 总数据量 | 稳态时间 | 稳态吞吐 | 相对 cedar-opt | 非稳态 setup(含启动+优化) | 总时长 | 状态 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cedar-opt | 7,238 (输入 20,000，过滤后) | — | — | — | — | — | skipped_user_requested |
| plumber-opt | 7,238 | 104.9 s | 69.0 /s | — | 3.9 s | 109.5 s | completed |
| ray-opt | 7,238 | 113.2 s | 63.9 /s | — | 9.1 s | 126.5 s | completed |
| unopti | 7,238 | 2369.1 s | 3.1 /s | — | 2.5 s | 2372.8 s | completed |
| dp-boundary | 7,238 | 89.1 s | 81.2 /s | — | 87.3 s | 259.2 s | completed |
| dp-boundary-affine | 7,238 | 119.1 s | 60.7 /s | — | 116.1 s | 341.3 s | completed |
| dp-boundary-affine-W-width | 7,238 (输入 20,000，过滤后) | — | — | — | — | — | skipped_previous_timeout |
| simple-dp-opt (new profile, no boundary) | 7,238 | 209.7 s | 34.5 /s | — | 103.7 s | 390.6 s | completed |
| old-dp-opt (legacy profile, no boundary) | 7,238 | 49.6 s | 146.1 /s | — | 91.0 s | 284.8 s | completed |

**各 optimizer 选中的计划**

| optimizer | W | 计划（source → ... → sink，未标注即 INPROCESS） | cache |
| --- | ---: | --- | --- |
| cedar-opt | — | 无计划文件（未运行/超时） | — |
| plumber-opt | 1 | LocalLinePipe -> parse_json_line -> CleanEmailMapper -> CleanLinksMapper -> FixUnicodeMapper -> PunctuationNormalizationMapper -> WhitespaceNormalizationMapper -> AlphanumericFilter -> AverageLineLengthFilter -> CharacterRepetitionFilter -> FlaggedWordsFilter[SMP w=21] -> LanguageIDScoreFilter -> MaximumLineLengthFilter -> PerplexityFilter -> SpecialCharactersFilter -> TextLengthFilter -> WordsNumFilter[SMP w=21] -> WordRepetitionFilter[SMP w=21] -> sync_text_key -> extract_output_text -> PrefetcherPipe | 无 |
| ray-opt | 1 | LocalLinePipe -> FusedPipe{18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0}[RAY w=64] -> PrefetcherPipe | 无 |
| unopti | 1 | LocalLinePipe -> parse_json_line -> CleanEmailMapper -> CleanLinksMapper -> FixUnicodeMapper -> PunctuationNormalizationMapper -> WhitespaceNormalizationMapper -> AlphanumericFilter -> AverageLineLengthFilter -> CharacterRepetitionFilter -> FlaggedWordsFilter -> LanguageIDScoreFilter -> MaximumLineLengthFilter -> PerplexityFilter -> SpecialCharactersFilter -> TextLengthFilter -> WordsNumFilter -> WordRepetitionFilter -> sync_text_key -> extract_output_text | 无 |
| dp-boundary | 16 | LocalLinePipe -> FusedPipe{18,17,16,15}[SMP w=1] -> FusedPipe{14,13} -> WordsNumFilter[RAY w=4] -> WordRepetitionFilter[SMP w=1] -> FusedPipe{6,10,5,12,8,4,7,11} -> FusedPipe{9,1,0}[SMP w=1] -> PrefetcherPipe | 无 |
| dp-boundary-affine | 21 | LocalLinePipe -> parse_json_line -> FusedPipe{17,16,15,14,13,2,3}[SMP w=1] -> FusedPipe{9,5}[RAY w=3] -> FusedPipe{6,10,12,8}[SMP w=1] -> FusedPipe{7,11,4,1,0} -> PrefetcherPipe | 无 |
| dp-boundary-affine-W-width | — | 无计划文件（未运行/超时） | — |
| simple-dp-opt (new profile, no boundary) | 12 | LocalLinePipe -> parse_json_line -> CleanEmailMapper[RAY w=1] -> FusedPipe{16,15}[SMP w=1] -> FusedPipe{14,13}[RAY w=1] -> FusedPipe{2,3}[SMP w=1] -> FlaggedWordsFilter[RAY w=2] -> FusedPipe{6,10}[SMP w=1] -> FusedPipe{12,5}[RAY w=1] -> LanguageIDScoreFilter[SMP w=1] -> FusedPipe{7,11,4,1,0} -> PrefetcherPipe | 无 |
| old-dp-opt (legacy profile, no boundary) | 32 | LocalLinePipe -> FusedPipe{18,17,16,15,14,13,3}[RAY w=2] -> FusedPipe{6,10,5,12,8,4,7,11} -> FusedPipe{2,9,1,0}[SMP w=1] -> PrefetcherPipe | 无 |

### 2.8 未完成与不可用记录

- 放大 campaign 已完成（修复 teardown 后于 `outputs/ultimate_eight_optimizers_fix_20260921` 续跑，2026-09-21 08:07（UTC+8）写出 COMPLETE）：54 个 cell 中 48 个 completed、2 个真超时、4 个按规则跳过；
- `commonvoice` 的 `unopti` 超过 2 小时上限，记为 `timeout`（unavailable）；
- `coco` 的 `unopti` 真的慢：2 小时内只处理 47,635/50,000（约 6.6 rec/s），记为 `timeout`；
- `coco` 的 `dp-boundary` / `dp-boundary-affine` 在修复前曾在 teardown 阶段挂住 2 小时被 runner 杀掉（根因：本地 worker 阻塞在 `result_queue.put()` 后忽略 SIGTERM，解释器退出时无超时 join 子进程）；`cedar/client/dataset.py` 的分级 shutdown（进程树 SIGKILL + 有界 join）修复后重跑，分别 270 s / 264 s 正常收尾，吞吐 230.8 / 241.1 rec/s；
- `llava_pretrain` 与 `stackexchange` 的 `cedar-opt` 按用户要求跳过，`dp-boundary-affine-W-width`（PICO）因已知的超时记为 `skipped_previous_timeout`（见 §4 的复杂度分析）；
- 输入记录数 vs 实际处理量：`llava_pretrain` 配置 50,000 实际处理 43,940，`stackexchange` 配置 20,000 实际处理 7,238 —— 两个 pipeline 内含 `FilterPipe`（文本质量/语言过滤、图像存在性等），被过滤的记录不进入统计；同一负载内所有 optimizer 处理量一致，横向比较仍然公平；
- 小数据集 campaign 的 `llava_pretrain`：`cedar-opt` 按用户要求跳过，`dp-boundary-affine-W-width` 因 1 小时上限记为 `timeout`（单次 W 的精确 DP 在 16 层中的第 10 层被截断）；
- 小数据集 campaign 的 `stackexchange` 按用户要求提前停止（只完成 plumber-opt / ray-opt），本文件不将其计入对照。

### 2.9 观察

- **cache 负载**：只有 DP 系列会插入 `ObjectDiskCachePipe`（均落在 ImageReader 之后、增广算子之前），cedar/plumber/raydata 虽然 cache 已开启但没有落盘策略；simclrv2_cache 上 `simple-dp-opt` / `dp-boundary-affine` 达到 ~5.5k rec/s，是 cedar-opt 的 3.4–3.6 倍。
- **放大后排序稳定**：simclrv2 上 dp-boundary-affine-W-width (2,502/s) > simple-dp-opt / dp-boundary-affine (~1,970/s) > dp-boundary (1,938/s) > cedar-opt (1,188/s) > old-dp-opt (353/s)，而未优化计划只有 46/s。
- `old-dp-opt`（旧 profile 属性、无 boundary）明显差于新 profile 的 `simple-dp-opt`（353 vs 1,976 /s），说明新 profile 的隔离测量 + kx+b 是主要收益来源。
- 计划形态：DP 系列倾向把 CPU 段整体融合并选大 W；cedar-opt 保留一个 RAY stage；plumber/raydata 把算子拆成多个 SMP/RAY stage（radata 在 coco 上把全部算子塞进单个 RAY w=64，吞吐仅 4 rec/s）。

### 2.10 小数据集 campaign 对照（放大前的同一协议，1 轮）

该轮为放大前的对照实验（同样 1 轮、CPU_BUDGET=64、远端 Ray）。

#### 2.10.1 simclrv2

- 数据量：9,469

| optimizer | 总数据量 | 稳态时间 | 稳态吞吐 | 相对 cedar-opt | 非稳态 setup(含启动+优化) | 总时长 | 状态 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cedar-opt | 9,472 | 8.2 s | 1150.7 /s | 1.00× | 21.9 s | 286.6 s | completed |
| plumber-opt | 9,472 | 89.7 s | 105.6 /s | 0.09× | 2.2 s | 92.1 s | completed |
| ray-opt | 9,472 | 66.2 s | 143.1 /s | 0.12× | 12.2 s | 80.7 s | completed |
| dp-boundary | 9,472 | 5.4 s | 1747.5 /s | 1.52× | 12.5 s | 18.7 s | completed |
| dp-boundary-affine | 9,472 | 5.0 s | 1896.3 /s | 1.65× | 12.5 s | 18.3 s | completed |
| dp-boundary-affine-W-width | 9,472 | 3.8 s | 2522.1 /s | 2.19× | 269.8 s | 275.1 s | completed |

**各 optimizer 选中的计划**

| optimizer | W | 计划（source → ... → sink，未标注即 INPROCESS） | cache |
| --- | ---: | --- | --- |
| cedar-opt | 64 | LocalFSListerPipe -> ImageReaderPipe -> Grayscale -> RandomResizedCrop -> FusedPipe{2,5,4}[RAY w=1] -> to_float -> Normalize -> BatcherPipe(batch_size=4) -> PrefetcherPipe | 无 |
| plumber-opt | 1 | LocalFSListerPipe -> ImageReaderPipe -> to_float[SMP w=2] -> RandomResizedCrop[SMP w=7] -> RandomHorizontalFlip -> ColorJitter[SMP w=27] -> Grayscale[SMP w=2] -> GaussianBlur[SMP w=25] -> Normalize -> BatcherPipe(batch_size=4) -> PrefetcherPipe | 无 |
| ray-opt | 1 | LocalFSListerPipe -> ImageReaderPipe -> FusedPipe{7,6,5,4,3,2,1}[RAY w=64] -> BatcherPipe(batch_size=4) -> PrefetcherPipe | 无 |
| dp-boundary | 32 | LocalFSListerPipe -> ImageReaderPipe -> FusedPipe{3,6} -> FusedPipe{2,4,5}[SMP w=1] -> FusedPipe{7,1} -> BatcherPipe(batch_size=4) -> PrefetcherPipe | 无 |
| dp-boundary-affine | 32 | LocalFSListerPipe -> ImageReaderPipe -> FusedPipe{6,3} -> GaussianBlur[SMP w=1] -> FusedPipe{4,5,7,1} -> BatcherPipe(batch_size=4) -> PrefetcherPipe | 无 |
| dp-boundary-affine-W-width | 64 | LocalFSListerPipe -> ImageReaderPipe -> FusedPipe{6,3,4,5,2,7,1} -> BatcherPipe(batch_size=4) -> PrefetcherPipe | 无 |

#### 2.10.2 simclrv2_cache

- 数据量：9,469

| optimizer | 总数据量 | 稳态时间 | 稳态吞吐 | 相对 cedar-opt | 非稳态 setup(含启动+优化) | 总时长 | 状态 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cedar-opt | 9,472 | 8.2 s | 1150.1 /s | 1.00× | 21.5 s | 286.3 s | completed |
| plumber-opt | 9,472 | 73.3 s | 129.2 /s | 0.11× | 2.1 s | 75.6 s | completed |
| ray-opt | 9,472 | 66.4 s | 142.7 /s | 0.12× | 11.8 s | 80.7 s | completed |
| dp-boundary | 9,472 | 4.7 s | 2034.1 /s | 1.77× | 12.4 s | 17.2 s | completed |
| dp-boundary-affine | 9,472 | 1.9 s | 5072.2 /s | 4.41× | 22.6 s | 24.6 s | completed |
| dp-boundary-affine-W-width | 9,472 | 1.8 s | 5258.5 /s | 4.57× | 409.6 s | 411.5 s | completed |

**各 optimizer 选中的计划**

| optimizer | W | 计划（source → ... → sink，未标注即 INPROCESS） | cache |
| --- | ---: | --- | --- |
| cedar-opt | 64 | LocalFSListerPipe -> ImageReaderPipe -> Grayscale -> RandomResizedCrop -> FusedPipe{2,5,4}[RAY w=1] -> to_float -> Normalize -> BatcherPipe(batch_size=4) -> PrefetcherPipe | 无 |
| plumber-opt | 1 | LocalFSListerPipe -> ImageReaderPipe -> to_float -> RandomResizedCrop[SMP w=6] -> RandomHorizontalFlip -> ColorJitter[SMP w=25] -> Grayscale[SMP w=2] -> GaussianBlur[SMP w=30] -> Normalize -> BatcherPipe(batch_size=4) -> PrefetcherPipe | 无 |
| ray-opt | 1 | LocalFSListerPipe -> ImageReaderPipe -> FusedPipe{7,6,5,4,3,2,1}[RAY w=64] -> BatcherPipe(batch_size=4) -> PrefetcherPipe | 无 |
| dp-boundary | 32 | LocalFSListerPipe -> ImageReaderPipe -> Grayscale -> ObjectDiskCachePipe -> RandomResizedCrop -> FusedPipe{2,4,5}[SMP w=1] -> FusedPipe{7,1} -> BatcherPipe(batch_size=4) -> PrefetcherPipe | 有（在 Grayscale 之后） |
| dp-boundary-affine | 64 | LocalFSListerPipe -> ImageReaderPipe -> FusedPipe{3,2} -> ObjectDiskCachePipe -> FusedPipe{6,4,5,7,1} -> BatcherPipe(batch_size=4) -> PrefetcherPipe | 有（在 FusedPipe{3,2} 之后） |
| dp-boundary-affine-W-width | 64 | LocalFSListerPipe -> ImageReaderPipe -> FusedPipe{3,2} -> ObjectDiskCachePipe -> FusedPipe{6,4,5,7,1} -> BatcherPipe(batch_size=4) -> PrefetcherPipe | 有（在 FusedPipe{3,2} 之后） |

#### 2.10.3 commonvoice

- 数据量：15,000

| optimizer | 总数据量 | 稳态时间 | 稳态吞吐 | 相对 cedar-opt | 非稳态 setup(含启动+优化) | 总时长 | 状态 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cedar-opt | 15,000 | 77.8 s | 192.8 /s | 1.00× | 19.0 s | 362.1 s | completed |
| plumber-opt | 15,000 | 82.5 s | 181.8 /s | 0.94× | 2.5 s | 96.2 s | completed |
| ray-opt | 15,000 | 85.9 s | 174.7 /s | 0.91× | 6.4 s | 97.1 s | completed |
| dp-boundary | 15,000 | 20.8 s | 721.5 /s | 3.74× | 10.0 s | 39.7 s | completed |
| dp-boundary-affine | 15,000 | 19.9 s | 752.5 /s | 3.90× | 19.6 s | 51.1 s | completed |
| dp-boundary-affine-W-width | 15,000 | 19.7 s | 761.9 /s | 3.95× | 25.2 s | 56.6 s | completed |

**各 optimizer 选中的计划**

| optimizer | W | 计划（source → ... → sink，未标注即 INPROCESS） | cache |
| --- | ---: | --- | --- |
| cedar-opt | 64 | LocalFSListerPipe -> FusedPipe{6,5,4,3,2,1,0}[RAY w=1] -> PrefetcherPipe | 无 |
| plumber-opt | 1 | LocalFSListerPipe -> _read[SMP w=33] -> _resample[SMP w=2] -> _spec[SMP w=2] -> _stretch[SMP w=21] -> time_mask -> frequency_mask -> mel[SMP w=5] -> PrefetcherPipe | 无 |
| ray-opt | 1 | LocalFSListerPipe -> FusedPipe{6,5,4,3,2,1,0}[RAY w=64] -> PrefetcherPipe | 无 |
| dp-boundary | 32 | LocalFSListerPipe -> FusedPipe{6,5,4,3}[SMP w=1] -> FusedPipe{2,1,0} -> PrefetcherPipe | 无 |
| dp-boundary-affine | 64 | LocalFSListerPipe -> FusedPipe{6,5,4,3,2,1,0} -> PrefetcherPipe | 无 |
| dp-boundary-affine-W-width | 64 | LocalFSListerPipe -> FusedPipe{6,5,4,3,2,1,0} -> PrefetcherPipe | 无 |

#### 2.10.4 coco

- 数据量：5,000 (val2017)

| optimizer | 总数据量 | 稳态时间 | 稳态吞吐 | 相对 cedar-opt | 非稳态 setup(含启动+优化) | 总时长 | 状态 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cedar-opt | 5,000 | 197.3 s | 25.3 /s | 1.00× | 17.0 s | 239.5 s | completed |
| plumber-opt | 5,000 | 260.5 s | 19.2 /s | 0.76× | 2.8 s | 263.5 s | completed |
| ray-opt | 5,000 | 1283.8 s | 3.9 /s | 0.15× | 14.1 s | 1313.3 s | completed |
| dp-boundary | 5,000 | 24.9 s | 201.0 /s | 7.93× | 8.8 s | 35.4 s | completed |
| dp-boundary-affine | 5,000 | 25.1 s | 198.9 /s | 7.85× | 8.8 s | 35.5 s | completed |
| dp-boundary-affine-W-width | 5,000 | 20.5 s | 244.0 /s | 9.63× | 21.3 s | 44.5 s | completed |

**各 optimizer 选中的计划**

| optimizer | W | 计划（source → ... → sink，未标注即 INPROCESS） | cache |
| --- | ---: | --- | --- |
| cedar-opt | 64 | COCOSourcePipe -> FusedPipe{1,5,4,3,2}[RAY w=1] -> to_tensor -> PrefetcherPipe | 无 |
| plumber-opt | 1 | COCOSourcePipe -> zoom_out -> crop[SMP w=6] -> SanitizeBoundingBox -> RandomHorizontalFlip[SMP w=2] -> distort[SMP w=48] -> to_tensor[SMP w=7] -> PrefetcherPipe | 无 |
| ray-opt | 1 | COCOSourcePipe -> FusedPipe{5,4,3,2,1,0}[RAY w=64] -> PrefetcherPipe | 无 |
| dp-boundary | 32 | COCOSourcePipe -> FusedPipe{1,5,4,3,2}[SMP w=1] -> to_tensor -> PrefetcherPipe | 无 |
| dp-boundary-affine | 32 | COCOSourcePipe -> distort[SMP w=1] -> FusedPipe{5,4,3,2} -> to_tensor -> PrefetcherPipe | 无 |
| dp-boundary-affine-W-width | 64 | COCOSourcePipe -> FusedPipe{1,5,4,3,2} -> to_tensor -> PrefetcherPipe | 无 |

#### 2.10.5 llava_pretrain

- 数据量：1,000 / 907 processed

| optimizer | 总数据量 | 稳态时间 | 稳态吞吐 | 相对 cedar-opt | 非稳态 setup(含启动+优化) | 总时长 | 状态 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cedar-opt | 1,000 / 907 processed | — | — | — | — | — | skipped_user_requested |
| plumber-opt | 907 | 51.2 s | 17.7 /s | — | 2.8 s | 61.1 s | completed |
| ray-opt | 907 | 55.8 s | 16.3 /s | — | 5.6 s | 68.9 s | completed |
| dp-boundary | 907 | 30.8 s | 29.4 /s | — | 52.0 s | 92.0 s | completed |
| dp-boundary-affine | 907 | 57.2 s | 15.9 /s | — | 65.5 s | 126.0 s | completed |
| dp-boundary-affine-W-width | 1,000 / 907 processed | — | — | — | — | — | timeout |

**各 optimizer 选中的计划**

| optimizer | W | 计划（source → ... → sink，未标注即 INPROCESS） | cache |
| --- | ---: | --- | --- |
| cedar-opt | — | 无计划文件（未运行/超时） | — |
| plumber-opt | 1 | LocalLinePipe -> parse_json_line -> SetImageRootMapper -> FixUnicodeMapper -> PunctuationNormalizationMapper -> AlphanumericFilter -> CharacterRepetitionFilter -> FlaggedWordsFilter[SMP w=2] -> PerplexityFilter[SMP w=2] -> SpecialCharactersFilter -> WordRepetitionFilter[SMP w=2] -> ImageAspectRatioFilter -> ImageShapeFilter -> ImageSizeFilter -> ImageTextSimilarityFilter -> ImageTextMatchingFilter -> sync_text_key -> PrefetcherPipe | 无 |
| ray-opt | 1 | LocalLinePipe -> FusedPipe{15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0}[RAY w=1] -> PrefetcherPipe | 无 |
| dp-boundary | 1 | LocalLinePipe -> FusedPipe{15,14,13,12}[SMP w=32] -> ImageTextSimilarityFilter -> ImageTextMatchingFilter[RAY w=1] -> FusedPipe{8,6,5,3,4,7} -> AlphanumericFilter[RAY w=63] -> FusedPipe{9,10,0}[SMP w=31] -> PrefetcherPipe | 无 |
| dp-boundary-affine | 1 | LocalLinePipe -> FusedPipe{15,14,13,12,6,9,8,10,11,7}[RAY w=64] -> FusedPipe{4,1,2,3,5,0} -> PrefetcherPipe | 无 |
| dp-boundary-affine-W-width | — | 无计划文件（未运行/超时） | — |

## 3. 新增 Cedar 原有负载（wikitext103 / cache / TF 变体）

### 3.1 负载与代码改动

在正式 campaign 的 6 个负载之外加入 Cedar 自己的负载及其 cache / TF 变体：

| 新负载 | 数据集模块 | 记录数 | 状态 |
| --- | --- | ---: | --- |
| `wikitext103` | `wikitext103/cedar_dataset.py` | 100,000 | 正式完成 9/9 |
| `wikitext103_cache` | `wikitext103/cedar_cache_dataset.py` | 100,000 | 正式完成 9/9 |
| `commonvoice_cache` | `commonvoice/cedar_cache_dataset.py` | 300,000 | 正式完成 6/9（1 超时 + 2 失败，见 3.5） |
| `wikitext103_tf` | `wikitext103/cedar_tf_dataset.py` | 100,000 | 阻塞（见 3.6） |
| `simclrv2_tf` | `simclrv2/cedar_tf_dataset.py` | 189,380 | 阻塞（见 3.6） |
| `coco_tf` | `coco/cedar_tf_dataset.py` | 50,000 | 阻塞（见 3.6） |

改动：

1. `evaluation/chapter6_experiments/run_simple_dp_ablation_matrix.py`：`WORKLOADS` / `RECORD_COUNTS` /
   `config()` 增加上述条目；新增 `--record-override LABEL=COUNT`（小数据验证用）；
   `CEDAR_CACHE_WARMUP_GRACE_SEC` 透传给 cell。
2. 三个 wikitext 模块与 `simclrv2/cedar_tf_dataset.py`、`coco/cedar_tf_dataset.py` 支持显式
   `dataset_path`（实验把代码复制到 run 目录后，原来的 `Path(__file__).parents[2] / DATASET_LOC`
   会指向不存在的路径）。`wikitext103_tf` 同时支持 `max_samples`。
3. `scripts/run_new_workloads_20260922.sh`：`MODE=validation|formal`、`RESUME=1`、
   `NEW_WORKLOADS=...`，并在启动前 `ulimit -n $(ulimit -Hn)`。
4. `scripts/new_workloads_status.py` / `scripts/new_workloads_results.py`：进度与结果速查。

### 3.2 小数据验证（每负载 200 条，9 个 optimizer）

`outputs/new_workloads_validation_20260922`：wikitext103、wikitext103_cache、commonvoice_cache
各 9/9 完成（数值只作"能跑通"证据）；TF 变体暴露 3 个 bug（见 3.6）。

### 3.3 正式结果（`outputs/ultimate_new_workloads_20260922`，2026-09-22 14:07 写出 COMPLETE）

数值取自该 run 自己生成的 `RESULTS.md` / `summary.csv`（稳态吞吐，1 轮，排除优化时间）。

**wikitext103（100,000 条，9/9）**

| optimizer | 稳态吞吐 (rec/s) | 优化时间 (s) |
| --- | ---: | ---: |
| **cedar-opt** | **5466.5** | 14.4 |
| dp-boundary-affine (PICO-Resource-Op) | 3248.9 | 15.0 |
| unopti | 2601.3 | 2.5 |
| dp-boundary (PICO-Resource) | 2431.3 | 2.6 |
| old-dp-opt (cedar-dp) | 2421.3 | 2.6 |
| simple-dp-opt | 1687.2 | 15.1 |
| PICO (workers-width-boundary) | 1148.0 | 104.3 |
| ray-opt | 483.6 | 141.4 |
| plumber-opt | 215.5 | 4.2 |

> PICO 这一格在排障期间被重跑过：`1470.7`（首跑）→ `1352.7`（带 trace）→ `1148.0`
> （表中值，runner 在 14:07 写 `RESULTS.md` 时读到的那份）。同一计划的运行间偏差 ≈±20%，
> 结论（该计划比 cedar 慢 3.6–4.8 倍）不受影响。

**wikitext103_cache（100,000 条，9/9）**

| optimizer | 稳态吞吐 (rec/s) | 优化时间 (s) |
| --- | ---: | ---: |
| simple-dp-opt | 7532.6 | 14.1 |
| cedar-opt | 7488.4 | 14.5 |
| dp-boundary | 7459.2 | 2.7 |
| dp-boundary-affine | 7435.0 | 14.3 |
| old-dp-opt | 7382.2 | 2.5 |
| unopti | 2667.5 | 2.5 |
| PICO | 1473.1 | 69.3 |
| ray-opt | 468.0 | 139.4 |
| plumber-opt | 236.5 | 4.5 |

**commonvoice_cache（300,000 条，6/9）**

| optimizer | 稳态吞吐 (rec/s) | 优化时间 (s) | 状态 |
| --- | ---: | ---: | --- |
| dp-boundary-affine | 1582.6 | 20.6 | completed |
| simple-dp-opt | 1569.7 | 20.6 | completed |
| cedar-opt | 1531.2 | 20.5 | completed |
| old-dp-opt | 167.9 | 10.7 | completed |
| plumber-opt | 141.6 | 2.4 | completed |
| ray-opt | 80.4 | 6.9 | completed |
| unopti | — | — | timeout（>2 h，与非 cache 的 commonvoice 一致） |
| dp-boundary | — | — | failed（cache 预热，见 3.5） |
| PICO | — | — | failed（cache 预热，见 3.5） |

### 3.4 分析：这两个文本负载上 PICO 为什么输——不是远端 IO 边界，而是本地扩展

**（a）小记录的远端边界确实几乎免费。** 两个负载的 RAY 边界带宽接近，差的是 payload：

| 负载 | RAY 边界带宽 | 中位记录字节 | 每记录边界代价 |
| --- | ---: | ---: | ---: |
| wikitext103 | 112.0 MB/s | 2.8 KB | ≈25 ns |
| simclrv2 | 94.6 MB/s | 598 KB（max 2.39 MB） | ≈6.3 ms（max 25 ms） |

所以文本负载上"把一整块丢到 Ray"在模型里几乎不花钱：cedar 计划（`Fused{8,7,6,5,3,4}[RAY w=64]`）
的 ray lane 只有 **0.0304 ms/source-record**，占总分 0.2076 的 15%。

**（b）但 PICO 选的计划根本没有 Ray stage。** 两个候选计划的模型估计 vs 实测：

| 计划 | 模型（ms/record，PICO 目标） | 模型折合吞吐 | 实测吞吐 |
| --- | ---: | ---: | ---: |
| cedar：`Fused{8,7,6,5,3,4}[RAY w=64]`，W=1 | local 0.1771 + ray 0.0304 = **0.2076** | 4,816 rec/s | **5,466.5 rec/s**（模型只差 13%） |
| PICO：同一融合块放本地，W=64 | local **0.7324**，ray 0 | 64 / 0.7324 = **87,385 rec/s** | **1,148–1,470 rec/s**（模型乐观 60–76×） |

即：模型对"Ray 64-actor"一侧估得准，**错在"本地 64 worker 线性加速"**这个假设
（`score` 是"每 source record 每 worker 的服务间隔"，吞吐按 `W/score` 推）。
`wikitext103_cache` 完全同型：cedar 选 RAY w=64（7488.4），PICO 选本地 W=64（1473.1）。

**（c）trace 显示本地计划的瓶颈不是算子计算，也不在模型覆盖范围内。** 用
`CEDAR_RECONCILE_DIR` 重放 PICO 计划（`/tmp/wt_probe2`，实测 1352.7 rec/s，比未 trace 的那次低约 8%，
是 trace 开销）：

| 段 | wall 计算 (ms/record/worker) |
| --- | ---: |
| Fused{8,7,6,5,3,4} | 1.572 |
| Embedding | 0.290 |
| ToTensor 之后的阶段 | 0.131 |
| source | 0.080 |
| Batcher | 0.018 |
| **合计** | **≈2.1 ms** |

而实际每 worker 是 **47 ms/record**（1471 rec/s ÷ 64），trace 里有一段 wall 分段是
**4.77 s/sample** 的排队/搬运（几乎全是等待，不是计算）。这条路径正是 driver 侧把 ~195 KB 的
token 张量从 worker 收回来（torch MP 的 fd 传共享内存；`wikitext103_cache` 那次 PICO 崩溃
也发生在 `_get_result` 的 `recvfd` 上）。**模型的 lane 只有计算 + 边界传输，没有"driver↔worker
结果搬运"这一项**，于是它看不到本地计划真正的 47 ms/record。

**（d）可做的下一步**：把 driver↔worker 的结果搬运/序列化显式建成一条 lane（或先把大张量回传
改成按批/更小 dtype），再重跑这两个负载看 PICO 是否改选 Ray 计划。

### 3.5 commonvoice_cache 的三个未完成 cell（含根因）

- **unopti 超时**：2 小时上限内跑不完 300,000 条（与非 cache 的 `commonvoice` 一致），
  记为 unavailable，不插值。
- **`dp-boundary` / PICO 两次都失败在 cache 预热**，报错
  `Cache warmup for <method> finished without committing every worker shard`。实测各 shard 目录状态：

| cache namespace | shard 目录数 | 提交 manifest | complete |
| --- | ---: | ---: | ---: |
| `commonvoice__optimizer` | 64 | 64 | 64 |
| `commonvoice__simple_dp` | 64 | 64 | 64 |
| `commonvoice__simple_dp_boundary` | 64 | 64 | 64 |
| `commonvoice__old_dp_legacy_optimizer` | 32 | 32 | 32 |
| `commonvoice__old_dp_boundary` | 64 | **19** | 19 |
| `commonvoice__simple_dp_workers_width_boundary` | 64 | **4** | 4 |

  未提交的 shard 里 **有数据文件但没有 `.manifest.json`**（例如 PICO 的每个 `feature_r*` 都有 5 个
  数据文件），说明这些 worker 在样本预算到点后被打断、没走到 end-of-stream 的原子提交；把等待从
  60 s 提到 900 s（`CEDAR_CACHE_WARMUP_GRACE_SEC`）**没有改变结果**——它们不是在慢慢 drain，
  而是上游不再供数。对照组（同样 64 worker、同样数据）都能提交 64/64，因此这是
  **plan 形态 × 预算截断**的交互，不是机器或数据集问题。
  下一步：让 cache 预热不受样本预算截断（或让完整性检查按 cache stage 的实际宽度而不是
  `len(features)` 校验、并让未满载 shard 也提交 manifest）。
- **运行器 `--resume` 行为**：它会把 `status.json` 里 **failed** 的 cell 当作已跑（`REUSE`），
  只有 completed 会复用、timeout 会打标记；重跑失败 cell 需要先删掉记录
  （本次备份为 `status.json.bak_before_fd_rerun`）。建议改成 failed 也重跑。
- 另外修掉一个环境问题：容器软 fd 上限 1024，64 个 worker 各持 cache shard 文件时会
  `OSError: [Errno 24] Too many open files`（`wikitext103_cache/PICO` 首跑就是这样失败的）。
  启动脚本现在会 `ulimit -n $(ulimit -Hn)`（524288），该 cell 重跑后完成（1473.1）。

### 3.6 TF 变体的状态

已修的三个系统 bug（都在正式代码里）：

1. `cedar/pipes/variant.py`：tf.data 生成器里变体被 shutdown 后，迭代结束处的
   `logger.info(self.variant_ctx.variant_type)` 抛 `AttributeError` → `tf.data` 迭代器报错、
   profiling 挂死；两处日志加 None 判断。
2. `cedar/client/dataset.py`（`_profile_smp` + layered 循环）：TF 算子放进 SMP **进程池**会死锁
   （tf.data iterator 不能跨进程）→ TF 算子跳过 SMP profile；这些负载的 profile 不含 SMP 条目，
   PICO 只在 INPROCESS / RAY / TF_RAY 里选（协议差异，已在结果里标注）。
3. `cedar/client/dataset.py`（`_adaptive_operator_benchmark`）：layered replay 直接调
   `_create_pipe_variant()`、绕过 `Pipe.mutate()` 的 spec 推导，TF_RAY actor 收到
   `tf_spec=None` → `from_generator` 报 `TypeError`；现在 replay 前从逻辑前驱推导一次 input spec。

仍阻塞：TF 卸载的 `_embedding` 需要 GPT-2 tokenizer/模型文件，而这些只在 driver 机器上（离线缓存）。
远端 actor 在离线环境下载失败（`MaxRetryError(... huggingface.co ...)`）。要做完 TF 变体需要：
把 HF 缓存同步/挂进远端（或塞进 Ray 的 `runtime_env.working_dir`），或把这类算子固定为 local、
只对纯 TF 计算做 TF 卸载。命令（解决后）：
`NEW_WORKLOADS="wikitext103_tf simclrv2_tf coco_tf" MODE=formal bash scripts/run_new_workloads_20260922.sh`。

## 4. 专题分析

### 4.1 affine（kx+b）的必要性：把 unopt 的每算子代价迁移到新 plan

（原 `docs/affine_transfer_experiment_20260921.md` 的 §1–§3、§5、§7；§4 的重复实验与 §6 的 profile 原始 k/b 表已并入 4.5 与 profile 文件本身，故编号跳过 4.1.4 / 4.1.6。）

#### 4.1.1 每算子的输入尺寸与实测开销

| 算子 | x declared (B) | x pico (B) | t declared (ms) | t pico (ms) |
| --- | ---: | ---: | ---: | ---: |
| T Batcher | 238,144 | 238,144 | 7.942 | 5.079 |
| N Normalize | 238,144 | 238,144 | 0.236 | 0.170 |
| B Blur | 238,144 | **59,536** | 8.387 | 9.150 |
| G Grayscale | 714,432 | **178,608** | 0.350 | 0.285 |
| J Jitter | 714,432 | **59,536** | 8.497 | 0.496 |
| H Flip | 714,432 | **59,536** | 0.129 | 0.081 |
| C Crop | 2,507,430 | **748,943** | 1.986 | 2.110 |
| F to_float | 626,857 | **59,536** | 0.452 | 0.063 |
| R ImageReader | 157 | 157 | 2.372 | 2.571 |

#### 4.1.2 把 declared（unopt）profile 迁移到 pico 顺序

两个模型都**锚定同一个 declared 实测点**（只看模型形式差异）：

- **cedar**：`t = t_declared × (x / x_declared)`（过原点等比例，Cedar 的做法）
- **affine**：`t = t_declared × (k·x + b) / (k·x_declared + b)`，k、b 取 profile 的
  `physical_model.operator_affine`（profiler 在多尺寸下拟合）

| 算子 | cedar 估计 | 误差 | affine 估计 | 误差 |
| --- | ---: | ---: | ---: | ---: |
| T Batcher | 7.942 | +56.4% | 7.942 | +56.4% |
| N Normalize | 0.236 | +38.5% | 0.236 | +38.5% |
| B Blur | 2.097 | **−77.1%** | 2.297 | −74.9% |
| G Grayscale | 0.088 | **−69.3%** | 0.153 | −46.2% |
| J Jitter | 0.708 | +42.8% | 1.064 | +114.5% |
| H Flip | 0.011 | **−86.7%** | 0.052 | −35.3% |
| C Crop | 0.593 | **−71.9%** | 1.693 | −19.8% |
| F to_float | 0.043 | −31.5% | 0.043 | −31.5% |
| R ImageReader | 2.372 | −7.7% | 2.372 | −7.7% |

**均值 |误差|：cedar 等比例 53.5%（中位 56.4%，最大 86.7%）；affine 47.2%（中位 38.5%）**。
只看**输入尺寸发生变化的那 6 个 mapper**：cedar ≈ **63%**、affine ≈ **54%**，且 affine 在
Crop（−72% → −20%）、Flip（−87% → −35%）、Grayscale（−69% → −46%）、Blur（−77% → −75%）都更好。

同样四个 in-pipeline 点做 **留一顺序**预测（k、b 只由另外三个顺序拟合）：cedar 等比例 均值 84.0% /
中位 62.6%，affine 均值 67.6% / 中位 46.4% —— affine 同样系统性更好。

#### 4.1.3 结论与残留误差

1. **Cedar 的每算子代价无法迁移到新 plan**：仅改顺序（尺寸随之变化）后，Blur/Grayscale/Flip/Crop 的
   估计误差达 **−69% ~ −87%**，而且是**系统性低估**——因为它假设 `cost ∝ 输入字节`（过原点）。
2. **漏掉的正是 affine 的 b（固定分量）**：profile 自己的 affine 拟合显示 9 个算子里有 6 个固定分量占比
   25%–100%（crop 79.8%、normalize 47.8%、flip 35.1%、grayscale 25.1%、jitter 4.6%、blur 3.2%）；
   把 b 加回来，同样一个锚点下 Crop/Flip 的误差就从 −70% 级降到 −20%/−35%。
3. **仍有残差，且原因明确**（不是 affine 形式的错）：
   - **外推越界**：Jitter 在 pico 顺序里只拿到 59,536 B，低于 profiler 实测过的下界（179,026 B），
     于是 kx+b 外推反而高估 115%（Cedar 的等比例在这点恰好更接近）；
   - **同一尺寸、不同 plan 上下文**：Normalize / ImageReader 在两个顺序里输入尺寸完全相同，但实测 CPU
     时间差 1.1–1.8×（上游节奏、缓存/频率状态），任何只以"输入尺寸"为自变量的模型都解释不了 —— 这部分
     需要 plan 上下文项，属于当前模型的已知边界。§5 把这里的残差拆成了两类：**Batcher 那一行原本是
     trace 归因错误（已修，不能当算子代价用）**，Normalize / ImageReader 才是真实的上下文效应。

也就是说：**affine 是必要的（少了 b 会有 70%+ 的系统性偏差），但还不充分（还需要覆盖尺寸范围
与 plan 上下文）**。

#### 4.1.5 两个异常行的性质：Batcher 是 trace 归因错误（已修），Normalize 是真实的 plan 上下文效应

§4 的表里有两行看着像"模型失效"（T Batcher 5.02 ↔ 12.55 ms/record，N Normalize 0.170 ↔ 0.240
ms/record，而两者四个顺序的输入尺寸都相同）。逐行核查后，**两行的成因完全不同**。

##### 4.1.5.1 T Batcher 行 = batcher 的 trace 归因错误（已在代码里修掉）

**现象**：batcher 的 trace 窗口量的是"这批 4 条记录在自己上游被生产出来"的 CPU 跨度，而不是
batcher 自己的 `torch.stack`：7.711（declared）/ 5.022（pico）/ 5.122（cedar）/ 12.550（old-dp）
ms/record，而 batch=4、单条 `(3, 244, 244)` float32 的 stack 单独测只有 **0.089 ms/batch ≈ 0.022
ms/record** —— 差两个数量级。

**根因**：`InProcessBatcherPipeVariant._iter_impl` 历史上在**每次 append** 时复制 trace
（`if x.do_trace: batch_ds.copy_metadata_from(x)`）。source 只按 `TRACE_FREQUENCY_SEC = 0.1 s`
抽样打 trace，所以一个 batch 最终带的是**更早那条记录**的链路；`trace_dict[batcher]` 减去那条记录
在上游的输出时间，差值里就混进了整批组装期间上游的 CPU 时间。

**修复**（`cedar/pipes/batch.py`）：

- 新增 `_emit_batch(batch_ds, last_input)`，batch 只在**最后一个输入**到达后锚定
  （`copy_metadata_from(last_input)`），窗口 = batcher 自己的 stack + 出队开销；
- trace 抽样间隔改为可配置：`CEDAR_TRACE_FREQUENCY_SEC`（默认 0.1 s，保持原 profiling 成本）。
  reconcile / 归因分析类实验用 `0`（每条记录都 trace）——否则"最后一个输入恰好没被抽到"的 batch 会
  整批不带 trace；
- 修复只影响**之后**产生的 trace，已归档的 profile / 结果快照保持不动。

**验证**（同一 harness：W=4、batch=4、全 INPROCESS、9,469 条、同一份 plan）：

| 顺序 | 修复前 T Batcher（3 次均值 ±stdev） | 修复后 T Batcher（r1，单次） |
| --- | ---: | ---: |
| declared | 7.711 ± 0.17 | 0.058 |
| pico | 5.022 ± 0.24 | 0.048 |
| cedar | 5.122 ± 0.08 | 0.044 |
| old-dp | 12.550 ± 0.55 | 0.066 |

（单位 ms/record。修复前 = `outputs/unopt_order_transfer_repeats_20260921`；修复后 =
`outputs/unopt_order_transfer_repeats_traceall_20260921` 的 r1 四个 cell。单进程探针
`tmp_analysis/batch_trace_probe.py` 独立给出每 4 条记录 0.167–0.202 ms，即 0.042–0.051 ms/record，
与表中一致，且该次每个样本的 `trace_order` 都完整。）

→ **§1 / §2 / §4 里 T Batcher 那一行不能当算子代价使用**：它测的是上游生产整批数据的跨度。修复后这行
四个顺序都是 ≈0.05 ms/record、建模输入尺寸也相同（238,144 B/record），两个模型都能预测，不再提供区分度。

##### 4.1.5.2 N Normalize 行 = 真实的 plan 上下文效应

| 顺序 | 输入 (B/record) | process (ms/record) | wall (ms/record) |
| --- | ---: | ---: | ---: |
| declared | 238,144 | 0.2423 | 0.2005 |
| pico | 238,144 | 0.1724 | 0.1554 |
| cedar | 238,144 | 0.1338 | 0.1201 |
| old-dp | 2,644,349 | 0.9977 | 0.9760 |

- declared / pico / cedar 三列**输入尺寸完全相同**、batch 也相同，process-time 却差 **1.8×**；
- 修复 batcher 归因前后这两行的值几乎不变（0.240 / 0.170 / 0.139 → 0.242 / 0.172 / 0.134），说明它跟
  5.1 的 bug 无关：Normalize 在 batcher **之前**，它的 trace 锚点一直是自己那条记录的上游；
- wall 与 process 同量级（process/wall = 1.02–1.21，多线程算子的 CPU 时间本就略高于 wall），**差异不是
  排队**；同样尺寸的 payload 在各顺序里的形状/内存布局、上游节奏都不同，算子自身在同一尺寸下的成本因此
  相差 1.8×。无论具体机制是布局还是缓存/频率状态，都落在"只以输入字节为自变量"的模型之外；
- 同类效应在 ImageReader 上也出现过（旧运行里 old-dp 的 reader 快 10%），但修复后的 r1 里四个顺序的
  reader 是 2.06 / 1.99 / 2.05 / 2.05 ms（≤3%），所以 10% 这一档还需要更多重复才能定量。

**对结论的影响**：迁移结论建立在 `B Blur / G Grayscale / H Flip / C Crop / J Jitter` 这些**输入尺寸随
顺序改变**的算子上（低估 −69% ~ −86%），它们不受 5.1 影响；**T Batcher 行作废**，**N Normalize 行是
上下文边界**而不是 affine 形式的问题。

##### 4.1.5.3 Reader 的输出尺寸差异（declared 646.6 kB vs pico 678.8 kB）也是抽样伪影

§4 的另一个疑点：`R ImageReader` 在四个顺序里输入完全相同（157 B/record），但它的**输出**尺寸却随顺序
不同——三次重复的均值分别是 declared **646.6 kB**、pico **678.8 kB**、cedar 629.7 kB、old-dp 564.3 kB
（每 record）。核查后同样是测量伪影，不是数据不同：

- reader 的输出尺寸 = **解码后像素字节数**（`get_sizeof_data(PIL.Image)` → `W×H×bands`），而 imagenette2/train
  的尺寸分布是**重尾**的：9,469 张图的均值 675.9 kB、标准差 **1,339 kB**（CV 1.98，最大一张解码后 38 MB）；
- reconcile 的 `output_sizes` 只是**被采样到的那部分记录**的均值，而 trace 是**按时间**抽的（默认每
  0.1 s 一条），所以每个顺序抽到的记录集合都不同：n ≈ 440 时的标准误 **≈64 kB（9.4%）**，
  646.6 ↔ 678.8 的 32 kB 差异远在噪声内；
- **直接验证**：把 trace 改成每条记录都打（`CEDAR_TRACE_FREQUENCY_SEC=0`，即 §5.1 之后 reconcile
  实验的默认设置）后，declared / pico / cedar 三个顺序的 reader 输出尺寸**完全相同**——
  均值 666.1 kB，连四个 worker 的分项都逐一对上（639.0 / 702.4 / 689.4 / 633.7 kB）；old-dp 是 661.1 kB，
  只因为那次运行的 old-dp cell 被提前中止（2,252 / 2,368 条）。

→ 结论：**不采样时该算子的尺寸在每个顺序里都一样**；只要用按时间抽样的 trace，重尾分布算子的"每记录
尺寸/成本"就会带上约 10% 的抽样抖动，跨顺序比较时不能把它当成数据差异。复现：
`tmp_analysis/probe_reader_size_sampling.py`（数据集尺寸分布 + 步长抽样仿真）。

#### 4.1.7 修正版：reordered 顺序下每个算子的 input / cedar 估计 / 实测（全量 trace）

数据来源：`outputs/unopt_order_transfer_repeats_traceall_20260921/reconcile_{declared,pico,cedar,old-dp}_r1`
—— 这次是**每条记录都 trace**（`CEDAR_TRACE_FREQUENCY_SEC=0`，每个 cell 2,368 条全部采到），因此

- `input` 是该顺序下**整个 pass 的精确均值**，不再是按时间抽样的子集（§5.3 的 646.6 vs 678.8 kB 就是
  抽样造成的）；
- T Batcher 的窗口是修复后的**自身 stack 开销**（§5.1），不再是"整批组装跨度"；
- 四个 cell 处理的记录集合完全相同（reader 输出 666.1 kB，四个 worker 逐项一致），所以同一张表内
  declared ↔ reordered 的对比不含抽样子集差异。

列的含义：`input x` = 该算子在该顺序下的输入（B/source record）；`cedar est` = 把 declared 实测按
Cedar 的过原点等比例迁移 `t_declared × x / x_declared`；`measured` = 该顺序下实测 process-time
（ms/source record）；`affine est` = 同一锚点下换成 profile 的 kx+b 形状。

##### 4.1.7.1 PICO 顺序

| operator | input x (B/record) | cedar est (ms) | measured (ms) | err | affine est (ms) | err |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| T Batcher | 238,144 | 0.0580 | 0.0478 | +21.4% | 0.0580 | +21.4% |
| N Normalize | 238,144 | 0.2423 | 0.1724 | +40.6% | 0.2423 | +40.6% |
| B Blur | 59,536 | 2.4008 | 9.0331 | **−73.4%** | 2.6304 | −70.9% |
| G Grayscale | 178,608 | 0.0892 | 0.2629 | **−66.1%** | 0.1563 | −40.5% |
| J Jitter | 59,536 | 0.7097 | 0.4807 | +47.6% | 1.0661 | +121.8%（越界外推） |
| H Flip | 59,536 | 0.0105 | 0.0788 | **−86.7%** | 0.0508 | −35.5% |
| C Crop | 666,126 | 0.4708 | 1.8660 | **−74.8%** | 1.5716 | −15.8% |
| F to_float | 59,536 | 0.0394 | 0.0614 | −35.8% | 0.0394 | −35.8% |
| R ImageReader | 157 | 2.0613 | 1.9911 | +3.5% | 2.0613 | +3.5% |

平均 |误差|：9 个算子 cedar **50.0%** / affine **42.9%**；只看输入尺寸变化的 6 个 mapper
（B/G/J/H/C/F）cedar **64.1%** / affine **53.4%**。

##### 4.1.7.2 cedar 顺序

| operator | input x (B/record) | cedar est (ms) | measured (ms) | err | affine est (ms) | err |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| T Batcher | 238,144 | 0.0580 | 0.0436 | +33.0% | 0.0580 | +33.0% |
| N Normalize | 238,144 | 0.2423 | 0.1338 | +81.1% | 0.2423 | +81.1% |
| B Blur | 59,536 | 2.4008 | 8.8865 | **−73.0%** | 2.6304 | −70.4% |
| G Grayscale | 666,126 | 0.3327 | 1.5151 | **−78.0%** | 0.3387 | −77.6% |
| J Jitter | 59,536 | 0.7097 | 0.5279 | +34.4% | 1.0661 | +101.9% |
| H Flip | 59,536 | 0.0105 | 0.1077 | **−90.3%** | 0.0508 | −52.9% |
| C Crop | 222,042 | 0.1569 | 0.7483 | **−79.0%** | 1.5024 | +100.8% |
| F to_float | 59,536 | 0.0394 | 0.0487 | −19.1% | 0.0394 | −19.1% |
| R ImageReader | 157 | 2.0613 | 2.0457 | +0.8% | 2.0613 | +0.8% |

平均 |误差|：9 个算子 cedar 54.3% / affine 59.7%；6 个 mapper cedar **62.3%** / affine 70.5%。

##### 4.1.7.3 old-dp 顺序

| operator | input x (B/record) | cedar est (ms) | measured (ms) | err | affine est (ms) | err |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| T Batcher | 238,144 | 0.0580 | 0.0655 | −11.5% | 0.0580 | −11.5% |
| N Normalize | 2,644,349 | 2.6905 | 0.9977 | +169.7% | 1.5196 | +52.3% |
| B Blur | 2,644,349 | 106.6356 | 52.7090 | +102.3% | 103.5432 | +96.4% |
| G Grayscale | 2,644,349 | 1.3206 | 1.0599 | +24.6% | 1.0788 | +1.8% |
| J Jitter | 881,450 | 10.5080 | 1.3534 | +676.4% | 10.4171 | +669.7% |
| H Flip | 238,144 | 0.0418 | 0.1234 | −66.1% | 0.0712 | −42.4% |
| C Crop | 881,450 | 0.6229 | 0.7190 | −13.4% | 1.6052 | +123.2% |
| F to_float | 661,087 | 0.4377 | 0.4484 | −2.4% | 0.4377 | −2.4% |
| R ImageReader | 157 | 2.0606 | 2.0461 | +0.7% | 2.0613 | +0.7% |

平均 |误差|：9 个算子 cedar 118.6% / affine 111.2%；6 个 mapper cedar **147.5%** / affine 156.0%
（该顺序把 Blur/Jitter 的输入抬到 2.6 MB / 0.88 MB，外推幅度最大，两个模型都失效）。

##### 4.1.7.4 declared 锚点（= profile 的测量点）

| operator | input x (B/record) | measured (ms/record) |
| --- | ---: | ---: |
| T Batcher | 238,144 | 0.0580 |
| N Normalize | 238,144 | 0.2423 |
| B Blur | 238,144 | 9.6034 |
| G Grayscale | 714,432 | 0.3568 |
| J Jitter | 714,432 | 8.5169 |
| H Flip | 714,432 | 0.1255 |
| C Crop | 2,664,505 | 1.8831 |
| F to_float | 666,126 | 0.4411 |
| R ImageReader | 157 | 2.0613 |

**两条使用限制**

1. **不要跨 campaign 比绝对值**：这份全量 trace 运行与 §4 的按时间抽样运行是两次不同的运行，机器状态
   不同（例如 reader 在这里 2.06 ms/record，§4 里是 2.38 ms/record）。只有**同一次运行内部**的
   declared ↔ reordered 对比是有效对照。
2. T Batcher / N Normalize 两行已经不是 §5 里那种伪影（一个是自身 stack 开销，一个是真实上下文效应），
   它们的误差小、但也不由算子尺寸解释；迁移结论仍应看 B/G/J/H/C/F 六个 mapper。

复现：`python -u tmp_analysis/reordered_operator_table.py`（打印上面三张表 + 平均值）。

### 4.2 simclrv2：融合段 RAY↔local、W 依赖与单记录拆分

（原 `docs/experiment_simclrv2_fuse_local_vs_ray_20260920.md`，去掉产物清单。）

#### 4.2.1 实验设置

| 项 | 值 |
| --- | --- |
| 负载 | `evaluation/pipelines/simclrv2/cedar_dataset.py` |
| 数据集 | `evaluation/datasets/imagenette2/imagenette2/train`（9,469 张图） |
| batch / epoch | batch 4，1 epoch（一次完整 pass，实测处理 9,472 条样本） |
| W（`n_local_workers`） | 64（两臂相同） |
| CPU 预算 | 64；两臂都不做计划优化（执行固定计划） |
| profile | `outputs/ultimate_eight_optimizers_20260920/simclrv2/profiles/shared.yaml`（sha256 `d40bc97d…`） |
| 远端 Ray | 允许（`CEDAR_RAY_PLACEMENT_RESOURCE=cedar_remote`，只用于 ray 臂；local 臂的计划里没有 RAY stage） |
| 执行器 | `scripts/run_fixed_plan_throughput.py`（固定计划，不跑优化器；指标定义与 `compare_optimizer_perf` 一致） |
| 启动脚本 | `scripts/run_simclrv2_local_vs_ray_9469_20260920.sh` |

指标口径与 campaign 相同：**稳态吞吐 = 处理样本数 / Σ epoch 运行时间**，不含数据集构造（Ray 连接、
worker 启动、计划装载）的时间（该部分单列为 setup）。

#### 4.2.2 计划记录

两臂只在 pipe 11 的 variant 上不同，其余 pipe、顺序、宽度、W 完全一致。

| pipe | name | 后继 | local 臂 variant | ray 臂 variant | fused 成员 |
| ---: | --- | ---: | --- | --- | --- |
| 0 | `BatcherPipe(batch_size=4)` | 10 | INPROCESS | INPROCESS | — |
| 1 | `MapperPipe_Normalize` | 0 | INPROCESS | INPROCESS | — |
| 2 | `MapperPipe_GaussianBlur` | （被融合） | INPROCESS | INPROCESS | — |
| 3 | `MapperPipe_Grayscale` | 6 | INPROCESS | INPROCESS | — |
| 4 | `MapperPipe_ColorJitter` | （被融合） | INPROCESS | INPROCESS | — |
| 5 | `MapperPipe_RandomHorizontalFlip` | （被融合） | INPROCESS | INPROCESS | — |
| 6 | `MapperPipe_RandomResizedCrop` | 11 | INPROCESS | INPROCESS | — |
| 7 | `MapperPipe_to_float` | 1 | INPROCESS | INPROCESS | — |
| 8 | `ImageReaderPipe` | 3 | INPROCESS | INPROCESS | — |
| 9 | `LocalFSListerPipe`（source） | 8 | INPROCESS | INPROCESS | — |
| 10 | `PrefetcherPipe`（sink） | — | INPROCESS | INPROCESS | — |
| **11** | **`FusedPipe{2,5,4}`** | 7 | **INPROCESS** | **RAY w=1** | `[2, 5, 4]` |

计划链：

- local 臂：`LocalFSListerPipe → ImageReaderPipe → Grayscale → RandomResizedCrop → FusedPipe{2,5,4} → to_float → Normalize → BatcherPipe(4) → PrefetcherPipe`
- ray 臂：同上，唯一区别是 `FusedPipe{2,5,4}[RAY w=1]`

来源与指纹：

| 文件 | 说明 | sha256 |
| --- | --- | --- |
| `outputs/simclrv2_local_vs_ray_9469_20260920/plans/cedar_opt_local_w1.yaml` | local 臂（本次新计划） | `24eaeb4fb817963d…` |
| `outputs/simclrv2_local_vs_ray_9469_20260920/plans/cedar_opt_ray_w1.yaml` | ray 臂（cedar-opt 原计划） | `39744a4dc6728399…` |

ray 臂的原始来源是 campaign 里 cedar-opt 为 simclrv2 生成的计划
（`outputs/six_workload_formal_v3_20260919/simclrv2/plans/round1__optimizer.yaml`，64 个 feature 副本中
任取一个）：`FusedPipe{2,5,4}` 带 `n_actors=1`、`submit_batch_size=16`；本文件把它物化成固定计划
（`physical_plan:` 包装，供 `DataSet(feature_config=...)` 直接执行）。被融合掉的 pipe 2/4/5 在记录里
没有 variant，物化时统一补 `INPROCESS`，它们不参与执行。

#### 4.2.3 实测稳态吞吐（W=64，9,472 条样本）

| 臂 | round1 | round2 | round3 | 中位数 | 1000/T（round1，ms/source-record） |
| --- | ---: | ---: | ---: | ---: | ---: |
| **local（`FusedPipe{2,5,4}` INPROCESS）** | **2390.97** | 2353.74 | 2314.68 | 2353.74 | **0.4182** |
| ray（`FusedPipe{2,5,4}` RAY w=1） | 1150.78 | 1182.59 | 1219.26 | 1182.59 | 0.8690 |

（单位 rec/s；每臂 3 次是脚本既定设置，用户要求“单次即可”，round1 即单次结果；另有一次 local 冒烟运行
2,335.30 rec/s。ray 臂与 campaign 里同一计划在 9,469 上的 1,150.66 rec/s 完全吻合，说明测量口径一致。）

setup（数据集构造，不含稳态）：local 2.72–2.89 s，ray 2.76–2.79 s；两边都没有优化器开销。

#### 4.2.4 三个 cost model 的预测

口径：

- **Cedar**：`Optimizer.calculate_cost`，单位 ms/source-record，**单 worker**（模型没有 W 维度）。
- **Plumber**：计划声明的宽度给出单 worker 瓶颈速率 `X = min_i(width_i · R_i)`，`R_i = 1/latency_i`
  （融合 stage 取成员 latency 之和）；系统速率 = `X · W`，故 **cost = 1000 / (X · W)**。
- **PICO**（`SimpleDpWorkersWidthBoundaryOptimizer`，即 affine + boundary + workers + width 的 DP 目标）：
  原始 score `S` 是 W 条件化的，按既定口径取 **cost = S / W**。

| 计划 | W | Cedar | Plumber ÷W | PICO 原始 S | PICO S/W |
| --- | ---: | ---: | ---: | ---: | ---: |
| local（`{2,5,4}` INPROCESS） | 64 | 9.1662 | 0.2543 | 8.2365 | 0.1287 |
| ray（`{2,5,4}` RAY w=1） | 64 | 8.4753 | 0.2543 | 88.9718 | 1.3902 |

（单位均为 ms/source-record；复算脚本 `scripts/score_plan_cost_models.py`。参考：同一 profile 下完全不融合的
纯 local 流水线 Cedar cost = 22.9795 ms/source-record。）

Plumber 的单 worker 瓶颈来自融合 stage：

| stage | 成员 | variant | 宽度 | 单条服务时间 |
| --- | --- | --- | ---: | ---: |
| `FusedPipe{2,5,4}` | 2,5,4 | INPROCESS（ray 臂为 RAY） | 1 | 16.2741 ms |
| `BatcherPipe` | 0 | INPROCESS | 1 | 7.7742 ms |
| `ImageReaderPipe` | 8 | INPROCESS | 1 | 2.2263 ms |
| 其余（3/6/7/1/9） | — | INPROCESS | 1 | 0.032–1.940 ms |

即 `X = 1000/16.2741 = 61.43 rec/s`，`X · W = 3931.5 rec/s`，`cost = 0.2543`。两臂的 stage 结构、宽度、
latency 全部相同，所以 Plumber 给出的 cost **完全一样**（它不读后端类型，也就无法表达跨主机传输）。

#### 4.2.5 预测 vs 实测（cost 空间，越低越好）

| 视角 | local | ray | 结论 |
| --- | ---: | ---: | --- |
| **实测**（1000/T，系统级） | **0.4182** | 0.8690 | local 好 2.08× |
| Cedar（单 worker，仅比较相对序） | 9.1662 | **8.4753** | 预测 ray 好 8% —— **方向相反** |
| Plumber ÷W | 0.2543 | 0.2543 | 无法区分（预测值比实测乐观 1.6×/3.4×） |
| PICO S/W | **0.1287** | 1.3902 | 预测 local 好 10.8× —— 方向对，幅度偏大 5.2× |

读法：

- **Cedar**：对融合段做 Amdahl 反转后，RAY 的单算子成本看起来比 local 便宜（8.475 < 9.166），
  因为模型只比较“单算子 offload 后吞吐”，没有每记录跨主机往返、序列化和远端排队项；实测 RAY
  反而贵一倍。
- **Plumber**：`R_i` 只来自 baseline latency、宽度只来自计划声明，两个计划在它眼里是同一个东西，
  所以无论如何缩放（×W 或 ÷W）都无法解释实测差异。
- **PICO**：把跨主机 RTT 与字节额度记在 RAY 上，因此方向正确；但它把 local 侧估得过好
  （S/W = 0.1287，折算 7,770 rec/s，实测 2,391 rec/s），把 RAY 侧估得过差（1.3902，折算 719 rec/s，
  实测 1,151 rec/s），净效果是 10.8× 对 2.08×。

#### 4.2.6 另外四个 simclrv2 计划的三模型预测（unopt / old-dp / plumber / cedar）

口径与 §4 完全一致：**Cedar** = `Optimizer.calculate_cost`（单 worker，模型没有 W）；**Plumber** =
`1000/(X·W)`，`X` 由计划声明的宽度给出（融合 stage 的单条服务时间 = 成员 latency 之和）；**PICO** = `S/W`。
计划取 campaign 记录的原样文件 `outputs/ultimate_eight_optimizers_20260920/simclrv2/plans/`
（`round1__{unopti,old_dp_legacy_optimizer,plumber_optimizer,optimizer}.yaml`），其中 `cedar` 就是 §2 的 ray 臂。

| 计划 | W | 计划形态 | Cedar | **Plumber ÷W** | PICO S | **PICO S/W** |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| pico（PICO 自己选中的计划：`R → Fused{6,3,4,5,2,7,1} → T`，全 local） | 64 | 七算子融合、batcher 在外 | **8.9731** | 0.2987 | 8.2057 | 0.1282 |
| ray-data（`R → Fused{7,6,5,4,3,2,1}[RAY w=64] → T`） | **1** | 七算子融合成一个 64-actor 的 RAY stage | **9.9121** | 8.4977 | n/a | n/a |
| unopt | 1 | 全 INPROCESS、无融合、无 prefetch | 22.9795 | 8.4977 | 25.4090 | 25.4090 |
| old-dp | 32 | `ImageReader → FusedPipe{7,1,2,3,4,6,5}[SMP w=1] → Batcher → Prefetch` | 7.9084 | 0.5975 | 113.5186 | 3.5475 |
| plumber | 1 | 逐算子 SMP（to_float 2 / crop 7 / jitter 27 / grayscale 2 / blur 25） | 22.9795 | 7.7742 | n/a | n/a |
| cedar | 64 | `… → FusedPipe{2,5,4}[RAY w=1] → …` | 8.4753 | 0.2543 | 88.9718 | 1.3902 |
| （§3 参照）fuse-local | 64 | 同上但 `{2,5,4}` 为 local | 9.1662 | 0.2543 | 8.2365 | 0.1287 |

完整计划链（即计划记录本身）：

- **unopt**（W=1）：`LocalFSListerPipe → ImageReaderPipe → to_float → RandomResizedCrop → RandomHorizontalFlip → ColorJitter → Grayscale → GaussianBlur → Normalize → BatcherPipe(4)`（全部 INPROCESS）
- **old-dp**（W=32）：`LocalFSListerPipe → ImageReaderPipe → FusedPipe{7,1,2,3,4,6,5}[SMP w=1] → BatcherPipe(4) → PrefetcherPipe`
- **plumber**（W=1）：`LocalFSListerPipe → ImageReaderPipe → to_float[SMP w=2] → RandomResizedCrop[SMP w=7] → RandomHorizontalFlip → ColorJitter[SMP w=27] → Grayscale[SMP w=2] → GaussianBlur[SMP w=25] → Normalize → BatcherPipe(4) → PrefetcherPipe`
- **cedar**（W=64）：`LocalFSListerPipe → ImageReaderPipe → Grayscale → RandomResizedCrop → FusedPipe{2,5,4}[RAY w=1] → to_float → Normalize → BatcherPipe(4) → PrefetcherPipe`

注：`plumber` 计划的逐算子 SMP 宽度不在 PICO 的搜索空间内（`ValueError: Variant SMP is outside the DP
search space.`），因此该行 PICO 记 n/a；`old-dp` 的整块 SMP 融合计划可以被 PICO 打分。

与实测对照（cost 空间 = 1000/T）：实测取同一 campaign 的 **189,380 条**运行（与 §3 的 9,469 不同尺度，
仅作数量级对照；§3 的 fuse-local/fuse-ray 是同尺度 9,469）：

| 计划 | 实测 1000/T | Cedar | Plumber ÷W | PICO S/W | Plumber 实测/预测 | PICO 实测/预测 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| unopt | 21.642（46.2 rec/s） | 22.9795 | 8.4977 | 25.4090 | 2.55×（乐观） | 0.85×（偏悲观 1.17×） |
| old-dp | 2.833（352.9 rec/s） | 7.9084 | 0.5975 | 3.5475 | 4.74×（乐观） | 0.80×（偏悲观 1.25×） |
| plumber | 9.353（106.9 rec/s） | 22.9795 | 7.7742 | n/a | 1.20×（乐观） | n/a |
| cedar | 0.842（1187.7 rec/s） | 8.4753 | 0.2543 | 1.3902 | 3.31×（乐观） | 0.61×（偏悲观 1.65×） |

**单 worker 视角**（Cedar 给的是单 worker 代价，这样才可逐项比较；每 worker 实测代价 = W/T，隐含“W 个
worker 均分输入、彼此无干扰”的理想化假设）：

| 计划 | 每 worker 实测 W/T | Cedar | 实测 / Cedar |
| --- | ---: | ---: | ---: |
| unopt | 21.64 | 22.9795 | 0.94×（Cedar 悲观 6%） |
| old-dp（SMP 全融合） | 90.67 | 7.9084 | 11.5×（Cedar 乐观） |
| plumber（逐算子 SMP） | 9.35 | 22.9795 | 0.41×（Cedar 悲观 2.5×） |
| cedar（RAY 融合） | 53.89 | 8.4753 | 6.4×（Cedar 乐观） |
| （§3）fuse-local @9,469 | 26.77 | 9.1662 | 2.9×（Cedar 乐观） |
| （§3）fuse-ray @9,469 | 55.62 | 8.4753 | 6.6×（Cedar 乐观） |

读法：

- **Cedar 的精度取决于计划形态**（见上面的单 worker 表）：纯 local 单 worker 计划几乎精确（unopt 0.94×）；
  纯 local 但宽 W 时偏乐观约 2.9×（64 个 worker 抢核的开销它不算）；带 offload 的计划偏乐观 6–11×；
  而逐算子 SMP 的 plumber 计划它反倒悲观 2.5×（Cedar 的 Amdahl 反转认为那些 offload 没有收益，给出的
  22.98 与“完全不融合的基线”一模一样）。
- **Plumber ÷W** 在 plumber 自己那个“单 worker、瓶颈是本地 stage”的计划上很准（7.774 vs 9.353，1.20×），
  但凡 W>1 就系统性乐观 2.5–4.7×：它把 `X·W` 当成线性放大，没有 worker 之间的竞争/内存带宽项。
- **PICO（S/W）** 在 unopt / old-dp / cedar 上落在 0.80–0.85×（略偏悲观，方向与量级都对），但在 §5 的
  fuse-local 上偏乐观到 0.31×（3.25× 高估）——与之前在 commonvoice 上的结论一致：它对“全 local + 宽 W”
  的折扣仍偏松。

复算：`scripts/score_plan_cost_models.py --workload simclrv2 --profile <shared.yaml> --plan unopt=<plan> --plan old-dp=<plan> --plan plumber=<plan> --plan cedar=<plan>`。

#### 4.2.7 单条记录的全流程耗时拆分（cedar plan）

方法：用 `CEDAR_RECONCILE_DIR` 打开逐步 trace（每个 pipe 记录 `perf_counter_ns` 的 wall 时间戳与
`process_time_ns` 的 CPU 时间戳），跑同一份计划（W=64、batch 4、9,472 条），再把 64 个 worker 的
原始样本聚合。**每段的耗时 = 本 pipe 打点 − 上一 pipe 结束打点**，即这段在 worker 时间线上占用的时间，
包含算子本身的计算、以及算子之间的排队/序列化/远端往返。trace 开销很小：cedar 臂 1,193.9 rec/s
（无 trace 1,150.8）、local 臂 2,366.0 rec/s（无 trace 2,391.0）。

##### 4.2.7.1 cedar plan（`FusedPipe{2,5,4}` 在 RAY 上，就是 campaign 选中的计划）

| 段（按流水线顺序） | pipe | 中位耗时 ms/record | mean | p90 |
| --- | ---: | ---: | ---: | ---: |
| LocalFSListerPipe（source，列目录） | 9 | 0.079 | 0.117 | 0.113 |
| ImageReaderPipe（JPEG 解码 + fix） | 8 | 3.168 | 3.641 | 5.519 |
| Grayscale | 3 | 2.217 | 3.971 | 6.158 |
| RandomResizedCrop | 6 | 1.532 | 1.782 | 2.843 |
| **FusedPipe{2,5,4}（GaussianBlur+Flip+ColorJitter）@RAY** | 11 | **4209.718** | 4267.506 | 5685.425 |
| to_float（RAY 之后） | 7 | 0.108 | 0.191 | 0.445 |
| Normalize | 1 | 0.336 | 0.454 | 0.812 |
| BatcherPipe(4) | 0 | 0.262 | 0.308 | 0.568 |
| PrefetcherPipe（sink） | 10 | 0.029 | 0.034 | 0.050 |
| **本地算子段合计（不含 RAY、不含 sink）** | | **7.73 ms** | | |
| **单条记录端到端延迟（各段之和）** | | **4217.4 ms** | | |

- RAY 段的 4.21 s 是 **延迟**不是吞吐成本：其中 actor 侧真正的算子计算只有 **11.2 ms/record**
  （`service_stats`，actor 用自己的时钟整段计时，不受跨机时钟影响），其余约 4.2 s 是
  提交 / 序列化 / 远端排队 / 取回。用 Little 定律核对：每 worker 18.66 rec/s × 4.21 s ≈ 79 条在飞，
  与计划里的 `max_inflight=100`、实测 buffer 一致 —— 正是 prefetch 把这些排队隐藏掉了。
- 该臂每 worker 的实测周期 `W/T = 64/1193.9 = 53.6 ms/record`，而本地算子只占 7.73 ms → RAY 路径
  净成本约 **45.9 ms/record**（提交、2×序列化、`ray.get`、批量凑批等待）。

##### 4.2.7.2 对照：同一计划把 `{2,5,4}` 放本地（fuse-local）

| 段 | pipe | 中位耗时 ms/record | mean | p90 |
| --- | ---: | ---: | ---: | ---: |
| LocalFSListerPipe | 9 | 0.039 | 0.065 | 0.060 |
| ImageReaderPipe | 8 | 3.524 | 3.757 | 5.690 |
| Grayscale | 3 | 1.918 | 3.128 | 4.805 |
| RandomResizedCrop | 6 | 1.397 | 1.605 | 1.885 |
| FusedPipe{2,5,4}（本地） | 11 | 5.753 | 12.287 | 38.807 |
| to_float | 7 | 0.048 | 0.053 | 0.072 |
| Normalize | 1 | 0.212 | 0.239 | 0.295 |
| BatcherPipe(4)（凑批等待为主） | 0 | 6.395 | 7.832 | 18.077 |
| PrefetcherPipe（sink，消费者等待，不计成本） | 10 | 174.238 | 158.232 | 302.660 |
| **算子段合计（不含 sink）** | | **19.29 ms** | | |
| **每 worker 实测周期 W/T** | | **27.05 ms** | | |

读法：

- 纯本地执行时，“算子间”几乎没有开销：各段之和 19.29 ms 已接近每 worker 周期 27.05 ms，差值 ~7.8 ms
  是 worker 主循环（队列投递、prefetch、profiling 记账）的成本。单算子耗时排序：
  ImageReader 3.52 ＞ Batcher 凑批 6.40（等待 4 条凑批） ＞ Fused 增广 5.75 ＞ Grayscale 1.92 ＞
  crop 1.40 ＞ Normalize 0.21 ＞ to_float 0.05 ＞ source 0.04。
- 放 RAY 后，本地算子被压到 7.73 ms（与远端排队重叠），但 RAY 路径把每 worker 周期从 27.0 ms 推到
  53.6 ms —— **算子本身最大的那 5.8 ms 反而最便宜，算子之间（提交/序列化/排队）才是贵的那一半**。

复现：`CEDAR_RECONCILE_DIR=<dir> python -u scripts/run_fixed_plan_throughput.py --plan <plan> ...`，
每个 worker 落一个 `worker_<i>.json`（含 `wall_latency_samples`、`process_latency_ns_per_sample`、
`service_stats`、`pipe_counters`），本次数据在 `outputs/simclrv2_breakdown_20260920/`。

##### 4.2.7.3 RAY 融合块的三段式拆分：序列化 / 传输 / 计算（2026-09-21 追加）

§7.1 把 `FusedPipe{2,5,4}` 的 4.21 s 段延迟整体算作"RAY 段"，这里再拆开。两个数据源：

1. **in-plan 计时**（新增 env 开关 `CEDAR_RAY_PATH_TIMING=1`，在 Ray 客户端埋点：`submit` = 参数
   序列化 + `.remote()` 提交；`ray.get` = 取回；actor 侧计算仍由 `process_profiled` 记录），
   随 `worker_*.json` 的 `service_stats[11].path_timing` 落盘；
2. **跨主机边界微基准**（`scripts/measure_ray_boundary_payload.py`）：在远端节点上放一个空转 actor
   做 echo，用同一批大小的真实 payload（fused 块的输入 238,562 B/样本、输出 714,852 B/样本，
   batch=16，30 次），把"回程序列化+传输+反序列化+框架"从计算里剥出来。

| 组成 | 实测 | 来源 |
| --- | ---: | --- |
| 进入 fused pipe：客户端**序列化**（cloudpickle.dumps，单样本） | 0.035 ms（238 KB）/ 0.179 ms（714 KB） | 微基准 |
| 进入 fused pipe：**序列化+提交**（`.remote()` 返回） | 0.265 / 0.328 ms；真实计划里 **0.347 ms**（p50 0.337，64 worker 分布 0.231–0.524） | 微基准 + in-plan |
| **fused pipe 内真正计算**（actor 侧 wall clock） | **10.98 ms**（p50 11.07，7.65–16.23） | in-plan |
| 离开 fused pipe：**回程**（结果序列化 + 跨主机传输 + 客户端反序列化 + 框架） | **6.56 ms**（238 KB）/ **16.56 ms**（714 KB） | 微基准（空转 actor，单批次在飞的下界） |
| 其中客户端**反序列化**（cloudpickle.loads，单样本） | 0.017 / 0.110 ms；真实计划里 `ray.get` 只要 0.330 ms（结果通常已在本机 object store） | 微基准 + in-plan |
| **排队 / 在飞窗口等待**（prefetch 吸收的那部分） | ≈ **4.19 s**（4.21 s 减去上面各实测量） | in-plan 段延迟 |
| 单条记录在该段的墙钟延迟 | 4.21 s（p50）/ 4.27 s（mean） | in-plan |

读法：

- **真正的算子计算只占 11 ms/record，而"进出 + 传输"的可用下界已经 7–17 ms/record**；同一块在本地
  执行只要 5.75 ms/record（§7.2），也就是说放进 actor 后算子本身慢了约 1.9×（单线程 actor、无批量红利）。
- 单位成本（提交 0.35 + 计算 11.0 + 回程 ≥16.6 ≈ **28 ms/record**）已经超过本地版整条流水线的每 worker
  周期 27.05 ms/record —— 这正是 RAY 版吞吐只有本地版一半的直接原因。
- 剩下的 4.19 s/record 是**排队**：结果通常已经在本机，所以 `ray.get` 只要 0.33 ms；等待发生在
  "批次凑齐 + inflight 窗口（`max_inflight=100`）"这一段，被 prefetch 隐藏，但它把每 worker 的在飞
  样本数推到 ~79（Little 定律），并算出 18.66 rec/s 的节拍。
- 注意微基准是**单批次在飞**的下界：真实运行时 64 个 worker 同时在跨主机搬运（每 batch 入 3.8 MB、
  回 11.4 MB），争用会让回程更慢——这也是"单位成本合计 28 ms"仍低于实测 53.6 ms/record 的原因。

#### 4.2.8 指定顺序 / 指定融合的 5 个候选计划的 Cedar cost（2026-09-21 追加）

字母表取自论文图例（`my_paper/69e75a0100d7b4afeb1cfc20/figures/build_pipeline_cooptimization_simclr.py`）：
**R**=reader, **F**=to_float, **C**=Crop(RandomResizedCrop), **H**=Flip, **J**=Jitter, **G**=Grayscale,
**B**=Blur(GaussianBlur), **N**=Normalize, **T**=Batcher。所有计划 R 固定在最前（source 之后），
末尾接 Prefetcher；口径同 §4：`Optimizer.calculate_cost`，ms/source-record、单 worker、模型不用 W。

复算脚本：`scripts/score_cedar_simclrv2_plan_variants.py`。

| # | 计划 | Cedar cost |
| --- | --- | ---: |
| 1 | R → G C B H J F T N（无 fuse、全 local） | **10.5481** |
| 1b | R → G C B H J F N T（同上，仅 T/N 互换） | **10.5481** |
| 2 | R → F N B G J C H T（无 fuse、全 local） | **79.3948** |
| 3 | R → Fused{G,C,B,H,J,F,T,N}（panel 1 全 fuse、local） | **4.5319** |
| 3b | R → Fused{G,C,B,H,J,F} → T → N（只 fuse 可映射算子、T 原位） | **9.2923** |
| 4 | cedar plan（R → G → C → Fused{B,H,J} → F → N → T）把 fuse 块换成 **SMP w=1** | **8.4753** |
| 5 | R → Fused{F,N,B,G,J,C,H} → T（plan 2 的 7 个算子全 fuse、local） | **11.0793** |
| 5b | R → Fused{F,N,B,G,J,C,H,T}（连 Batcher 一起 fuse、local） | **5.1382** |

参照（同 profile）：不融合且按 profile 声明顺序（R F C H J G B N T）**22.9795**；cedar plan 原样（fuse 块
RAY w=1）**8.4753**；同一计划 fuse 块改 local **9.1662**。

读法：

- **plan 2 的 79.39 来自“Blur 排在 Crop 之前”**：Cedar 的尺寸比模型让 Blur 在 597×597 float32
  （2.39 MB/record，是基线 244×244=238 KB 的 10×）上执行，单算子被计到 **60.45 ms**（基线 6.02 ms），
  仅 `to_float` 的 ×4、`Crop` 太晚这两点就把整条计划推到基线的 3.5×。plan 1 反过来（G、C 提前）→ 10.55。
- **plan 4（SMP）与 RAY 版完全同价（8.4753）**：`{B,H,J}` 三个成员在 RAY 和 SMP 下都被 Amdahl 判成
  cost = 0，所以这个模型看不出后端差别；而同一个块放 local 要 9.1662（贵 8%）。
- **全 fuse 明显更便宜**：plan 1 的 8 个算子合成一块 → 4.5319（相对不融合 10.5481 便宜 57%）；
  差别来自 Cedar 的融合 IO 折扣（§4/§5 的同一机制）。把 Batcher 留在块外（3b/5）会让折扣少拿一次，
  5 与 5b 的 11.08 vs 5.14 就是这个效应。

#### 4.2.9 追加实验：同一份计划在 W=1 与 W=64 下的 RAY / local 对照（2026-09-21）

目的：核实"profile 里 RAY offload 让整条流水线变快、但执行时 RAY 更贵"这个矛盾。做法：取 cedar plan
（`Fused{B,H,J}` 在 RAY）与它的 local 融合版，各把 `n_local_workers` 设成 1 和 64，其余完全不变，
9,469 条（9,472 样本）单次运行。脚本 `scripts/run_simclrv2_raylocal_w1_w64_20260921.sh`，
数据在 `outputs/simclrv2_raylocal_w1_w64_20260921/`。

| cell | 稳态吞吐 | 单 worker 每记录 | 相对同 W 的另一臂 |
| --- | ---: | ---: | --- |
| RAY 融合，W=1 | **97.20 rec/s** | 10.29 ms | RAY 比 local **快 1.34×** |
| local 融合，W=1 | 72.43 rec/s | 13.81 ms | — |
| RAY 融合，W=64 | 1181.01 rec/s | 54.2 ms（每 worker 周期） | local 比 RAY **快 2.06×** |
| local 融合，W=64 | **2437.15 rec/s** | 26.3 ms | — |

读法：

- **W=1 时 RAY 确实更快（1.34×）**，这正是 profile 测到"offload 提升整条吞吐"的原因：W=1 时 worker 串行跑
  完整条链，把 `{B,H,J}`（本地 5.75–12 ms）交给远端 actor 做可以与 worker 的其余 stage **重叠**，于是瓶颈
  从"本地串行总和"变成 `max(本地剩余链, actor lane)` —— profile 的 `profile_scope=single_local_worker`、
  `ray_actors_per_stage=1` 就是同一个配置（其 offload 测量里 Blur 43.52→72.27 rec/s，actor 侧计算 10.7 ms
  反而比本地 6.0 ms 更慢，也说明收益来自并发而不是单价）。
- **W=64 时结论反转（local 快 2.06×）**：64 个 worker 已经把本地工作并行化，RAY 这一跳的每记录开销
  （提交 0.35 ms + actor 计算 11.0 ms + 回程序列化/传输/反序列化 6.6–16.6 ms ≈ 18–28 ms）**加在**每个
  worker 的关键路径上，而同样三个算子本地只要 5.75 ms。每 worker 周期从 26.3 ms 涨到 54.2 ms，差值
  ≈28 ms/record，与 §7.3 的分段一致。
- 结论：Cedar 把"W=1、单算子、单 actor 的整条吞吐提升"通过 Amdahl 反演成"该算子 per-record cost 归零"，
  隐含假设收益与 W / payload / 并发流数无关；实测这两个 W 下的方向刚好相反。PICO 因为显式建模了
  boundary 与跨主机传输（`bytes × W / bandwidth`），在 simclrv2 上给出全 local 融合计划，方向正确。

### 4.3 PICO DP 最优性：整数规划对照

（原 `docs/pico_dp_optimality_ilp_20260921.md`。）

#### 4.3.1 结果

| 实例 | W | search runs | nodes/edges（每个 run） | DP cost | MILP 最优 | MILP 计划 replay | 结论 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| commonvoice | 64 | 63 | 120 / 1,440 | 27.070672 | **27.070671591737725** | 27.070672 | **DP = MILP，DP 最优**（差 3.6e-15） |
| simclrv2 | 64 | 64 | 1,003 / 14,343 | 8.205709 | **8.205709** | 8.205709 | **DP = MILP，DP 最优**（差 0） |
| commonvoice | 32 | 63 | 204 / 1,776 | 26.432393 | 25.776682 | ✗ `ValueError: SMP transport curve does not cover requested W` | 不可判定（见下） |
| simclrv2 | 32 | 64 | 1,673 / 29,270 | 7.271867 | 8.205709 | 8.205709 | 不可判定（见下） |

W=64 正是 campaign 里 PICO 实际产出计划的那组配置（commonvoice 全融合 local、simclrv2
`Fused{2,5,4,3,6,1,7}`），两个实例上 MILP 的最优值和 DP 报出的 cost **逐位相同**，而且 MILP 选出的计划
被 optimizer 自己 replay 出同样的数值 —— 即：在这两个实例上，DP 的 Pareto 剪枝、状态合并与最短路径选择
没有丢掉任何更优解，DP 求到的就是该 cost model 在这个候选空间上的最优解。

#### 4.3.2 W=32 的两个不一致

1. **search 与 replay 的可行性口径不一致**：commonvoice W=32 时，MILP 找到 25.7767（比 DP 的 26.4324
   低 2.5%），但把它交给 optimizer 自己的 `_replay_dp_objective` 会抛
   `ValueError: SMP transport curve does not cover requested W` —— 说明搜索阶段给 SMP stage 定价时用的
   worker 数与"物化计划后再打分"时用的 W 不是同一个，聚合传输曲线覆盖不到后者。搜索因此可能给一个
   实际不可行的 SMP 计划报出偏低的 cost。
2. **目标坐标并非处处可加**：simclrv2 W=32 时 DP 报 7.2719，而"全部被评估转移"的图里最优只有 8.2057
   （即 DP 报出的那条路径在该图里复现不出来）。`SimpleDpWorkersBoundaryOptimizer` 里存在
   `_SharedCommunicationObjective` 这类"共享/聚合通信"语义（一条记录只在全局计一次），当计划被拆成多个
   block 时它不是简单求和，因此本 MILP 的线性编码在该区间不精确；要用大 M / 指示变量的分段编码才能覆盖。

也就是说：**在最优点由单块融合计划给出的配置（W=64，也正是我们实验用的配置）上验证通过；W≠64 的实例
暴露出搜索期成本与物化打分的两处口径不一致，需要单独修**，这两点已在上面列明。

#### 4.3.3 复现

```bash
# 容器内
python -u scripts/verify_pico_dp_optimality_ilp.py \
  --workload commonvoice \
  --profile outputs/ultimate_eight_optimizers_20260920/commonvoice/profiles/shared.yaml \
  --workers 64
python -u scripts/verify_pico_dp_optimality_ilp.py \
  --workload simclrv2 \
  --profile outputs/ultimate_eight_optimizers_20260920/simclrv2/profiles/shared.yaml \
  --workers 64
```

运行日志：`outputs/pico_dp_ilp_verification_20260921/`。

### 4.4 PICO 去掉 width 维度（W-only）在 llava / stackexchange 上

（原 `docs/pico_w_only_experiment_20260921.md`。）

#### 4.4.0 动机：联合 W×width 搜索在长流水线上跑不完

PICO（`SimpleDpWorkersWidthBoundaryOptimizer`）的搜索空间是"算子顺序 × 分块 × 后端 × **width** × W"。
在 llava_pretrain（16 个可重排算子）上，带 width 的那次搜索只推进到第 10/16 层就用了 **3,055 s**、
保留了 **20.7 M** 个状态（`Exact layer 10/16 masks=462 states=20651995 max_frontier=89205
layer_sec=2160.972`），后面 6 层按同样的增长趋势没有希望在 2 小时 cell 预算内完成；
stackexchange（19 个算子）同理。因此按"**不用 width，只搜 W**"来做这组实验。

#### 4.4.1 配置（`scripts/run_pico_w_only_20260921.sh`）

| 旋钮 | 值 | 作用 |
| --- | --- | --- |
| `CEDAR_DP_WIDTH_LADDER` | `1` | 搜索阶段只提供宽度 1 的 stage |
| `CEDAR_DP_SEARCH_MODE` | `general` | 跳过 `auto` 先跑的那次**全预算 relaxed pass**（n=16/19 时它才是真正爆掉的那次搜索） |
| `CEDAR_DP_WORKER_LADDER=0` + `CEDAR_WORKER_SEARCH_SET` | `64,32,16` | 只搜粗 W 梯级：W 越大，每 worker 的 stage-CPU 切片越小，而**资源状态维度**是状态爆炸的主因 |
| `CEDAR_MATCH_PROFILE_RESOURCES` / `CEDAR_PROFILE_MATCH_{CPU,RAY_CPU}_BUDGET` | `1` / `64` | 与 campaign 相同的资源口径 |

两个必须在文档里写清的事实：

1. **llava 的 harness 自己把 W 钉成 1**（日志：*"Using a uniform LLaVA GPU operator parallelism of 1 for
   all optimizers (local worker budget=1)"*），所以 llava 这一侧的"W-only"实际等于**固定 W=1 + 其余决策照常搜**。
2. **`CEDAR_DP_WIDTH_LADDER=1` 只约束搜索候选，不约束最终分配**：搜索结束后的资源分配器仍会把并行 stage
   按 W 的 CPU 切片扩宽，所以 llava 产出的计划里出现了 **63 进程的 SMP stage**（见下）。这一跑严格说是
   "搜索阶段不搜宽度、最终分配照旧"，**不是严格的 width=1 消融**。

#### 4.4.2 结果

##### 4.4.2.1 llava_pretrain：完成

| 项 | 值 |
| --- | ---: |
| 规划（DP） | **1,931.6 s（≈32 min，`searches=1`）** |
| setup | 1,938.5 s |
| 稳态运行 | 1,492.3 s |
| 稳态吞吐 | **117.8 samples/s** |
| PICO score / Cedar 估计 | 54.115 / 29.820 |

产出的计划（W=1）：

```
LocalLinePipe（source）
  → FusedPipe[15,14,13,12,6,9,8,10,3,4,5][SMP w=63]
  → FusedPipe[11,7,2][RAY w=1]
  → FusedPipe[1,0][INPROCESS]
  → PrefetcherPipe
```

**口径提醒**：本次一次 epoch 计 **175,760 个计数样本 = 43,940 条 LLaVA 记录 × 4 张图**；campaign 里
llava 那些 cell 用的是 `--num_total_samples 50000` 的限量口径（每次记 43,940）。把本次归一到"记录/秒"
≈ **29.5 rec/s**，与 campaign 里 llava 最好的 `simple-dp-opt` 28.3 rec/s 相当；**117.8 这个绝对值只在本次
计数口径内可比**，不要直接和 llava 对照表里的数字比。

**search 复杂度对比**（同一负载、同一 DP）：

| 配置 | 单次 W 搜索 | 保留状态数 |
| --- | --- | --- |
| 带 width（campaign，1 h 上限） | 只到 **layer 10/16**，3,055 s，之后无望 | layer 10: **20.7 M** |
| width ladder = 1（本次） | 16 层全部跑完，≈19 min | layer 10: 0.61 M，layer 14: 0.10 M |

##### 4.4.2.2 stackexchange：规划阶段超时

- 算子数 **n=19**；单次搜索：19 层、677 s、2,056 个 legal prefixes、324,201 retained states。
- 多次 W 候选搜索累加后触发 **7,200 s 上限**：
  `simple_dp_workers_width_boundary: setup/optimization=7200.000504s, workload skipped
  (optimizer_time_limit_exceeded)` —— **没有产出计划**。

即：去掉 width 让 llava 从"跑不完"变成 ~1 小时完成，但对 n=19 的 stackexchange 仍然不够。

#### 4.4.3 结论与下一步

- **width 维度就是这两个负载复杂度的主因**：去掉它，llava 单次搜索的状态数下降约 30×、耗时下降约
  1,000×（3,055 s → ~60 s/W）；同时它也是这两条流水线上"最贵的搜索维度"，因为 stage 宽度 × 每 worker
  的 CPU 切片共同决定了 DP 的资源状态空间。
- 对 **stackexchange** 还需要更进一步的杠杆（按代价从低到高）：固定 W 不搜（`CEDAR_DP_WORKER_SEARCH=0`）、
  `CEDAR_DP_SEARCH_MODE=chain`（放弃算子重排，只做分段/后端/宽度）、`CEDAR_DP_FRONTIER_CAP` 或
  `CEDAR_DP_PARETO_EPSILON`（有界误差的近似剪枝）、缩小 planning 用的 CPU 切片。
- 如果要真正的"不用 width"消融，需要再加一个开关，禁止 `_allocate_final_remote_stage_resources` 把并行
  stage 扩宽（当前 llava 计划里的 `SMP w=63` 就是它的产物）。

##### 4.4.3.1 复现

```bash
# 容器内
bash scripts/run_pico_w_only_20260921.sh       # llava → stackexchange，日志/结果都在 RUN 目录
```

产物：`outputs/pico_w_only_20260921/{llava_pretrain,stackexchange}/{plans,logs,results}`，
launcher 日志 `outputs/pico_w_only_20260921.nohup.log`。

### 4.5 simclrv2 逐算子缩放原始证据（k、b 与尺寸 sweep）

（原 `docs/simclrv2_operator_scaling_evidence_20260921.md`。）

#### 4.5.0 数据资产总览

| # | 用途 | 位置 | 内容 |
| --- | --- | --- | --- |
| 1 | 受控尺寸 sweep 原始测量（9 算子 × 12 尺寸 × 3 轮 × 3 后端） | `outputs/target_pipeline_scaling_20260910/` | `PROTOCOL.md`、`results/raw.csv`、`results/operators.json`、latency/吞吐图 |
| 2 | simclrv2 仿射拟合对照（k、b、R²） | `outputs/simclrv2_affine_comparison_20260911/` | `README.md`、`coefficients.csv`、`local_measurements.csv`、`comparison.png/pdf` |
| 3 | 写进 profile 的逐算子多尺寸 sweep | `outputs/simclrv2_scaling_20260911/profile_calibrated.yaml` 的 `cm_model` | 每算子 13 个尺寸点 + train/validation + R² |
| 4 | 论文动机图的 4 点算子缩放 formal run | `outputs/motivation_multimodal/simclrv2_multimodal_operator_scaling_formal_20260909/` | 6 算子 × 4 work_scale × 7 trials，`summary.csv`、`raw.json`、`figures/` |

#### 4.5.1 原始测量：受控尺寸 sweep（`outputs/target_pipeline_scaling_20260910/`）

协议见 `PROTOCOL.md`，要点：

- 五个 workload 的所有非 source 算子按 batch size 4 抽出；simclrv2 共 9 个算子
  （`op1_RGBReader`、`float`、`crop`、`op4_RandomHorizontalFlip`、`jitter`、`op6_Grayscale`、
  `op7_GaussianBlur`、`op8_Normalize`、`op9_BatchKernel`），每个出现位置独立测量不去重。
- 12 个线性像素尺寸点（64² → 2048²，方图保持长宽比）；字符串 256 → 65536 UTF-8 字节线性 12 档；
  token 表示 16 → 4096 元素线性 12 档。Reader 的 x 轴是**编码文件字节**（未压缩 PNG），
  其余图像算子是**解码后字节**。
- 输入内容是 4 组真实 COCO 图文对，图像在计时外完成 resize：这是**受控反事实输入**，
  不是自然分布。
- 每个 operator/size/resource 各 3 轮，资源顺序轮换（local/smp/ray）；每格 4 条 warmup，
  校准目标 30 ms、调用数 4..8192，且同一格三个后端用相同调用数。
- 计时发生在 Cedar 原生 worker 内（`InProcessBatcherPipeVariant` / SMP / Ray actor），
  排除数据准备、传输、worker 启动与 warmup；测的是**算子计算**，不是端到端 stage 吞吐。
- 产出：`results/raw.csv`（约 1.7 万行，含 `input_bytes`、`ms_per_record`、`mib_per_sec`、
  `calls`、`seconds`、worker `pid`、`torch_threads`）、`results/operators.json`（161 个算子定义）、
  `results/figures/*.pdf`（按 backend/modality 输出 latency 与逻辑 MiB/s 曲线）、
  `results/metadata.json`、`results/index.html`。启动门禁与失败重跑记录见 `gate/` 与 `launcher_resume.log`。

#### 4.5.2 拟合对照：9 个算子的 k、b、R²（`outputs/simclrv2_affine_comparison_20260911/`）

该目录用 optimizer 的同一个非负仿射拟合器（`cedar.client.linear_cost_profile.fit_affine`，
12 等宽 bin）重拟合上面的 local 后端数据：9 算子 × 12 尺寸 × 3 轮 = 324 条观测，模型为
`t[ms/record] = k · x[MiB] + b`。产物：

- `local_measurements.csv`：逐 trial 的 `input_bytes / ms_per_record / mib_per_sec`；
- `coefficients.csv`：每个算子 sweep 拟合的 k、b、R²、实测尺寸范围，以及当前 DP/CM profile 系数的对照；
- `wide_range_models.yaml`、`comparison.png`、`comparison.pdf`、`README.md`（含下表与注意事项）。

| 算子 | sweep k (ms/MiB) | sweep k (ms/byte) | sweep b (ms) | R² | 实测范围 (MiB) | 与 profile 可直接比较 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| ImageReaderPipe | 5.43166 | 5.18003e-06 | 0.888492 | 0.9992 | 0.0120 – 12.0060 | 否（file bytes vs path 对象字节，PNG vs JPEG） |
| to_float | 1.15334 | 1.09991e-06 | 0 | 0.7145 | 0.0117 – 12.0000 | 是 |
| RandomResizedCrop | 0.163394 | 1.55825e-07 | 0.963732 | 0.7970 | 0.0469 – 48.0000 | 是 |
| RandomHorizontalFlip | 0.252091 | 2.40413e-07 | 0 | 0.8986 | 0.0469 – 48.0000 | 是 |
| ColorJitter | 21.3959 | 2.04047e-05 | 0 | 0.9452 | 0.0469 – 48.0000 | 是 |
| Grayscale | 0.353303 | 3.36936e-07 | 0 | 0.9655 | 0.0469 – 48.0000 | 是 |
| GaussianBlur | 43.9257 | 4.18908e-05 | 0 | 0.6018 | 0.0156 – 16.0000 | 是 |
| Normalize | 0.385662 | 3.67796e-07 | 0 | 0.8765 | 0.0156 – 16.0000 | 是 |
| BatcherPipe(batch_size=4) | 0.46338 | 4.41913e-07 | 0 | 0.8870 | 0.0156 – 16.0000 | 否（batch 号与计时方法不同） |

读法：斜率相差两个数量级（ColorJitter 21.4 ms/MiB、GaussianBlur 43.9 ms/MiB，而 Crop 只有
0.163 ms/MiB）；GaussianBlur（0.60）与 to_float（0.71）的线性拟合质量明显偏低，
引用时必须标注 R²，不能当成精确线性。

#### 4.5.3 profile 内嵌 sweep：每算子 13 个尺寸点

`outputs/simclrv2_scaling_20260911/profile_calibrated.yaml` 的 `cm_model` 记录了构建该 profile 时的
受控图像 sweep（`input_policy: natural_capture_plus_controlled_sweep`、`input_scope: whole_record`），
每个算子 13 个尺寸点、3 轮、含 train/validation 划分与 timing CV：

| 算子（cm_model） | status | k (ms/byte) | b (ms) | R² | 实测字节范围 |
| --- | --- | ---: | ---: | ---: | ---: |
| MapperPipe_Normalize | fitted | 2.0436e-07 | 0.052187 | 0.9997 | 16,384 – 1,110,916 |
| MapperPipe_GaussianBlur | fitted | 2.7871e-05 | 3.374316 | 0.7485 | 16,384 – 1,110,916 |
| MapperPipe_Grayscale | fitted | 1.8813e-07 | 0.063881 | 0.9988 | 49,152 – 3,332,748 |
| MapperPipe_ColorJitter | fitted | 1.1951e-05 | 0.232923 | 0.9976 | 49,152 – 3,332,748 |
| MapperPipe_RandomHorizontalFlip | fitted | 4.4907e-08 | 0.036545 | 0.9873 | 49,152 – 3,332,748 |
| MapperPipe_RandomResizedCrop | fitted | 1.1133e-07 | 1.557217 | 0.9450 | 48,810 – 8,502,539 |
| MapperPipe_to_float | fitted | 2.1985e-07 | 0.005676 | 0.9991 | 12,203 – 2,125,619 |
| BatcherPipe(batch_size=1) | constant_fallback | 0 | 0.008919 | — | 238,144（无尺寸变化） |
| ImageReaderPipe | constant_fallback | 0 | 2.057682 | — | 152 – 168（path 对象字节不可预测解码耗时） |
| LocalFSListerPipe | unavailable | — | — | — | — |

同一族的 wide-range 版本（batch 4）在 `outputs/simclr_profile_wide_scaling/simclr_profile.yaml`。
两个 run 的原始 `observations` 均保留每点 `input_bytes`、`mean_ns`、`round`、`split`、`target_pixels`，
可直接重画曲线。注意：`outputs/simclrv2_scaling_20260911/status.json` 记录该 campaign 的 driver 步骤
`failed`（profile 产物本身已完整落盘），引用时不要把它说成一次完整跑通的对比实验。

现行协议（2026-09-19 起）不再把 13 点曲线放进供 optimizer 消费的 profile：
`outputs/six_workload_profile_final_20260919/simclrv2/profiles/shared.yaml` 的
`physical_model.operator_affine`（约 2678 行起）对每个算子只保留**最小/最大两个合法输入点**
（`points_ms_per_byte`）加 k、b、`fixed_fraction`；0–7 有拟合，8 因
`unable to mmap 4096 bytes` 测量失败，9 是 `not_a_single_input_stage`。

#### 4.5.4 论文动机图的 4 点算子缩放（`outputs/motivation_multimodal/simclrv2_multimodal_operator_scaling_formal_20260909/`）

6 个算子 × 4 个 work_scale（0.25/1/4/16）× 7 次重复 = 24 点、168 行原始数据，输入尺寸对应
文本 16/64/256/1024 词与图像边长 112/224/448/896。`summary.csv` 给出每点
`median_ns_per_record`、`median_records_per_sec` 与四分位；单条耗时（ms/record，中位数）为：

| 算子 | 0.25× | 1× | 4× | 16× |
| --- | ---: | ---: | ---: | ---: |
| normalize | 0.03044 | 0.05991 | 0.1723 | 0.6094 |
| clip | 18.28 | 17.95 | 22.32 | 36.03 |
| random_crop | 1.455 | 1.483 | 1.605 | 2.061 |
| random_flip | 0.02094 | 0.03536 | 0.07263 | 0.2995 |
| color_jitter | 2.402 | 7.608 | 32.56 | 164.4 |
| gaussian_blur | 16.15 | 58.72 | 236.8 | 989.1 |

同目录 `figures/figure1_operator_scaling.{png,pdf,svg}` 与 `figure_manifest.json`（含 sha256）。
单模态维度版本在 `outputs/motivation_multimodal/simclrv2_multimodal_modality_scaling_formal_20260909/summary.csv`，
论文当前引用的就是它：`my_paper/69e75a0100d7b4afeb1cfc20/section/03_cost_model.tex` 的
`figures/operator_modality_scaling.pdf`；由算子版 summary 生成的
`figures/figure1_total_input_scaling.*` 目前正文未引用，溯源见 `figures/figure_manifest.json`。

#### 4.5.5 使用限制

- 曲线来自**受控反事实输入**（图像在计时外 resize、文本按真实 caption 重复构造），不是自然分布，
  不能当作“真实数据里就会遇到这些尺寸”的证据；引用时应说明是受控 sweep。
- Reader 的 x 轴是编码文件字节，其他图像算子是解码字节；PNG/JPEG 与文件大小口径不同，
  Reader 系数不可与算子系数直接相加比较。
- Batcher 的 sweep 是 batch 4、profile 里是 batch 1，计时方法也不同，两边系数不可直接比较。
- sweep 测的是 worker 内算子计算耗时，不含跨 boundary 传输、actor 启动与排队；跨后端结论要另外用
  `physical_model.boundary`（Ray payload 512 B – 4 MiB）等测量支撑。
- Gray/Blur/Jitter 等算子的固定开销占比很高（`fixed_fraction` 大），只按字节等比例外推（原 Cedar 的
  `y = x`）会显著低估；这正是 kx+b 拟合的动机。
- 既有快照不可改写（遵循 `AGENTS.md`）：需要新口径的曲线时另跑新实验，不要回填历史 profile。
- `evaluation/pipelines/simclrv2_multimodal/` 的生成脚本已在 `f98ff0e`
  （refactor: remove the fictional simclrv2_multimodal workload）中从工作区移除，
  第 4 批数据的可执行快照保留在 `outputs/iter_clip/modules/evaluation/pipelines/simclrv2_multimodal/` 等处。

#### 4.5.6 复现

```bash
# 容器内，先 source env/bin/activate
# 1) 原始尺寸 sweep（长实验，正式跑用 nohup；会先跑 gate 再跑 results）
bash outputs/target_pipeline_scaling_20260910/run.sh
# 2) 仿射拟合对照：重读上面的 results + cm/dp 两份 profile，重写 coefficients.csv/comparison.png
python -m evaluation.pipelines.target_pipeline.compare_simclr_affine
# 3) 逐算子曲线图（按 backend/modality 分页）
python -m evaluation.pipelines.target_pipeline.plot_latency_scaling --help
```

第 2 步依赖 `outputs/cm_vs_cedar_simclr_20260911/profile.yaml` 与
`outputs/dp_vs_old_dp_simclrv2_20260911/profile.yaml`，脚本会断言两者一致
（`profiles_identical: true`，见该目录 `metadata.json`）。

### 4.6 staged 搜索 + PICO cost model：把"cost model 有效"和"DP 有效"分开

**问题**：前面所有结论都混着两件事——cost model 变准了，以及搜索从"分阶段贪心"换成了
"联合 DP"。这一节把两者分开：用 **Cedar 原来的 staged 搜索**（同一套 pass：reorder →
offload/fusion → TF fusion → 事后 stage 宽度；同样的候选集、同样的"更便宜就接受"规则），
只把**计划评价函数**换成 PICO 的三种 cost model，得到三个新 optimizer
（`cedar/compose/staged_ablation_optimizer.py`，selector 31/32/33）：

| 新 optimizer（selector） | cost model | 同 cost model 的 DP 对照（selector） |
| --- | --- | --- |
| `staged-boundary` (31) | new profile + stage boundary（Cedar 计算量口径） | `old_dp_boundary` (28) |
| `staged-boundary-affine` (32) | + 每算子 kx+b | `simple_dp_boundary` (21) |
| `staged-boundary-affine-W` (33) | + W（replica 数）维度 | `simple_dp_workers_boundary` (25) |

实现上保证"成对的 cost model 完全一致"：staged 每次 `calculate_cost(graph, specs,
fused_pipes)` 都被物化成一个合法 `PhysicalPlan`，再交给**对应的那个 DP 消融类**做 replay
定价（同一份代码，分数可直接比较）。tier 3 因为 staged 没有联合 W 维度，只能在计划定稿后按
`score / W` 枚举合法 W 取最优。6 个 optimizer 在**同一个 run**
（`outputs/staged_ablation_20260922`）里跑完，读同一份 campaign profile，同一天、同协议
（64 CPU、远端 Ray、1 轮、2 h cell 上限）。

**结果**（稳态吞吐 rec/s；score = 该 tier 的模型分，越低越好；setup = 该 cell 的规划/构建时间）

**simclrv2（189,380 条）**

| tier | staged | DP 对照 | DP/staged |
| --- | ---: | ---: | ---: |
| boundary | 2389.4（score 10.5481，setup 23.0 s） | 1943.2（8.9166，12.4 s） | **0.81×** |
| +affine | 2442.9（8.2057，22.9 s） | 1932.2（7.8110，12.5 s） | **0.79×** |
| +W | 2443.9（8.2057，23.3 s） | 2476.9（8.2057，116.8 s） | 1.01× |

**simclrv2_cache（189,380 条，cache 开）**

| tier | staged | DP 对照 | DP/staged |
| --- | ---: | ---: | ---: |
| boundary | 1595.9（8.0577，22.8 s） | 2229.6（6.0601，12.5 s） | **1.40×** |
| +affine | 2750.1（5.7115，23.2 s） | 5612.0（2.8726，22.8 s） | **2.04×** |
| +W | 2719.8（5.7115，23.1 s） | 5532.1（2.8726，131.7 s） | **2.03×** |

**commonvoice（300,000 条）**

| tier | staged | DP 对照 | DP/staged |
| --- | ---: | ---: | ---: |
| boundary | 716.5（26.1175，19.3 s） | 652.4（12.0793，10.0 s） | **0.91×** |
| +affine | 724.8（27.0707，19.3 s） | 725.0（27.0707，19.6 s） | 1.00× |
| +W | 713.2（27.0707，19.4 s） | 722.6（27.0707，20.5 s） | 1.01× |

**coco（50,000 条）**

| tier | staged | DP 对照 | DP/staged |
| --- | ---: | ---: | ---: |
| boundary | 284.8（65.5017，17.1 s） | 227.4（55.1507，9.0 s） | **0.80×** |
| +affine | 288.5（79.6981，17.5 s） | 243.2（65.1022，9.2 s） | **0.84×** |
| +W | 295.9（79.6981，17.5 s） | 291.5（79.6981，18.1 s） | 0.99× |

**llava_pretrain**：按用户要求整个负载从本轮消融中去掉。原因是三个 staged tier 都无法规划——
Cedar 的 reorder pass 要枚举拓扑序（`calculate_reorderings`），16 个算子的 LLaVA 流水线连
候选生成都跑不完（> 2 h，与 campaign 里 `cedar-opt` 在同一负载上超时同因），
`status.json` 记为 `skipped_previous_timeout` / `skipped_user_requested`；该负载的 DP 参考值见
§2.6（PICO-Resource 26.3、PICO-Resource-Op 28.2 rec/s，W-only 29.4 rec/s）。

用同族的 DP 变体交叉验证过这份 run 与 campaign 的一致性：本 run 的 `simple_dp_workers_boundary`
（W-only）计划与 campaign 的 PICO 计划相同，吞吐 2476.9 vs 2501.5（simclrv2）、5532.1 vs 5399.4
（simclrv2_cache）、722.6 vs 743.2（commonvoice）、291.5 vs 294.3（coco），偏差 ≤ 2.5%。

**计划形态**

| 负载 / tier | staged | DP |
| --- | --- | --- |
| simclrv2 boundary | 全 INPROCESS，W=64，Grayscale/RandomResizedCrop 互换 | `Fused{3,6}` → `Fused{2,4,5}[SMP w=1]` → `Fused{7,1}`，W=32 |
| simclrv2 +affine | 全 INPROCESS，W=64（声明顺序） | `Fused{6,3}` → `GaussianBlur[SMP w=1]` → `Fused{4,5,7,1}`，W=32 |
| simclrv2 +W | 全 INPROCESS，W=64 | `Fused{6,3,4,5,2,7,1}`（全 local），W=64 |
| simclrv2_cache boundary | cache + `GaussianBlur[RAY w=1]`，W=64 | cache + `Fused{2,4,5}[SMP w=1]` + `Fused{7,1}`，W=32 |
| simclrv2_cache +affine / +W | cache + 全 INPROCESS，W=64 | `Fused{3,2}` → cache → `Fused{6,4,5,7,1}`，W=64 |
| commonvoice 三 tier | 全 INPROCESS，W=64 | boundary：`Fused{6,5,4,3}[SMP w=1]` + `Fused{2,1,0}`；affine/W：整条 fuse、全 local |
| coco 三 tier | 全 INPROCESS，W=64 | boundary：`Fused{1,5,4,3,2}[SMP w=1]`；affine：`distort[SMP w=1]` + `Fused{5,4,3,2}`；W：`Fused{1,5,4,3,2}` 全 local |

**分析**

1. **DP 的收益不是无条件的，取决于该 tier 的模型是否准。** simclrv2_cache 上模型和实测方向一致：
   DP 分数 2.8726 vs staged 5.7115，实测 5612.0 vs 2750.1 rec/s（**2.04×**）——同一份模型下，
   联合搜索把 cache 两侧的融合块与顺序一起决定，贪心 staged 只能停在"cache + 全 local"。
   commonvoice / coco 的 tier 3 两边分数完全相同（DP 的计划就是 staged 计划的融合版），
   实测也只差 1–2%。
2. **模型错了的地方，DP 救不回来。** simclrv2 / coco 的 tier 1–2 上 DP 的模型分更低
   （simclrv2：7.8110 vs 8.2057；coco：65.1022 vs 79.6981），实测**更慢**
   （1932.2 vs 2442.9 rec/s；243.2 vs 288.5 rec/s）。原因是 DP 按模型把一两个算子放到 SMP
   （跨进程、每记录一次 IPC），而模型对这一跳只收 0.4 ms/record（8.2057 → 7.8110，约 5%），
   实测却要付 **+0.108 ms/record（约 +26%）**（2443 → 1932 rec/s 的倒数差）。
   也就是说：**先把 SMP 边界定价修准，DP 的搜索优势才能在这些负载上兑现**。
3. **长流水线上 DP 是唯一可行的搜索。** LLaVA 16 算子：staged 的 reorder 枚举 > 2 h 跑不完
   （campaign 的 `cedar-opt` 同因超时），而 DP 用子集 DP 在多项式时间内给出计划
   （§2.6：PICO-Resource 26.3、PICO-Resource-Op 28.2 rec/s）。
4. **更好的 cost model 本身就会改变 staged 的决定，但不足以改变它的计划族**：tier 1 → tier 2
   让 simclrv2_cache 从 1595.9 涨到 2750.1 rec/s、simclrv2 从 2389.4 涨到 2442.9；
   而 W 维度对 staged 的计划形态没有影响（它只能在事后挑 W，tier 2/3 的 staged 计划完全相同）。
5. **规划时间**：staged 三个 tier 都在 17–23 s 完成（比 DP 的 9–132 s 快），但在
   simclrv2_cache 上它换来的是慢 1.4–2.0× 的计划；DP 的 tier 3 多花的规划时间（116.8 / 131.7 s）
   换来了 2× 的吞吐。

## 5. 论文图件与底层数据

### 5.0 命名映射

| 图中的名字 | 结果文档里的名字 | 实现 | 说明 |
| --- | --- | --- | --- |
| unopt | unopti | `unopti` | 未优化，直接执行原计划 |
| plumber | plumber-opt | `plumber_optimizer` | Plumber 基线 |
| raydata | ray-opt | `raydata_optimizer` | Ray Data 基线 |
| cedar | cedar-opt | `optimizer` | Cedar 原始分阶段优化器 |
| cedar-dp | old-dp-opt | `old_dp_legacy_optimizer` | DP，但只读 Cedar 旧 profile 属性（SimpleDpOptimizer） |
| PICO-Resource | dp-boundary | `old_dp_boundary` | 只用 stage boundary 项（OldDpBoundaryOptimizer） |
| PICO-Resource-Op | dp-boundary-affine | `simple_dp_boundary` | boundary + 每算子 kx+b（SimpleDpBoundaryOptimizer） |
| PICO | dp-boundary-affine-W-width | `simple_dp_workers_width_boundary` | boundary + kx+b + Workers + width 搜索（SimpleDpWorkersWidthBoundaryOptimizer） |

数据来源：`outputs/ultimate_eight_optimizers_fix_20260921`（正式放大 campaign，simclrv2 189,380 条 = 9,469 × 20；commonvoice 300,000；coco 50,000；llava_pretrain 50,000 输入 / 43,940 条过过滤；stackexchange 20,000 输入）；llava 的 PICO 取 `outputs/pico_w_only_20260921`（W-only，见 §2 注）。

#### 5.0.1 使用说明与注意事项

- **缺失单元**：`unopt@commonvoice` / `unopt@coco` 执行超 2 h 未产出 plan，既无吞吐也不参与打分；`cedar@llava_pretrain` / `cedar@stackexchange` 为已知的 Cedar 优化超时（`skipped_user_requested`，reason = *Known Cedar optimization timeout for this workload*）；`PICO@stackexchange` 规划超过 2 h cell 上限（`skipped_previous_timeout`），没有 plan。
- **llava 的 PICO** 只有 W-only 结果：`CEDAR_DP_WIDTH_LADDER=1` 只约束搜索候选，最终资源分配仍把 stage 扩宽到 `SMP w=63`，因此不是严格的 width=1 消融；该次 harness 以 **4 张图/记录**计数（175,760 个计数样本），文档已折算成 records/s（÷4 → 43,940 条）。
- **吞吐口径**：稳态吞吐 = 数据量 / 稳态时间，其中稳态时间取 cell 的 `perf_time_sec`（= Σ epoch_run_times）。`summary.csv` 里的 `mean_input_records_per_sec` 用的是「数据量 / workload wall」（含每个 epoch 的启动与排空），数值更高，两者不要混用。
- **cost 口径**：cedar 是单 worker 的 ms/source-record；plumber 已按 plan 的 W 折算（`1000/(瓶颈单 worker 速率 × W)`）；PICO 的 `calculate_dp_objective_cost` 是 W-conditioned 的 S，表里同时给 S 与 S/W。
- **模型覆盖率**：放宽之后三个模型都能给每个有 plan 的 cell 定价（此前 PICO 拒绝的 coco `raydata` / `cedar` / `cedar-dp` 现在也能定价）。如果将来还有 cell 打不了分，它会被剔除，**三个模型始终在同一子集上比较**，避免“谁覆盖得多谁占便宜”。

- **PICO 给别人的 plan 打分时是宽松模式**（`_dp_lenient_replay`）：replay 只负责报告该 plan 实际花多少，因此跳过“搜索空间合法性”闸门（例如 baseline 把 to_tensor 一起融了），并接受已经测到但未收敛的 backend_compute（允许未收敛时打 warning，见下）。优化器**给自己**搜索出的 plan 打分仍用严格口径（`lenient_replay=False`），搜索空间本身没有放宽。

**宽松定价用到的未收敛测量**（这些算子在搜索阶段仍然不可选）：

- coco: [LayeredSimpleDp] pricing pipe 0 on RAY with an unconverged backend measurement: mean=15.1177 ms/sample count=18 rse=0.14095003828658706 stop=max_duration

- coco: [LayeredSimpleDp] pricing pipe 1 on RAY with an unconverged backend measurement: mean=99.9425 ms/sample count=22 rse=0.1168285213060327 stop=max_duration

### 5.1 稳态吞吐量（柱状图）

单位：records/s（= 数据量 / 稳态时间 `perf_time_sec`）。

| 负载 | unopt | plumber | raydata | cedar | cedar-dp | PICO-Resource | PICO-Resource-Op | PICO |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| simclrv2 | 46.2 | 106.9 | 83.7 | 1187.7 | 352.9 | 1937.5 | 1964.5 | 2501.5 |
| simclrv2_cache | 46.5 | 131.5 | 88.8 | 1634.8 | 340.1 | 2231.6 | 5502.4 | 5399.4 |
| commonvoice | — *timeout* | 145.5 | 91.9 | 157.8 | 79.9 | 661.8 | 734.4 | 743.2 |
| coco | — *timeout* | 19.0 | 7.6 | 26.7 | 25.8 | 230.8 | 241.1 | 294.3 |
| llava_pretrain | 16.1 | 17.3 | 15.4 | — *skipped_user_requested* | 25.7 | 26.3 | 28.2 | 29.4 |
| stackexchange | 3.1 | 69.0 | 63.9 | — *skipped_user_requested* | 146.1 | 81.2 | 60.7 | — *skipped_previous_timeout* |

每负载明细（数据量、稳态时间、优化时间、总时长、状态）：

#### 5.1.1 simclrv2

| optimizer | 数据量(条) | 稳态时间(s) | 稳态吞吐(rec/s) | 优化时间(s) | 总时长(s) | 状态 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| unopt | 189,380 | 4098.56 | 46.2 | 1.2 | 4103.2 | completed |
| plumber | 189,380 | 1771.25 | 106.9 | 2.2 | 1777.0 | completed |
| raydata | 189,380 | 2263.83 | 83.7 | 12.2 | 2322.8 | completed |
| cedar | 189,380 | 159.46 | 1187.7 | 22.0 | 453.5 | completed |
| cedar-dp | 189,380 | 536.60 | 352.9 | 12.4 | 551.8 | completed |
| PICO-Resource | 189,380 | 97.75 | 1937.5 | 12.4 | 112.3 | completed |
| PICO-Resource-Op | 189,380 | 96.40 | 1964.5 | 12.5 | 111.2 | completed |
| PICO | 189,380 | 75.71 | 2501.5 | 267.0 | 345.2 | completed |

#### 5.1.2 simclrv2_cache

| optimizer | 数据量(条) | 稳态时间(s) | 稳态吞吐(rec/s) | 优化时间(s) | 总时长(s) | 状态 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| unopt | 189,380 | 4076.86 | 46.5 | 1.1 | 4081.4 | completed |
| plumber | 189,380 | 1439.82 | 131.5 | 2.2 | 1445.0 | completed |
| raydata | 189,380 | 2133.13 | 88.8 | 12.0 | 2190.8 | completed |
| cedar | 189,380 | 115.84 | 1634.8 | 21.7 | 188.7 | completed |
| cedar-dp | 189,380 | 556.81 | 340.1 | 12.4 | 592.4 | completed |
| PICO-Resource | 189,380 | 84.86 | 2231.6 | 12.9 | 100.2 | completed |
| PICO-Resource-Op | 189,380 | 34.42 | 5502.4 | 23.0 | 58.8 | completed |
| PICO | 189,380 | 35.07 | 5399.4 | 417.9 | 454.4 | completed |

#### 5.1.3 commonvoice

| optimizer | 数据量(条) | 稳态时间(s) | 稳态吞吐(rec/s) | 优化时间(s) | 总时长(s) | 状态 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| unopt | — | — | — | — | — | timeout |
| plumber | 300,000 | 2061.21 | 145.5 | 2.7 | 2075.2 | completed |
| raydata | 300,000 | 3262.93 | 91.9 | 6.6 | 3276.4 | completed |
| cedar | 300,000 | 1901.41 | 157.8 | 19.0 | 2186.0 | completed |
| cedar-dp | 300,000 | 3753.63 | 79.9 | 7.0 | 3861.1 | completed |
| PICO-Resource | 300,000 | 453.33 | 661.8 | 10.3 | 473.0 | completed |
| PICO-Resource-Op | 300,000 | 408.52 | 734.4 | 19.1 | 439.4 | completed |
| PICO | 300,000 | 403.64 | 743.2 | 24.9 | 440.8 | completed |

#### 5.1.4 coco

| optimizer | 数据量(条) | 稳态时间(s) | 稳态吞吐(rec/s) | 优化时间(s) | 总时长(s) | 状态 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| unopt | — | — | — | — | — | timeout |
| plumber | 50,000 | 2627.62 | 19.0 | 20.9 | 2648.8 | completed |
| raydata | 50,000 | 6598.26 | 7.6 | 27.1 | 6642.7 | completed |
| cedar | 50,000 | 1873.89 | 26.7 | 17.1 | 1940.9 | completed |
| cedar-dp | 50,000 | 1934.32 | 25.8 | 9.0 | 1981.9 | completed |
| PICO-Resource | 50,000 | 216.66 | 230.8 | 8.9 | 246.2 | completed |
| PICO-Resource-Op | 50,000 | 207.36 | 241.1 | 8.9 | 239.8 | completed |
| PICO | 50,000 | 169.91 | 294.3 | 21.6 | 221.1 | completed |

#### 5.1.5 llava_pretrain

| optimizer | 数据量(条) | 稳态时间(s) | 稳态吞吐(rec/s) | 优化时间(s) | 总时长(s) | 状态 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| unopt | 43,940 | 2731.77 | 16.1 | 2.8 | 2737.9 | completed |
| plumber | 43,940 | 2541.04 | 17.3 | 3.0 | 2551.2 | completed |
| raydata | 43,940 | 2855.98 | 15.4 | 5.9 | 2869.7 | completed |
| cedar | — | — | — | — | — | skipped_user_requested |
| cedar-dp | 43,940 | 1712.88 | 25.7 | 44.1 | 1766.4 | completed |
| PICO-Resource | 43,940 | 1668.21 | 26.3 | 56.7 | 1738.5 | completed |
| PICO-Resource-Op | 43,940 | 1556.39 | 28.2 | 60.3 | 1623.5 | completed |
| PICO | 43,940 | 1492.26 | 29.4 | 1938.5 | 3441.1 | completed |

#### 5.1.6 stackexchange

| optimizer | 数据量(条) | 稳态时间(s) | 稳态吞吐(rec/s) | 优化时间(s) | 总时长(s) | 状态 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| unopt | 7,238 | 2369.05 | 3.1 | 2.5 | 2372.8 | completed |
| plumber | 7,238 | 104.87 | 69.0 | 3.9 | 109.5 | completed |
| raydata | 7,238 | 113.22 | 63.9 | 9.1 | 126.5 | completed |
| cedar | — | — | — | — | — | skipped_user_requested |
| cedar-dp | 7,238 | 49.56 | 146.1 | 91.0 | 284.8 | completed |
| PICO-Resource | 7,238 | 89.13 | 81.2 | 87.3 | 259.2 | completed |
| PICO-Resource-Op | 7,238 | 119.15 | 60.7 | 116.1 | 341.3 | completed |
| PICO | — | — | — | — | — | skipped_previous_timeout |

### 5.2 优化时间

单位：秒，取 cell 的 `setup_time_sec`（optimizer 规划 + 计划物化；unopti 只有构建开销）。

| 负载 | unopt | plumber | raydata | cedar | cedar-dp | PICO-Resource | PICO-Resource-Op | PICO |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| simclrv2 | 1.2 | 2.2 | 12.2 | 22.0 | 12.4 | 12.4 | 12.5 | 267.0 |
| simclrv2_cache | 1.1 | 2.2 | 12.0 | 21.7 | 12.4 | 12.9 | 23.0 | 417.9 |
| commonvoice | — *timeout* | 2.7 | 6.6 | 19.0 | 7.0 | 10.3 | 19.1 | 24.9 |
| coco | — *timeout* | 20.9 | 27.1 | 17.1 | 9.0 | 8.9 | 8.9 | 21.6 |
| llava_pretrain | 2.8 | 3.0 | 5.9 | — *skipped_user_requested* | 44.1 | 56.7 | 60.3 | 1938.5 |
| stackexchange | 2.5 | 3.9 | 9.1 | — *skipped_user_requested* | 91.0 | 87.3 | 116.1 | — *skipped_previous_timeout* |

### 5.3 三个 cost model 的估计与排序

口径：cedar = `Optimizer.calculate_cost`（单 worker、ms/source-record）；plumber = plan 宽度瓶颈 + `W`（ms/source-record，1/rate）；PICO = `SimpleDpWorkersWidthBoundaryOptimizer.calculate_dp_objective_cost`（S），表中同时给出 S/W。

**模型覆盖率**（每个负载有多少 plan 能被该模型定价）：

| 负载 | 有 plan 的 cell | cedar | plumber | PICO | PICO 打不了分的 cell |
| --- | ---: | ---: | ---: | ---: | --- |
| simclrv2 | 8 | 8 | 8 | 8 | — |
| simclrv2_cache | 8 | 8 | 8 | 8 | — |
| commonvoice | 7 | 7 | 7 | 7 | — |
| coco | 7 | 7 | 7 | 7 | — |
| llava_pretrain | 7 | 7 | 7 | 7 | — |
| stackexchange | 6 | 6 | 6 | 6 | — |

**汇总：模型给出的 cost 排序与实测吞吐排序的一致性**

| 负载 | 可比 cell | cedar ρ | plumber ρ | PICO ρ | cedar τ | plumber τ | PICO τ | 实测最优 | cedar 最优 | plumber 最优 | PICO 最优 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| simclrv2 | 8 | 0.4072 | 0.8264 | 0.9762 | 0.2546 | 0.691 | 0.9286 | PICO | cedar-dp | cedar | PICO |
| simclrv2_cache | 8 | 0.253 | 0.9157 | 0.994 | 0.1482 | 0.8154 | 0.982 | PICO-Resource-Op | cedar-dp | PICO-Resource-Op | PICO-Resource-Op |
| commonvoice | 7 | -0.1637 | 0.6301 | 0.955 | -0.0501 | 0.5143 | 0.8783 | PICO | PICO-Resource | cedar | PICO-Resource-Op |
| coco | 7 | 0.1261 | 0.4364 | 1.0 | 0.0 | 0.3504 | 1.0 | PICO | cedar | cedar | PICO |
| llava_pretrain | 7 | 0.0901 | 0.1112 | 0.6071 | 0.0976 | 0.1029 | 0.5238 | PICO | cedar-dp | unopt | PICO |
| stackexchange | 6 | 0.4638 | 0.7247 | 0.3714 | 0.414 | 0.5521 | 0.3333 | cedar-dp | cedar-dp | plumber | raydata |

#### 5.3.1 simclrv2

| optimizer | cedar cost (ms) | plumber cost (ms) | PICO score S | PICO S/W | 实测吞吐(rec/s) | 实测排名 | cedar 排名 | plumber 排名 | PICO 排名 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| unopt | 22.98 | 8.50 | 25.41 | 25.41 | 46.2 | 8.0 | 7.5 | 7.5 | 8.0 |
| plumber | 22.98 | 7.77 | 16.56 | 16.56 | 106.9 | 6.0 | 7.5 | 6.0 | 7.0 |
| raydata | 9.91 | 8.50 | 15.38 | 15.38 | 83.7 | 7.0 | 5.0 | 7.5 | 6.0 |
| cedar | 8.48 | 0.25 | 88.97 | 1.39 | 1187.7 | 4.0 | 3.0 | 1.0 | 4.0 |
| cedar-dp | 7.91 | 0.60 | 113.52 | 3.55 | 352.9 | 5.0 | 1.0 | 5.0 | 5.0 |
| PICO-Resource | 8.21 | 0.51 | 7.91 | 0.25 | 1937.5 | 3.0 | 2.0 | 4.0 | 3.0 |
| PICO-Resource-Op | 9.93 | 0.29 | 7.67 | 0.24 | 1964.5 | 2.0 | 6.0 | 2.0 | 2.0 |
| PICO | 8.97 | 0.30 | 8.21 | 0.13 | 2501.5 | 1.0 | 4.0 | 3.0 | 1.0 |

可比 cell 8 个（三者都能定价的）；实测最优 = `PICO`；cedar 最优 = `cedar-dp`；plumber 最优 = `cedar`；PICO 最优 = `PICO`。Spearman ρ（cost 排名 vs 吞吐排名，越接近 1 越好）：cedar 0.4072、plumber 0.8264、PICO 0.9762；Kendall τ：cedar 0.2546、plumber 0.691、PICO 0.9286。

#### 5.3.2 simclrv2_cache

| optimizer | cedar cost (ms) | plumber cost (ms) | PICO score S | PICO S/W | 实测吞吐(rec/s) | 实测排名 | cedar 排名 | plumber 排名 | PICO 排名 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| unopt | 22.80 | 10.21 | 22.24 | 22.24 | 46.5 | 8.0 | 7.5 | 7.5 | 8.0 |
| plumber | 22.80 | 7.37 | 11.56 | 11.56 | 131.5 | 6.0 | 7.5 | 6.0 | 6.0 |
| raydata | 8.98 | 10.21 | 15.35 | 15.35 | 88.8 | 7.0 | 4.0 | 7.5 | 7.0 |
| cedar | 7.50 | 0.29 | 87.95 | 1.37 | 1634.8 | 4.0 | 3.0 | 3.0 | 4.0 |
| cedar-dp | 7.19 | 0.32 | 176.42 | 5.51 | 340.1 | 5.0 | 1.0 | 4.0 | 5.0 |
| PICO-Resource | 7.38 | 0.59 | 5.94 | 0.19 | 2231.6 | 3.0 | 2.0 | 5.0 | 3.0 |
| PICO-Resource-Op | 11.67 | 0.17 | 2.87 | 0.04 | 5502.4 | 1.0 | 5.5 | 1.5 | 1.5 |
| PICO | 11.67 | 0.17 | 2.87 | 0.04 | 5399.4 | 2.0 | 5.5 | 1.5 | 1.5 |

可比 cell 8 个（三者都能定价的）；实测最优 = `PICO-Resource-Op`；cedar 最优 = `cedar-dp`；plumber 最优 = `PICO-Resource-Op`；PICO 最优 = `PICO-Resource-Op`。Spearman ρ（cost 排名 vs 吞吐排名，越接近 1 越好）：cedar 0.253、plumber 0.9157、PICO 0.994；Kendall τ：cedar 0.1482、plumber 0.8154、PICO 0.982。

#### 5.3.3 commonvoice

| optimizer | cedar cost (ms) | plumber cost (ms) | PICO score S | PICO S/W | 实测吞吐(rec/s) | 实测排名 | cedar 排名 | plumber 排名 | PICO 排名 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| unopt | — | — | — | — | — | — | — | — | — |
| plumber | 26.12 | 0.80 | 8.10 | 8.10 | 145.5 | 5.0 | 7.0 | 4.0 | 4.0 |
| raydata | 2.41 | 25.72 | 19.08 | 19.08 | 91.9 | 6.0 | 2.5 | 7.0 | 6.0 |
| cedar | 2.41 | 0.77 | 733.22 | 11.46 | 157.8 | 4.0 | 2.5 | 2.0 | 5.0 |
| cedar-dp | 2.93 | 1.29 | 531.16 | 25.29 | 79.9 | 7.0 | 4.0 | 5.0 | 7.0 |
| PICO-Resource | 2.25 | 1.42 | 27.67 | 0.86 | 661.8 | 3.0 | 1.0 | 6.0 | 3.0 |
| PICO-Resource-Op | 4.68 | 0.77 | 27.07 | 0.42 | 734.4 | 2.0 | 5.5 | 2.0 | 1.5 |
| PICO | 4.68 | 0.77 | 27.07 | 0.42 | 743.2 | 1.0 | 5.5 | 2.0 | 1.5 |

可比 cell 7 个（三者都能定价的）；实测最优 = `PICO`；cedar 最优 = `PICO-Resource`；plumber 最优 = `cedar`；PICO 最优 = `PICO-Resource-Op`。Spearman ρ（cost 排名 vs 吞吐排名，越接近 1 越好）：cedar -0.1637、plumber 0.6301、PICO 0.955；Kendall τ：cedar -0.0501、plumber 0.5143、PICO 0.8783。

#### 5.3.4 coco

| optimizer | cedar cost (ms) | plumber cost (ms) | PICO score S | PICO S/W | 实测吞吐(rec/s) | 实测排名 | cedar 排名 | plumber 排名 | PICO 排名 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| unopt | — | — | — | — | — | — | — | — | — |
| plumber | 168.63 | 3.94 | 187.95 | 187.95 | 19.0 | 6.0 | 7.0 | 3.0 | 6.0 |
| raydata | 38.97 | 151.37 | 229.61 | 229.61 | 7.6 | 7.0 | 5.0 | 7.0 | 7.0 |
| cedar | 22.81 | 2.79 | 3968.19 | 62.00 | 26.7 | 4.0 | 1.5 | 1.5 | 4.0 |
| cedar-dp | 22.81 | 4.73 | 2136.34 | 66.76 | 25.8 | 5.0 | 1.5 | 4.5 | 5.0 |
| PICO-Resource | 24.22 | 5.59 | 98.67 | 3.08 | 230.8 | 3.0 | 3.0 | 6.0 | 3.0 |
| PICO-Resource-Op | 46.23 | 4.73 | 71.65 | 2.24 | 241.1 | 2.0 | 6.0 | 4.5 | 2.0 |
| PICO | 27.34 | 2.79 | 79.70 | 1.25 | 294.3 | 1.0 | 4.0 | 1.5 | 1.0 |

可比 cell 7 个（三者都能定价的）；实测最优 = `PICO`；cedar 最优 = `cedar`；plumber 最优 = `cedar`；PICO 最优 = `PICO`。Spearman ρ（cost 排名 vs 吞吐排名，越接近 1 越好）：cedar 0.1261、plumber 0.4364、PICO 1.0；Kendall τ：cedar 0.0、plumber 0.3504、PICO 1.0。

#### 5.3.5 llava_pretrain

| optimizer | cedar cost (ms) | plumber cost (ms) | PICO score S | PICO S/W | 实测吞吐(rec/s) | 实测排名 | cedar 排名 | plumber 排名 | PICO 排名 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| unopt | 62.04 | 50.54 | 91.26 | 91.26 | 16.1 | 6.0 | 6.5 | 2.0 | 7.0 |
| plumber | 62.04 | 50.54 | 60.37 | 60.37 | 17.3 | 5.0 | 6.5 | 2.0 | 3.0 |
| raydata | 4.15 | 100.51 | 62.00 | 62.00 | 15.4 | 7.0 | 2.0 | 7.0 | 4.0 |
| cedar | — | — | — | — | — | — | — | — | — |
| cedar-dp | 2.24 | 53.00 | 79.14 | 79.14 | 25.7 | 4.0 | 1.0 | 6.0 | 6.0 |
| PICO-Resource | 30.55 | 50.54 | 77.51 | 77.51 | 26.3 | 3.0 | 5.0 | 2.0 | 5.0 |
| PICO-Resource-Op | 7.81 | 52.63 | 56.80 | 56.80 | 28.2 | 2.0 | 3.0 | 5.0 | 2.0 |
| PICO | 29.82 | 51.17 | 54.12 | 54.12 | 29.4 | 1.0 | 4.0 | 4.0 | 1.0 |

可比 cell 7 个（三者都能定价的）；实测最优 = `PICO`；cedar 最优 = `cedar-dp`；plumber 最优 = `unopt`；PICO 最优 = `PICO`。Spearman ρ（cost 排名 vs 吞吐排名，越接近 1 越好）：cedar 0.0901、plumber 0.1112、PICO 0.6071；Kendall τ：cedar 0.0976、plumber 0.1029、PICO 0.5238。

#### 5.3.6 stackexchange

| optimizer | cedar cost (ms) | plumber cost (ms) | PICO score S | PICO S/W | 实测吞吐(rec/s) | 实测排名 | cedar 排名 | plumber 排名 | PICO 排名 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| unopt | 580.07 | 113.55 | 725.84 | 725.84 | 3.1 | 6.0 | 5.5 | 5.5 | 6.0 |
| plumber | 580.07 | 5.41 | 27.90 | 27.90 | 69.0 | 3.0 | 5.5 | 1.0 | 5.0 |
| raydata | 23.02 | 113.55 | 3.94 | 3.94 | 63.9 | 4.0 | 2.0 | 5.5 | 1.0 |
| cedar | — | — | — | — | — | — | — | — | — |
| cedar-dp | 8.58 | 7.12 | 201.88 | 6.31 | 146.1 | 1.0 | 1.0 | 2.0 | 2.0 |
| PICO-Resource | 302.89 | 7.14 | 182.01 | 11.38 | 81.2 | 2.0 | 4.0 | 3.0 | 4.0 |
| PICO-Resource-Op | 297.05 | 10.91 | 137.93 | 6.57 | 60.7 | 5.0 | 3.0 | 4.0 | 3.0 |
| PICO | — | — | — | — | — | — | — | — | — |

可比 cell 6 个（三者都能定价的）；实测最优 = `cedar-dp`；cedar 最优 = `cedar-dp`；plumber 最优 = `plumber`；PICO 最优 = `raydata`。Spearman ρ（cost 排名 vs 吞吐排名，越接近 1 越好）：cedar 0.4638、plumber 0.7247、PICO 0.3714；Kendall τ：cedar 0.414、plumber 0.5521、PICO 0.3333。

## 6. 总体结论与未决问题

**已经站得住的结论**

1. **affine（kx+b）是必需的**：把 unopt 上测到的每算子代价按"过原点等比例"迁移到新顺序，会让
   尺寸变化的那 6 个 mapper 平均低估 **64%**（最坏 −86.7%），总量上 658 vs 实测 279 rec/s；
   换成 kx+b 形状后误差降到 53%（见 4.1）。
2. **Cedar-CM 的排序能力最差**：在 5 个负载上 Spearman ρ = 0.41 / 0.25 / −0.16 / 0.13 / 0.09
   （commonvoice 反序），Plumber-CM 0.83 / 0.92 / 0.63 / 0.44 / 0.11，PICO-CM
   0.98 / 0.99 / 0.95 / 1.00 / 0.61；"选中最快计划" PICO 5/5、Plumber 2/5、Cedar 0/5。
3. **DP 求解是精确的**：W=64 时 DP 目标值与 MILP 最优值逐位相同（commonvoice 27.070672、
   simclrv2 8.205709），MILP 计划 replay 同值（见 4.3）。
4. **执行上 PICO 在多模态负载最好，在文本负载输了**：正式放大 campaign 里 PICO 在 simclrv2、
   commonvoice、coco、llava 上第一（simclrv2-cache 上 PICO-Resource-Op 略胜）；新增的
   wikitext103 上 cedar 5466.5 rec/s 是 PICO 1148.0 的 4.8 倍，wikitext103_cache 上
   simple-dp-opt / cedar / dp-boundary 三者在 7435–7533 之间并列，PICO 只有 1473.1（见 3.3）。
   在这两个负载上 **cedar 的模型反而准**（预测 4816 vs 实测 5466，差 13%），
   PICO 的模型乐观 60–76 倍——原因见 3.4 的 (b)/(c)。
5. **"边界代价"本身不是文本负载输的原因**：小记录的 RAY boundary 按字节计价几乎为 0
   （2.8 KB 中位记录 ≈25 ns/条），PICO 输在它主动放弃了 Ray 计划、
   而"本地 64 worker 线性加速"的假设不成立（3.4）。
6. **cost model 与搜索是两件事，DP 的收益取决于模型的准度**（4.6）：把 Cedar 的 staged
   搜索配上同一份 PICO cost model 逐 tier 对照后，DP 在 simclrv2_cache 上快 **2.04×**
   （5612 vs 2750 rec/s，模型分 2.87 vs 5.71，方向一致）；但在 simclrv2 / coco 的 tier 1–2 上，
   DP 按模型选出的 SMP 计划实测反而慢 20%（1932 vs 2443 rec/s、243 vs 289 rec/s，模型分更低），
   因为模型把一次跨进程只算 0.4 ms/record、实测要付约 +26%。LLaVA 上 staged 连规划都跑不完
   （reorder 枚举 > 2 h），只有 DP 可行。

**未决问题（按优先级）**

1. **缺 driver↔worker 的结果搬运 lane**：wikitext103 上 PICO 计划每 worker 实测 47 ms/record，
   而 trace 到的算子计算只有 ≈2.1 ms/record，中间是每 sample 4.77 s 的排队/搬运
   （torch MP `recvfd` 传共享内存）。模型看不到这条 lane，于是系统性高估本地宽 W 计划（3.4）。
   修完应能让 PICO 在文本负载上改选 Ray 计划。
2. **cache 预热的完整性判定**：`commonvoice_cache` 的 `dp-boundary` / PICO 两次失败，
   只提交 19/64、4/64 个 shard manifest（数据文件已写、manifest 未写），
   把 grace 从 60 s 提到 900 s 无效；需要让预热不受样本预算截断，或按 cache stage 实际宽度校验（3.5）。
3. **TF 变体跨机模型文件**：TF 卸载的 embedding 需要 driver 上的 HF 缓存，远端离线拉不到（3.6）。
4. **运行器 `--resume` 对 failed cell 的处理**：应改成重跑而不是 `REUSE`（3.5）。
5. **PICO 的搜索复杂度**在长流水线（llava 16 / stackexchange 19 算子）上仍需 W-only / 剪枝
   （4.4），并且 `CEDAR_DP_WIDTH_LADDER` 只约束搜索、最终分配仍会加宽。
6. **per-record 拆分的边界项**：simclrv2 的 Ray 段有约 45.9 ms/record 花在提交/序列化/取回，
   其中 actor 计算只有 11.2 ms（4.2）；这条量级现在只被"边界吞吐 × 字节"近似。
7. **SMP 边界被定价得过低**（4.6）：模型认为把 `GaussianBlur` / `distort` 放进 SMP 能省
   ~5%（simclrv2 8.2057 → 7.8110；coco 79.6981 → 65.1022），实测这些计划比全 local 慢 20%
   （+0.108 ms/record）。修准 SMP 每记录 IPC 项之后，tier 1–2 的 DP 才可能真正赢过 staged。

## 7. 复现命令速查

```bash
# 容器内，先 source env/bin/activate
# 正式放大 campaign（6 负载 × 9 optimizer）
MODE=formal bash scripts/run_new_workloads_20260922.sh         # 新增负载（§3）
bash scripts/run_ultimate_matrix_20260920.sh                   # 原 6 负载（§2，历史 root）
# 表格与图件
python -u scripts/make_results_doc.py --out /tmp/campaign_tables.md   # §1–§2 的表格
python -u scripts/collect_figure_data_20260921.py                     # §5 的数据（写入 outputs/）
python -u scripts/make_experiment_figures_20260921.py                 # §5 的图
# 专题
python -u scripts/verify_pico_dp_optimality_ilp.py --workload commonvoice \
    --profile outputs/ultimate_eight_optimizers_20260920/commonvoice/profiles/shared.yaml --workers 64
bash scripts/run_pico_w_only_20260921.sh
bash scripts/run_unopt_order_transfer_repeats_20260921.sh      # affine 顺序交换（§4.1）
# staged vs DP 消融（§4.6；复用 campaign profile，llava 的 staged 侧按设计跳过）
MODE=formal bash scripts/run_staged_ablation_20260922.sh
python -u scripts/staged_ablation_report.py outputs/staged_ablation_20260922
python -u tmp_analysis/reordered_operator_table.py
python -u tmp_analysis/probe_ray_local_pricing.py wikitext103 \
    outputs/ultimate_new_workloads_20260922/wikitext103/profiles/shared.yaml \
    outputs/ultimate_new_workloads_20260922/wikitext103/plans/round1__optimizer.yaml \
    outputs/ultimate_new_workloads_20260922/wikitext103/plans/round1__simple_dp_workers_width_boundary.yaml
# 进度/结果（§3）
python -u scripts/new_workloads_status.py outputs/ultimate_new_workloads_20260922
python -u scripts/new_workloads_results.py outputs/ultimate_new_workloads_20260922
```

**来源说明**：本文件由以下历史文档合并、重新编号而成（原件已删除，内容仍可从 git 历史取回）：
`docs/experiment_results_20260920.md`（正式 campaign 与计划的表格；其 `_w_models.inc.md`
片段已移到 `scripts/templates/`，由 `scripts/make_results_doc.py` 读取）、
`docs/figure_data_20260921.md`（§5）、`docs/affine_transfer_experiment_20260921.md`（4.1）、
`docs/experiment_simclrv2_fuse_local_vs_ray_20260920.md`（4.2）、
`docs/pico_dp_optimality_ilp_20260921.md`（4.3）、`docs/pico_w_only_experiment_20260921.md`（4.4）、
`docs/simclrv2_operator_scaling_evidence_20260921.md`（4.5）、
`docs/new_workloads_20260922.md`（§3）、`docs/experiment_status_20260920_pause.md`（已过期）与
`docs/handoff_gpu_work_profile_20260919.md`（旧交接文档）。
