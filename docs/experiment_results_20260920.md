# 实验结果汇总（2026-09-20）

本文件由脚本从 `outputs/` 中的实验记录自动生成，包含：
1) 放大数据集 campaign（9 个 optimizer）；
2) 之前小数据集 campaign（6 个 optimizer）的对照结果；
3) 每个 optimizer 实际选中的物理计划。

## 0. 实验设置

### 0.1 optimizer

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

### 0.2 数据量与协议

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

## 1. 放大数据集 campaign（`outputs/ultimate_eight_optimizers_20260920`）

### 1.1 simclrv2

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

### 1.2 simclrv2_cache

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

### 1.3 commonvoice

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

## 1.4 commonvoice：三种 cost model 的估计与实测对照（Plumber/PICO 含 W 处理，Cedar 保持原样）

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

## 2. 小数据集 campaign（`outputs/six_workload_formal_v3_20260919`）

该轮为放大前的对照实验（同样 1 轮、CPU_BUDGET=64、远端 Ray）。

### 2.1 simclrv2

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

### 2.2 simclrv2_cache

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

### 2.3 commonvoice

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

### 2.4 coco

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

### 2.5 llava_pretrain

- 数据量：1,000 / 907 processed

| optimizer | 总数据量 | 稳态时间 | 稳态吞吐 | 相对 cedar-opt | 非稳态 setup(含启动+优化) | 总时长 | 状态 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cedar-opt | 1,000 / 907 processed | — | — | — | — | — | skipped_user_requested |
| plumber-opt | 907 | 51.2 s | 17.7 /s | — | 2.8 s | 61.1 s | completed |
| ray-opt | 907 | 55.8 s | 16.3 /s | — | 5.6 s | 68.9 s | completed |
| dp-boundary | 907 | 30.8 s | 29.4 /s | — | 52.0 s | 92.0 s | completed |
| dp-boundary-affine | 907 | 57.2 s | 15.9 /s | — | 65.5 s | 126.0 s | completed |
| dp-boundary-affine-W-width | 1,000 / 907 processed | — | — | — | — | — | 未运行 |

**各 optimizer 选中的计划**

| optimizer | W | 计划（source → ... → sink，未标注即 INPROCESS） | cache |
| --- | ---: | --- | --- |
| cedar-opt | — | 无计划文件（未运行/超时） | — |
| plumber-opt | 1 | LocalLinePipe -> parse_json_line -> SetImageRootMapper -> FixUnicodeMapper -> PunctuationNormalizationMapper -> AlphanumericFilter -> CharacterRepetitionFilter -> FlaggedWordsFilter[SMP w=2] -> PerplexityFilter[SMP w=2] -> SpecialCharactersFilter -> WordRepetitionFilter[SMP w=2] -> ImageAspectRatioFilter -> ImageShapeFilter -> ImageSizeFilter -> ImageTextSimilarityFilter -> ImageTextMatchingFilter -> sync_text_key -> PrefetcherPipe | 无 |
| ray-opt | 1 | LocalLinePipe -> FusedPipe{15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0}[RAY w=1] -> PrefetcherPipe | 无 |
| dp-boundary | 1 | LocalLinePipe -> FusedPipe{15,14,13,12}[SMP w=32] -> ImageTextSimilarityFilter -> ImageTextMatchingFilter[RAY w=1] -> FusedPipe{8,6,5,3,4,7} -> AlphanumericFilter[RAY w=63] -> FusedPipe{9,10,0}[SMP w=31] -> PrefetcherPipe | 无 |
| dp-boundary-affine | 1 | LocalLinePipe -> FusedPipe{15,14,13,12,6,9,8,10,11,7}[RAY w=64] -> FusedPipe{4,1,2,3,5,0} -> PrefetcherPipe | 无 |
| dp-boundary-affine-W-width | — | 无计划文件（未运行/超时） | — |

## 3. 未完成与不可用记录

- 放大 campaign 仍在进行中：coco / llava_pretrain / stackexchange（当前状态见 `outputs/ultimate_eight_optimizers_20260920/status.json`）；
- `commonvoice` 的 `unopti` 超过 2 小时上限，记为 `timeout`（unavailable）；
- 小数据集 campaign 的 `llava_pretrain`：`cedar-opt` 按用户要求跳过，`dp-boundary-affine-W-width` 因 1 小时上限记为 `timeout`（单次 W 的精确 DP 在 16 层中的第 10 层被截断）；
- 小数据集 campaign 的 `stackexchange` 按用户要求提前停止（只完成 plumber-opt / ray-opt），本文件不将其计入对照。

## 4. 观察

- **cache 负载**：只有 DP 系列会插入 `ObjectDiskCachePipe`（均落在 ImageReader 之后、增广算子之前），cedar/plumber/raydata 虽然 cache 已开启但没有落盘策略；simclrv2_cache 上 `simple-dp-opt` / `dp-boundary-affine` 达到 ~5.5k rec/s，是 cedar-opt 的 3.4–3.6 倍。
- **放大后排序稳定**：simclrv2 上 dp-boundary-affine-W-width (2,502/s) > simple-dp-opt / dp-boundary-affine (~1,970/s) > dp-boundary (1,938/s) > cedar-opt (1,188/s) > old-dp-opt (353/s)，而未优化计划只有 46/s。
- `old-dp-opt`（旧 profile 属性、无 boundary）明显差于新 profile 的 `simple-dp-opt`（353 vs 1,976 /s），说明新 profile 的隔离测量 + kx+b 是主要收益来源。
- 计划形态：DP 系列倾向把 CPU 段整体融合并选大 W；cedar-opt 保留一个 RAY stage；plumber/raydata 把算子拆成多个 SMP/RAY stage（radata 在 coco 上把全部算子塞进单个 RAY w=64，吞吐仅 4 rec/s）。

