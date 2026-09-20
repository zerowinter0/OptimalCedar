## 1.{no} commonvoice：三种 cost model 的估计与实测对照（Plumber/PICO 含 W 处理，Cedar 保持原样）

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
