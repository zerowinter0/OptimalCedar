# 实验结果图所需数据（2026-09-21）

本文件由 `scripts/collect_figure_data_20260921.py` 生成，包含三张图的数据：
1. 每个负载上各 optimizer 的**稳态吞吐量**（柱状图）；
2. 每个 optimizer 在各负载上的**优化时间**（cedar 在 llava/stackexchange 上无结果）；
3. 用 **cedar / plumber / PICO** 三个 cost model 给每个 plan 打分，并与实测吞吐排序对比（模型准确率）。

## 0. 命名映射

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

### 0.1 使用说明与注意事项

- **缺失单元**：`unopt@commonvoice` / `unopt@coco` 执行超 2 h 未产出 plan，既无吞吐也不参与打分；`cedar@llava_pretrain` / `cedar@stackexchange` 为已知的 Cedar 优化超时（`skipped_user_requested`，reason = *Known Cedar optimization timeout for this workload*）；`PICO@stackexchange` 规划超过 2 h cell 上限（`skipped_previous_timeout`），没有 plan。
- **llava 的 PICO** 只有 W-only 结果：`CEDAR_DP_WIDTH_LADDER=1` 只约束搜索候选，最终资源分配仍把 stage 扩宽到 `SMP w=63`，因此不是严格的 width=1 消融；该次 harness 以 **4 张图/记录**计数（175,760 个计数样本），文档已折算成 records/s（÷4 → 43,940 条）。
- **吞吐口径**：稳态吞吐 = 数据量 / 稳态时间，其中稳态时间取 cell 的 `perf_time_sec`（= Σ epoch_run_times）。`summary.csv` 里的 `mean_input_records_per_sec` 用的是「数据量 / workload wall」（含每个 epoch 的启动与排空），数值更高，两者不要混用。
- **cost 口径**：cedar 是单 worker 的 ms/source-record；plumber 已按 plan 的 W 折算（`1000/(瓶颈单 worker 速率 × W)`）；PICO 的 `calculate_dp_objective_cost` 是 W-conditioned 的 S，表里同时给 S 与 S/W。
- **模型覆盖率**：只有 PICO 会拒绝 plan（coco 3 个、原因见 §3.0）。被拒绝的 cell 从排序统计里剔除，**三个模型都在同一子集上比较**，避免“谁覆盖得多谁占便宜”。

## 1. 稳态吞吐量（柱状图）

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

### simclrv2

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

### simclrv2_cache

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

### commonvoice

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

### coco

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

### llava_pretrain

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

### stackexchange

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

## 2. 优化时间

单位：秒，取 cell 的 `setup_time_sec`（optimizer 规划 + 计划物化；unopti 只有构建开销）。

| 负载 | unopt | plumber | raydata | cedar | cedar-dp | PICO-Resource | PICO-Resource-Op | PICO |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| simclrv2 | 1.2 | 2.2 | 12.2 | 22.0 | 12.4 | 12.4 | 12.5 | 267.0 |
| simclrv2_cache | 1.1 | 2.2 | 12.0 | 21.7 | 12.4 | 12.9 | 23.0 | 417.9 |
| commonvoice | — *timeout* | 2.7 | 6.6 | 19.0 | 7.0 | 10.3 | 19.1 | 24.9 |
| coco | — *timeout* | 20.9 | 27.1 | 17.1 | 9.0 | 8.9 | 8.9 | 21.6 |
| llava_pretrain | 2.8 | 3.0 | 5.9 | — *skipped_user_requested* | 44.1 | 56.7 | 60.3 | 1938.5 |
| stackexchange | 2.5 | 3.9 | 9.1 | — *skipped_user_requested* | 91.0 | 87.3 | 116.1 | — *skipped_previous_timeout* |

## 3. 三个 cost model 的估计与排序

口径：cedar = `Optimizer.calculate_cost`（单 worker、ms/source-record）；plumber = plan 宽度瓶颈 + `W`（ms/source-record，1/rate）；PICO = `SimpleDpWorkersWidthBoundaryOptimizer.calculate_dp_objective_cost`（S），表中同时给出 S/W。

**模型覆盖率**（每个负载有多少 plan 能被该模型定价）：

| 负载 | 有 plan 的 cell | cedar | plumber | PICO | PICO 打不了分的 cell |
| --- | ---: | ---: | ---: | ---: | --- |
| simclrv2 | 8 | 8 | 8 | 8 | — |
| simclrv2_cache | 8 | 8 | 8 | 8 | — |
| commonvoice | 7 | 7 | 7 | 7 | — |
| coco | 7 | 7 | 7 | 4 | raydata（ValueError）、cedar（ValueError）、cedar-dp（ValueError） |
| llava_pretrain | 7 | 7 | 7 | 7 | — |
| stackexchange | 6 | 6 | 6 | 6 | — |

**汇总：模型给出的 cost 排序与实测吞吐排序的一致性**

| 负载 | 可比 cell | cedar ρ | plumber ρ | PICO ρ | cedar τ | plumber τ | PICO τ | 实测最优 | cedar 最优 | plumber 最优 | PICO 最优 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| simclrv2 | 8 | 0.4072 | 0.8264 | 0.9762 | 0.2546 | 0.691 | 0.9286 | PICO | cedar-dp | cedar | PICO |
| simclrv2_cache | 8 | 0.253 | 0.9157 | 0.994 | 0.1482 | 0.8154 | 0.982 | PICO-Resource-Op | cedar-dp | PICO-Resource-Op | PICO-Resource-Op |
| commonvoice | 7 | -0.1637 | 0.6301 | 0.955 | -0.0501 | 0.5143 | 0.8783 | PICO | PICO-Resource | cedar | PICO-Resource-Op |
| coco | 4 | 0.4 | 0.4 | 1.0 | 0.3333 | 0.3333 | 1.0 | PICO | PICO-Resource | PICO | PICO |
| llava_pretrain | 7 | 0.0901 | 0.1112 | 0.6071 | 0.0976 | 0.1029 | 0.5238 | PICO | cedar-dp | unopt | PICO |
| stackexchange | 6 | 0.4638 | 0.7247 | 0.3714 | 0.414 | 0.5521 | 0.3333 | cedar-dp | cedar-dp | plumber | raydata |

### 3.1 simclrv2

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

### 3.2 simclrv2_cache

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

### 3.3 commonvoice

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

### 3.4 coco

| optimizer | cedar cost (ms) | plumber cost (ms) | PICO score S | PICO S/W | 实测吞吐(rec/s) | 实测排名 | cedar 排名 | plumber 排名 | PICO 排名 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| unopt | — | — | — | — | — | — | — | — | — |
| plumber | 168.63 | 3.94 | 187.95 | 187.95 | 19.0 | 4.0 | 4.0 | 2.0 | 4.0 |
| raydata | 38.97 | 151.37 | — | — | 7.6 | — | — | — | — *(pico error)* |
| cedar | 22.81 | 2.79 | — | — | 26.7 | — | — | — | — *(pico error)* |
| cedar-dp | 22.81 | 4.73 | — | — | 25.8 | — | — | — | — *(pico error)* |
| PICO-Resource | 24.22 | 5.59 | 98.67 | 3.08 | 230.8 | 3.0 | 1.0 | 4.0 | 3.0 |
| PICO-Resource-Op | 46.23 | 4.73 | 71.65 | 2.24 | 241.1 | 2.0 | 3.0 | 3.0 | 2.0 |
| PICO | 27.34 | 2.79 | 79.70 | 1.25 | 294.3 | 1.0 | 2.0 | 1.0 | 1.0 |

可比 cell 4 个（三者都能定价的）；实测最优 = `PICO`；cedar 最优 = `PICO-Resource`；plumber 最优 = `PICO`；PICO 最优 = `PICO`。Spearman ρ（cost 排名 vs 吞吐排名，越接近 1 越好）：cedar 0.4、plumber 0.4、PICO 1.0；Kendall τ：cedar 0.3333、plumber 0.3333、PICO 1.0。

被排除在排序之外的 cell（有 plan 但 PICO 无法定价）：`raydata` — ValueError: The materialized block contains a non-fusable operator.；`cedar` — ValueError: Operator 1 has no RAY cost.；`cedar-dp` — ValueError: Operator 1 has no RAY cost.。

### 3.5 llava_pretrain

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

### 3.6 stackexchange

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

## 4. 复现

```bash
# 容器内
python -u scripts/collect_figure_data_20260921.py
```

JSON（每个 plan 的完整打分与计划链）：`outputs/figure_data_20260921/figure_data.json`
