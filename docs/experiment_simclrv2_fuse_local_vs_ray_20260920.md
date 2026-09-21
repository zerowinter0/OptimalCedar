# SimCLRv2：cedar-opt 的融合段 RAY → local（2026-09-20）

**问题**：cedar-opt 在 simclrv2 上把增广三件套 `[2,5,4]`（GaussianBlur + RandomHorizontalFlip +
ColorJitter）融合后放到 RAY 上执行。把这一段换成 **local（INPROCESS）**、其余完全不变（W 仍 = 64），
稳态吞吐会怎样？

**结论（单次运行）**：本地融合 **2,390.97 rec/s**，RAY 融合（同会话对照）1,150.78 rec/s —— **2.08×**。
三个 cost model 里，Cedar 认为 RAY 更便宜（方向相反），Plumber 完全区分不出这两个计划，
PICO 方向正确但把差距放大到 10.8×。

## 1. 实验设置

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

## 2. 计划记录

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

## 3. 实测稳态吞吐（W=64，9,472 条样本）

| 臂 | round1 | round2 | round3 | 中位数 | 1000/T（round1，ms/source-record） |
| --- | ---: | ---: | ---: | ---: | ---: |
| **local（`FusedPipe{2,5,4}` INPROCESS）** | **2390.97** | 2353.74 | 2314.68 | 2353.74 | **0.4182** |
| ray（`FusedPipe{2,5,4}` RAY w=1） | 1150.78 | 1182.59 | 1219.26 | 1182.59 | 0.8690 |

（单位 rec/s；每臂 3 次是脚本既定设置，用户要求“单次即可”，round1 即单次结果；另有一次 local 冒烟运行
2,335.30 rec/s。ray 臂与 campaign 里同一计划在 9,469 上的 1,150.66 rec/s 完全吻合，说明测量口径一致。）

setup（数据集构造，不含稳态）：local 2.72–2.89 s，ray 2.76–2.79 s；两边都没有优化器开销。

## 4. 三个 cost model 的预测

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

## 5. 预测 vs 实测（cost 空间，越低越好）

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

## 6. 另外四个 simclrv2 计划的三模型预测（unopt / old-dp / plumber / cedar）

口径与 §4 完全一致：**Cedar** = `Optimizer.calculate_cost`（单 worker，模型没有 W）；**Plumber** =
`1000/(X·W)`，`X` 由计划声明的宽度给出（融合 stage 的单条服务时间 = 成员 latency 之和）；**PICO** = `S/W`。
计划取 campaign 记录的原样文件 `outputs/ultimate_eight_optimizers_20260920/simclrv2/plans/`
（`round1__{unopti,old_dp_legacy_optimizer,plumber_optimizer,optimizer}.yaml`），其中 `cedar` 就是 §2 的 ray 臂。

| 计划 | W | 计划形态 | Cedar | **Plumber ÷W** | PICO S | **PICO S/W** |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| pico（PICO 自己选中的计划：`R → Fused{6,3,4,5,2,7,1} → T`，全 local） | 64 | 七算子融合、batcher 在外 | **8.9731** | 0.2987 | 8.2057 | 0.1282 |
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

## 7. 单条记录的全流程耗时拆分（cedar plan）

方法：用 `CEDAR_RECONCILE_DIR` 打开逐步 trace（每个 pipe 记录 `perf_counter_ns` 的 wall 时间戳与
`process_time_ns` 的 CPU 时间戳），跑同一份计划（W=64、batch 4、9,472 条），再把 64 个 worker 的
原始样本聚合。**每段的耗时 = 本 pipe 打点 − 上一 pipe 结束打点**，即这段在 worker 时间线上占用的时间，
包含算子本身的计算、以及算子之间的排队/序列化/远端往返。trace 开销很小：cedar 臂 1,193.9 rec/s
（无 trace 1,150.8）、local 臂 2,366.0 rec/s（无 trace 2,391.0）。

### 7.1 cedar plan（`FusedPipe{2,5,4}` 在 RAY 上，就是 campaign 选中的计划）

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

### 7.2 对照：同一计划把 `{2,5,4}` 放本地（fuse-local）

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

### 7.3 RAY 融合块的三段式拆分：序列化 / 传输 / 计算（2026-09-21 追加）

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

## 8. 指定顺序 / 指定融合的 5 个候选计划的 Cedar cost（2026-09-21 追加）

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

## 9. 追加实验：同一份计划在 W=1 与 W=64 下的 RAY / local 对照（2026-09-21）

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

## 10. 产物

- 计划：`outputs/simclrv2_local_vs_ray_9469_20260920/plans/{cedar_opt_local_w1,cedar_opt_ray_w1}.yaml`
- 结果 JSON：`outputs/simclrv2_local_vs_ray_9469_20260920/results/round{1,2,3}__{local,ray}.json`（+ `smoke_local.json`）
- 日志：`outputs/simclrv2_local_vs_ray_9469_20260920/logs/`
- §7 的逐步 trace：`outputs/simclrv2_breakdown_20260920/{reconcile_ray,reconcile_local}/worker_*.json`（各 64 个），
  汇总结果 `results/trace_{ray,local}.json`、日志 `logs/trace_{ray,local}.log`
- 复算：`python -u scripts/score_plan_cost_models.py --workload simclrv2 --profile <shared.yaml> --plan local=<plan> --plan ray=<plan>`
