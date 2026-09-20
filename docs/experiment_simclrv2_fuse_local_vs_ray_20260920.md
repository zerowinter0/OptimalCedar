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

读法：

- **Cedar 只在 unopt 这种“单 worker 纯 local”计划上自洽**：预测 22.98 vs 实测 21.64（差 6%，唯一可直接比
  的一行）。其余计划的实测是 W 个 worker 的系统吞吐，而 Cedar 给的是单 worker 代价，两者不可直接比较
  （§5 已展示它在 ray/local 上方向相反）。
- **Plumber ÷W** 在 plumber 自己那个“单 worker、瓶颈是本地 stage”的计划上很准（7.774 vs 9.353，1.20×），
  但凡 W>1 就系统性乐观 2.5–4.7×：它把 `X·W` 当成线性放大，没有 worker 之间的竞争/内存带宽项。
- **PICO（S/W）** 在 unopt / old-dp / cedar 上落在 0.80–0.85×（略偏悲观，方向与量级都对），但在 §5 的
  fuse-local 上偏乐观到 0.31×（3.25× 高估）——与之前在 commonvoice 上的结论一致：它对“全 local + 宽 W”
  的折扣仍偏松。

复算：`scripts/score_plan_cost_models.py --workload simclrv2 --profile <shared.yaml> --plan unopt=<plan> --plan old-dp=<plan> --plan plumber=<plan> --plan cedar=<plan>`。

## 7. 产物

- 计划：`outputs/simclrv2_local_vs_ray_9469_20260920/plans/{cedar_opt_local_w1,cedar_opt_ray_w1}.yaml`
- 结果 JSON：`outputs/simclrv2_local_vs_ray_9469_20260920/results/round{1,2,3}__{local,ray}.json`（+ `smoke_local.json`）
- 日志：`outputs/simclrv2_local_vs_ray_9469_20260920/logs/`
- 复算：`python -u scripts/score_plan_cost_models.py --workload simclrv2 --profile <shared.yaml> --plan local=<plan> --plan ray=<plan>`
