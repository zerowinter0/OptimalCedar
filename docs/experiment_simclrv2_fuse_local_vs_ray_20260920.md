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

## 6. 产物

- 计划：`outputs/simclrv2_local_vs_ray_9469_20260920/plans/{cedar_opt_local_w1,cedar_opt_ray_w1}.yaml`
- 结果 JSON：`outputs/simclrv2_local_vs_ray_9469_20260920/results/round{1,2,3}__{local,ray}.json`（+ `smoke_local.json`）
- 日志：`outputs/simclrv2_local_vs_ray_9469_20260920/logs/`
- 复算：`python -u scripts/score_plan_cost_models.py --workload simclrv2 --profile <shared.yaml> --plan local=<plan> --plan ray=<plan>`
