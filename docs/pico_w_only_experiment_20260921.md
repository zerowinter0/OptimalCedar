# PICO 去掉 width 维度（W-only）在 llava / stackexchange 上（2026-09-21）

## 0. 动机：联合 W×width 搜索在长流水线上跑不完

PICO（`SimpleDpWorkersWidthBoundaryOptimizer`）的搜索空间是"算子顺序 × 分块 × 后端 × **width** × W"。
在 llava_pretrain（16 个可重排算子）上，带 width 的那次搜索只推进到第 10/16 层就用了 **3,055 s**、
保留了 **20.7 M** 个状态（`Exact layer 10/16 masks=462 states=20651995 max_frontier=89205
layer_sec=2160.972`），后面 6 层按同样的增长趋势没有希望在 2 小时 cell 预算内完成；
stackexchange（19 个算子）同理。因此按"**不用 width，只搜 W**"来做这组实验。

## 1. 配置（`scripts/run_pico_w_only_20260921.sh`）

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

## 2. 结果

### 2.1 llava_pretrain：完成

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

### 2.2 stackexchange：规划阶段超时

- 算子数 **n=19**；单次搜索：19 层、677 s、2,056 个 legal prefixes、324,201 retained states。
- 多次 W 候选搜索累加后触发 **7,200 s 上限**：
  `simple_dp_workers_width_boundary: setup/optimization=7200.000504s, workload skipped
  (optimizer_time_limit_exceeded)` —— **没有产出计划**。

即：去掉 width 让 llava 从"跑不完"变成 ~1 小时完成，但对 n=19 的 stackexchange 仍然不够。

## 3. 结论与下一步

- **width 维度就是这两个负载复杂度的主因**：去掉它，llava 单次搜索的状态数下降约 30×、耗时下降约
  1,000×（3,055 s → ~60 s/W）；同时它也是这两条流水线上"最贵的搜索维度"，因为 stage 宽度 × 每 worker
  的 CPU 切片共同决定了 DP 的资源状态空间。
- 对 **stackexchange** 还需要更进一步的杠杆（按代价从低到高）：固定 W 不搜（`CEDAR_DP_WORKER_SEARCH=0`）、
  `CEDAR_DP_SEARCH_MODE=chain`（放弃算子重排，只做分段/后端/宽度）、`CEDAR_DP_FRONTIER_CAP` 或
  `CEDAR_DP_PARETO_EPSILON`（有界误差的近似剪枝）、缩小 planning 用的 CPU 切片。
- 如果要真正的"不用 width"消融，需要再加一个开关，禁止 `_allocate_final_remote_stage_resources` 把并行
  stage 扩宽（当前 llava 计划里的 `SMP w=63` 就是它的产物）。

## 复现

```bash
# 容器内
bash scripts/run_pico_w_only_20260921.sh       # llava → stackexchange，日志/结果都在 RUN 目录
```

产物：`outputs/pico_w_only_20260921/{llava_pretrain,stackexchange}/{plans,logs,results}`，
launcher 日志 `outputs/pico_w_only_20260921.nohup.log`。
