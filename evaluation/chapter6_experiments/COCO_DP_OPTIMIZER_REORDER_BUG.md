# COCO `dp_optimizer` 融合块组内重排问题

## 1. 任务目标

修复 `dp_optimizer` / `my_optimizer` 在 COCO 工作负载上的融合块组内重排与代价估计问题。

当前 `dp_optimizer` 生成的核心计划为：

```text
Ray Fused Stage 1 (16 actors):
zoom → crop → sanitize → flip → distort

Ray Stage 2 (16 actors):
to_tensor
```

实测该计划明显慢于原始 `optimizer`。主要性能问题不是 fusion materialize 改乱了顺序，而是组内 DP 本身返回了将 `distort` 放在末尾的顺序。

本修复应以正式顶会实验的正确性和公平性为标准，不能通过降低数据量、减少并行度或特判 COCO 来规避问题。

## 2. 项目与运行约束

- 主机仓库：`/home/xieruiyang/OptimalCedar`
- Docker 容器：`optimalcedar-torch201-dev`
- 容器仓库：`/workspace/OptimalCedar`
- 所有代码运行和修改前：

  ```bash
  docker exec optimalcedar-torch201-dev bash
  cd /workspace/OptimalCedar
  source env/bin/activate
  ```

- `dp_optimizer.py` 和 `my_optimizer.py` 逻辑应一致；不一致处以 `my_optimizer.py` 为准。
- 工作树中已有其他用户修改，不得覆盖或清理无关改动。
- Ray 正式并行度语义：每个活跃 Ray 算子各有 16 个 actors；其他算子并行度为 16。

## 3. 相关文件

- 组内重排实现：`cedar/compose/my_optimizer.py`
  - `_dp_naive_reorder_cost_per_variant`：约第 558 行起
  - Pareto frontier 裁剪：约第 623–636 行
  - 最终 fusion cost：约第 701–732 行
  - `_reconstruct_naive_reorder_order`：约第 740 行起
- DP block candidate：`cedar/compose/dp_optimizer.py`
  - `BlockCandidateProvider.prepare`：约第 80–171 行
- 单算子 physical variant 成本：`cedar/compose/optimizer.py`
  - `_calculate_pipe_cost`：约第 1716–1767 行
- COCO pipeline：`evaluation/pipelines/coco/cedar_dataset.py`
  - `_compose`：约第 53–64 行
- 原始 COCO profile：
  - `evaluation/chapter6_experiments/formal_results/profiles/coco.yaml`
- 两种计划的真实 COCO profiling 结果：
  - `evaluation/chapter6_experiments/formal_results/raw/coco_optimizer_profile_result.json`
  - `evaluation/chapter6_experiments/formal_results/raw/coco_dp_optimizer_profile_result.json`
  - `evaluation/chapter6_experiments/formal_results/raw/coco_optimizer_process_profile.json`
  - `evaluation/chapter6_experiments/formal_results/raw/coco_dp_optimizer_process_profile.json`
- 对应日志：
  - `evaluation/chapter6_experiments/coco_optimizer_process_profile.log`
  - `evaluation/chapter6_experiments/coco_dp_optimizer_process_profile.log`

## 4. COCO 算子、ID 与约束

```text
pipe 5: zoom_out
pipe 4: crop
pipe 3: SanitizeBoundingBox
pipe 2: RandomHorizontalFlip
pipe 1: distort
pipe 0: to_tensor
pipe 6: source
```

代码中只有以下显式依赖：

```text
zoom → crop → sanitize → flip
```

`distort` 没有依赖约束，可以放在上述链条的任意位置。原始 `optimizer` 能将其移动到 `zoom` 前。因此不要通过添加固定依赖来掩盖 optimizer bug。

## 5. 已确认的精确复现结果

使用现有 `coco.yaml`，不重新执行数据，仅调用现有组内 DP：

- mask 固定为五算子 `{5, 4, 3, 2, 1}`，在完整 inner slot 中为 `mask=31`
- variant 固定为 `PipeVariantType.RAY`
- 只观察该 block 的组内 reorder/backtracking

现有实现精确返回：

```text
INNER_OPS [5, 4, 3, 2, 1, 0]
ORDER_INDICES [0, 1, 2, 3, 4]
ORDER_PIPE_IDS [5, 4, 3, 2, 1]
ORDER_NAMES [zoom_out, crop, sanitize, flip, distort]
MODELED_BLOCK_COST_NORMALIZED 3.7630654917755432e-06
RAY_COSTS_BY_INNER_SLOT [13.374169162396104, 4.829857213725129,
                         7.3406305702400045, 0, 0, 0]
DEPENDENCY_PREDECESSORS_BY_SLOT [[], [0], [1], [2], [], []]
```

这证明错误顺序由组内 DP 产生，而不是 `_fuse_pipe` 或 physical plan materialization 产生。

## 6. 根因一：Pareto frontier 的支配方向错误

`my_optimizer.py::_prune_frontier` 当前把 `(cost, io_base_partial)` 两个维度都当作越小越好：

```python
states_sorted = sorted(states, key=lambda x: (x[0], x[1]))
best_io = float("inf")
for st in states_sorted:
    c, io_b, pl, pi = st
    if io_b < best_io:
        pruned.append((c, io_b, pl, pi))
        best_io = io_b
```

但后面的最终目标是：

```python
baseline_io = io_base_partial + r_prod[mask]
io_ratio = fused_io / baseline_io
final_cost = cost * io_ratio
```

在这个公式下：

- `cost` 越小越好；
- `io_base_partial` / `baseline_io` 越大，`io_ratio` 越小，最终 cost 反而越低。

因此就当前目标函数而言，一个状态应由“更低 compute cost、同时更高 baseline I/O”的状态支配，而不是由两个值都更小的状态支配。

对五个合法位置穷举当前目标函数，结果为：

| 排名 | 顺序 | 当前模型 cost |
|---:|---|---:|
| 1 | zoom → distort → crop → sanitize → flip | 2.746694988 |
| 2 | zoom → crop → distort → sanitize → flip | 3.051134094 |
| 3 | zoom → crop → sanitize → distort → flip | 3.051137670 |
| 4 | zoom → crop → sanitize → flip → distort | 3.051137670 |
| 5 | distort → zoom → crop → sanitize → flip | 3.572641043 |

即使完全接受当前 cost model，DP 当前返回的第四名也不是最优。

检查 `mask=31` 的实际 frontier，裁剪后只剩：

```text
distort → zoom → crop → sanitize → flip
zoom → crop → sanitize → flip → distort
```

真正的当前模型最优解 `zoom → distort → crop → sanitize → flip` 已被错误裁掉。

### 对根因一的修复要求

- 修正 Pareto 支配方向。
- 注意相同/近似相同浮点 compute cost 的稳定处理；不能依赖偶然的浮点加法顺序。
- 用穷举 oracle 校验小规模 block 的 DP 最优值与回溯顺序。
- 不能只硬编码预期 COCO 顺序。

仅修复这一项后，按当前错误模型，预期会得到：

```text
zoom → distort → crop → sanitize → flip
```

但这仍不是实际性能合理的最终顺序，因为还存在根因二。

## 7. 根因二：Ray cost 将关键算子估成零，且错误折扣全部计算成本

`optimizer.py::_calculate_pipe_cost` 中，当 offload throughput 超过 Amdahl 理论上限时：

```python
if total_speedup >= 1 / (1 - self._fractional_latencies[p_id]):
    cost = 0
```

COCO 强制 Ray 时得到：

```text
zoom:     13.3742
crop:      4.8299
sanitize:  7.3406
flip:      0
distort:   0
```

因此 DP 完全看不到 `distort` 的高计算代价。`distort` 和 `flip` 的位置主要由有问题的 I/O 比例驱动。

此外，`_dp_naive_reorder_cost_per_variant` 最终执行：

```python
final_cost = compute_cost * (fused_io / baseline_io)
```

这相当于用 fusion 消除的中间序列化/I/O 比例去折扣所有算子计算时间。fusion 可以减少 boundary/serialization cost，但不会按相同比例减少 `distort`、`crop` 等函数本身的计算量。

如果使用原 profile 的算子计算延迟，仅按输入大小线性缩放，并且不使用 fusion I/O 比例折扣计算成本，五个合法位置的预测为：

| 顺序 | 计算成本 |
|---|---:|
| distort → zoom → crop → sanitize → flip | **45.176355** |
| zoom → crop → sanitize → distort → flip | 120.574468 |
| zoom → crop → sanitize → flip → distort | 120.574468 |
| zoom → crop → distort → sanitize → flip | 120.575074 |
| zoom → distort → crop → sanitize → flip | 177.826097 |

原因很直接：

- source 输出约 `0.811 MB/sample`；
- zoom 输出约 `5.293 MB/sample`；
- crop 后约 `3.358 MB/sample`；
- `distort` 是最耗时算子；
- 将 `distort` 放在 zoom 前可让它处理约小 4.1 倍的数据。

### 对根因二的修复要求

建议把 block/stage 代价至少拆成：

```text
total_cost = operator_compute_cost
           + stage_boundary_serialization_cost
           + Ray scheduling/queue cost
```

要求：

- 不得用 `io_ratio` 乘整个 operator compute cost。
- Amdahl 反演异常时不得令关键算子成本为零。
- 对 fused Ray block，内部算子的计算成本应由可靠的 per-op latency/profile 建模；Ray boundary 只在 block 输入/输出处计费。
- 如果缺少足够 profile，应采用保守、可解释的 fallback，而不是零成本。
- 修改 `my_optimizer.py` 后同步检查 `dp_optimizer.py` 的调用语义。

## 8. 真实执行证据

在同一当前 COCO 数据、同一 profile、每个活跃 Ray stage 各 16 actors 的 profile 对比中：

```text
optimizer:
  4952 samples
  workload wall: 41.6804 s
  throughput: 162.747 samples/s

dp_optimizer:
  4952 samples
  workload wall: 67.5274 s
  throughput: 95.983 samples/s
```

即：

```text
dp_optimizer time = 1.695× optimizer
dp_optimizer throughput = -41.0%
```

原始 `optimizer` 的顺序为：

```text
distort → zoom → crop → sanitize → flip → to_tensor
```

`dp_optimizer` 的顺序为：

```text
zoom → crop → sanitize → flip → distort | to_tensor
```

竖线表示额外 Ray stage 边界。

## 9. 系统公平性背景：不要与本 bug 混淆

已有 profiling 还发现 Ray actors 使用 `@ray.remote(num_cpus=0)`，当前实验环境没有对整个 Ray head 强制同一 16-CPU cpuset，因此严格固定硬件公平性仍需另行修复。

但是这不能解释或否定本组内 reorder bug：

- 每个活跃 Ray stage 的 actor 数量已经验证；
- 五算子固定 Ray、仅使用已有 profile 的组内 DP 复现无需运行 Ray；
- 错误顺序可以直接由 frontier 和 cost model 代码确定。

不要用减少第二个 stage actor 数量的方式“修复”此次 optimizer 顺序问题。

## 10. 必需测试

至少增加以下测试：

1. **DP 与穷举一致性测试**
   - 生成 4–7 个算子的小型有约束 DAG。
   - 对每个 variant 和 mask 穷举所有合法拓扑序。
   - 比较 DP 最优 cost 与回溯 order。
   - 覆盖 ratio `<1`、`=1`、`>1` 和零/近零 cost。

2. **Pareto 支配方向测试**
   - 构造两个 compute cost 相同但 `io_base_partial` 不同的状态。
   - 确认不会裁掉最终目标更优的状态。

3. **COCO 五算子回归测试**
   - 使用现有 `coco.yaml`。
   - mask `{5,4,3,2,1}`，固定 Ray。
   - 结果必须与同一修复后 cost model 的穷举 oracle 一致。
   - 若采用计算/边界分离模型，预期 `distort` 应位于 zoom 前。

4. **异常 offload profile 测试**
   - throughput 超过 Amdahl 可反演范围时，不得返回零 compute cost。
   - 输出必须有限、非负、保守且有日志说明 fallback。

5. **全计划回归测试**
   - 构造 COCO plan-only，确认 fusion block 的 `fused_pipes` 顺序与 DP 回溯一致。
   - 确认修改没有破坏依赖：`zoom → crop → sanitize → flip`。

6. **真实 COCO 验证**
   - 修复单元测试后再运行真实 COCO。
   - 每个 Ray stage 各 16 actors；其他算子并行度 16。
   - 至少记录最终 physical plan、model cost、wall time、throughput。
   - 正式结论应做多次重复和置信区间，不以单次运行作为论文结果。

## 11. 验收标准

- 组内 DP 对测试小图与穷举 oracle 完全一致。
- 不再出现耗时关键算子的 Ray compute cost 被无依据设为零。
- compute cost 与 stage boundary/serialization cost 分离。
- COCO 五算子 fixed-Ray 的回溯顺序与修复后的模型最优解一致。
- COCO 最终 physical plan 保持所有语义依赖。
- 真实 COCO 性能不再因把 `distort` 放在大输入之后而显著退化。
- 不引入 COCO 专用条件分支，不牺牲其他 workload 的公平性或正确性。

## 12. 建议的修复顺序

1. 先写穷举 oracle 测试，复现当前 frontier 错误。
2. 修复 Pareto 支配方向与浮点稳定性，使现有目标下 DP 与穷举一致。
3. 将融合代价重构为 compute 与 boundary cost 相加。
4. 移除/替换 Amdahl 异常时的 `cost = 0`。
5. 跑所有 optimizer 单元测试和 COCO plan-only 回归。
6. 最后按正式并行度运行真实 COCO 重复实验。

