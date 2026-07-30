# 当前 `dp_optimizer` 代价模型详解

本文档基于当前工作区中的实际实现，主要涉及：

- `cedar/compose/dp_optimizer.py`
- `cedar/compose/my_optimizer.py`
- `cedar/compose/optimizer.py`
- `cedar/compose/constants.py`

`dp_optimizer` 选择器最终创建的是 `DpOptimizer`。它继承 `MyOptimizer` 的 profile 统计和单算子估价方法，但使用 `dp_optimizer.py` 中的“块候选 + 策略状态”DP 搜索计划。

需要首先区分两种 cost：

1. **DP 搜索 cost**：真正决定重排、offload、fusion 和 cache。
2. **`calculate_cost` 报告 cost**：计划生成后日志中的 `Optimized plan cost`，主要用于报告，并参与 cache 的二次校验。

二者目前并不完全一致。本文先解释真正用于选择计划的 DP cost，最后再说明报告 cost 的差异。

## 1. 基本单位与基线成本

所有 DP cost 的基本单位都是：

```text
ms / source sample
```

也就是处理一个原始 source 样本所需的估计毫秒数，而不是整个数据集的总执行时间。

基线流水线总成本来自 profile：

```text
baseline_total_cost = 1000 / baseline_throughput
```

例如 baseline throughput 为 200 samples/s，则：

```text
baseline_total_cost = 1000 / 200 = 5 ms/sample
```

默认情况下，根据各算子的 profile latency 占比分配这 5 ms：

```text
fraction_i = latency_i / sum(all_operator_latencies)

base_cost_i = fraction_i * baseline_total_cost
```

## 2. 数据量传播：size、selectivity 与 volume

DP 将两个概念分开建模。

第一，单个幸存样本的大小变化：

```text
size_ratio_i = profiled_output_size_i / profiled_input_size_i
```

第二，记录数量变化：

```text
selectivity_i = output_record_count_i / input_record_count_i
```

旧 profile 若没有 selectivity，默认取 1。

对于已经执行的算子集合 `S`：

```text
size_prod(S) = product(size_ratio_i for i in S)

cardinality_prod(S) = product(selectivity_i for i in S)

work_prod(S) = size_prod(S) * cardinality_prod(S)
```

`work_prod(S)` 表示相对于一个 source record，当前仍需处理的总字节量比例。

例如某个 filter 保留 20% 的样本，后面的 resize 又让每个幸存样本大小变为原来的 50%，则：

```text
work_prod = 0.2 * 0.5 = 0.1
```

此后算子的计算量和 stage boundary 数据量都按原始输入的 10% 估算。

这套模型假设 size ratio 和 selectivity 都是与执行顺序无关的乘法系数。对于固定输出形状、内容相关压缩率或相关过滤器，这只是近似。

## 3. INPROCESS 单算子成本

对于本地 `INPROCESS` 算子，先将 profile 成本换算为每输入字节成本：

```text
per_byte_cost_i = base_cost_i / baseline_input_size_i
```

若算子 `i` 前面已经执行集合 `S`：

```text
modeled_input_bytes = source_output_size * work_prod(S)

operator_cost_i = modeled_input_bytes * per_byte_cost_i
```

展开后为：

```text
operator_cost_i =
    source_output_size
    * work_prod(S)
    * base_cost_i
    / baseline_input_size_i
```

因此，当前模型的核心假设之一是算子 compute cost 与输入字节量线性相关。

## 4. offload 算子的计算成本

候选 backend 来自 profile 中的 `offloads`，通常包括：

- `SMP`
- `RAY`
- `TF_RAY`
- 默认的 `INPROCESS`

算子只有在支持 `can_mutate_to(variant)` 且存在对应 backend profile 时，才会生成相应候选。

### 4.1 旧 profile：通过 Amdahl 定律反推

定义：

```text
f_i = 算子 i 在 baseline 总成本中的占比

R = offload_throughput / baseline_throughput
```

根据只改变该算子 backend 后的整条流水线吞吐，反推出算子自身加速比：

```text
operator_speedup =
    f_i / (1 / R - (1 - f_i))
```

于是：

```text
offload_total_cost_i = baseline_cost_i / operator_speedup
```

这里的 `offload_total_cost_i` 来自端到端吞吐变化，不一定是纯 worker
compute。DP 将它原样作为 offload 算子 cost；第 5 节的显式 boundary 仍会
另行加入，因此端到端观测中的集成开销可能与 boundary 项有一定重叠。

### 4.2 Amdahl 反演不可识别时

若观测总加速达到或超过单算子可解释的理论上限：

```text
R >= 1 / (1 - f_i)
```

原 Cedar 会将该算子成本设为 0。当前 DP 不允许真正的零成本，而是设置有限加速上限：

```text
regularized_cost =
    max(baseline_cost_i / 64, 1e-12)
```

即单算子最多被认为加速 64 倍。即使 Amdahl 反演得到大于零但极度乐观的结果，也会应用同样的成本下限。

DP 直接使用上述 Amdahl 反推 cost。Profile 中可能存在的
`wall_latencies` 和 `backend_compute` 仅作为诊断数据，不参与计划排名，
也不会从 Amdahl cost 中减去 boundary。

## 5. stage boundary 成本

对于 `SMP`、`RAY` 和 `TF_RAY`，每个物理 stage 单独支付一次输入和输出 boundary 成本：

```text
boundary_cost =
    fixed_latency_ms / max_inflight
    + (input_bytes + output_bytes)
      / boundary_throughput_bytes_per_sec
      * 1000
```

其中：

- fixed latency 除以 profile 时的并发请求数；
- 带宽项不会除，因为它仍是每个样本实际消耗的传输服务时间；
- `TF_RAY` 使用 `RAY` boundary 模型；
- `INPROCESS` 的 boundary cost 为 0。

模型优先使用：

```text
physical_model.boundary.<variant>.throughput_bytes_per_sec
physical_model.boundary.<variant>.fixed_latency_ms
```

若 profile 没有提供，则默认：

- SMP throughput：100 MB/s；
- RAY/TF_RAY throughput：10 GB/s；
- fixed latency：0；
- max inflight：100。

若 stage 放在已执行集合 `S` 后，并覆盖 block `B`：

```text
boundary_input =
    source_output_size * work_prod(S)

boundary_output =
    source_output_size * work_prod(S union B)
```

## 6. fusion 成本

当前 `dp_optimizer` 已经不再使用旧 `MyOptimizer` 的下列模型来决定 fusion：

```text
fused_cost = compute_cost * fused_IO / baseline_IO
```

当前模型中，一个 fusion block 必须：

- 包含一个或多个满足依赖关系的算子；
- 块内所有算子使用同一个 backend；
- 多算子 block 可以物化为 `FusedPipe`；
- backend 支持当前 fusion cost 模型。

一个 block 的 operator cost 是块内算子正常估价之和：

```text
block_operator_cost =
    sum(
        当前输入工作量
        * operator_per_byte_cost
        for each operator in block order
    )
```

fusion 不会对算子 cost 打折。其收益来自：

- 未融合时，每个 parallel stage 都支付自己的 boundary；
- 融合后，整个 block 只支付一次输入 boundary 和一次输出 boundary；
- block 内部 stage boundary 被消除；
- fusion 减少 stage 数量及相应 CPU reservation。

因此当前模型可以概括为：

```text
fused_block_cost =
    undiscounted_operator_cost
    + one_stage_boundary_cost
```

而不是：

```text
fused_block_cost =
    operator_compute * IO_discount
```

### 6.1 fusion backend 限制

当 offload 搜索开启时，多算子 fusion 只考虑：

- `SMP`
- `RAY`
- `TF_RAY`

不会让未经 profile 支持的多算子 `INPROCESS` IO 折扣击败有实测依据的 parallel 候选。

当 `enable_offload=False` 时，为保留本地模式行为，允许 `INPROCESS` fusion。需要注意，当前实现关闭 offload 后仍允许 `{INPROCESS, SMP}`，并非严格只允许 `INPROCESS`。

### 6.2 SMP fusion 可行性检查

SMP fusion block 还必须满足：

- block 最后一个算子的 SMP Amdahl cost 可识别；
- 输入和输出单项大小不超过约 1 MB；
- 估计聚合输入速率不超过 SMP boundary throughput。

检查输入输出大小时，会取“重排后的模型值”和“原始 profile 值”的较大者，避免 resize 等固定输出形状算子被过度低估。

`RAY` 和 `TF_RAY` 不执行这组 SMP 限制。

## 7. 默认 DP 目标函数

假设已经完成算子集合 `S`，现在追加 block `B`。

首先计算：

```text
operator_cost = work_prod(S) * block.cost
```

其中 `block.cost` 已经是以 source size 为尺度计算的块内 operator cost。

然后：

```text
regular_cost = operator_cost + stage_boundary_cost
```

默认使用 additive objective：

```text
new_cost = previous_cost + regular_cost
```

整条计划的 DP cost 就是所有 block 的 operator cost 与 boundary cost 之和。

DP 不含 source 算子的成本，因为 source 对所有候选计划相同，不影响排名。`inner_ops` 包含最终 output 算子，因此 output 参与 backend 和 fusion 选择，但依赖约束会保证其位置合法。

## 8. cache 成本

DP 状态显式记录 `cache_active`。

允许插入 cache 的条件是 cache 前所有算子均为 deterministic，即不存在随机算子。

若在集合 `S` 完成后打开 cache：

```text
cache_read_cost =
    disk_read_time_per_byte
    * 1000
    * source_output_size
    * work_prod(S)
```

打开 cache 的转移会用 cache read cost 替换此前累计的 prefix cost：

```text
new_cost = cache_read_cost
```

cache 后面的随机或未缓存 suffix 再正常累加。

因此 cache DP 估计的是 cache-hit 或后续 epoch 的成本，没有计入：

- 第一次填充 cache 的 prefix 计算；
- cache 写入成本；
- 多 epoch 总成本；
- cache 容量限制。

若整条流水线全部 deterministic，代码只允许在最终输出处 cache。原因是当前 profile 中的内存对象大小不一定等于真实序列化 cache 大小，不足以支持仅凭该 proxy 选择更早的 cache 位置。

DP 选出 cache 后，物化阶段还会调用旧 `calculate_cost` 比较有 cache 和无 cache 的计划；若旧模型认为没有收益，cache 会被拒绝。因此 cache 最终决策实际受两套模型共同影响。

## 9. CPU 资源约束

如果设置：

```text
CEDAR_MATCH_PROFILE_RESOURCES=1
```

并提供固定 local worker 数和统一 CPU budget，则每个 local worker 可使用的 parallel-stage CPU 上限为：

```text
stage_cpu_limit =
    CPU_BUDGET // fixed_local_workers - 1
```

减去的 1 是 local worker 自身。

每个 `RAY` 或 `TF_RAY` block 消耗：

```text
ray_actors_per_stage
```

每个 `SMP` block 消耗：

```text
smp_procs_per_stage
```

fusion block 只算一个 stage，所以 fusion 也可能因为降低 CPU reservation 而成为唯一可行的计划。

这一约束只负责排除资源上不可物化的候选，不会根据 actor 或 process 数量进一步缩放 operator cost。

## 10. DP 搜索空间与状态

DP 的 mask 表示已经执行的算子集合。它只枚举满足依赖约束的 dependency-closed prefix。

一次转移可以联合决定：

- 追加一个单算子 block 或多算子 fusion block；
- block 内算子的顺序；
- block 使用的 backend；
- 是否在 block 后打开 cache。

因此重排、backend、fusion 和 cache 是联合搜索，而不是先重排、再分别做贪心优化。

状态摘要为：

```text
state =
    cache_active
    parallel_stage_cpus
```

同一 mask、cache 状态和 CPU 占用只保留最低 additive cost。

若两个候选 cost 在极小浮点容差内相同，代码倾向选择包含更多算子的 fusion block。

## 11. `Optimized plan cost` 与 DP cost 的差异

真正决定计划的是日志：

```text
[DpOptimizer] DP state cost (inner ops only)
```

计划生成后打印的：

```text
[DpOptimizer] Optimized plan cost = ...
```

来自继承的 `Optimizer.calculate_cost`。这套报告模型仍保留旧语义：

- 沿最终 critical path 累计成本；
- 使用 size ratio，但没有显式传播 filter selectivity；
- offload 使用 Amdahl 反推及 64 倍加速下限；
- fusion 使用旧的 IO 比例乘整个 compute cost；
- 没有按 DP 搜索方式显式拆分 compute 与 stage boundary；
- cache 通过删除 prefix cost，再添加磁盘读取 cost；
- 包含 source cost。

旧 fusion 报告公式为：

```text
baseline_IO =
    first_input
    + 2 * sum(intermediate_inputs)
    + final_output

fused_IO = first_input + final_output

reported_fused_cost =
    sum(operator_costs) * fused_IO / baseline_IO
```

因此 `DP state cost` 和 `Optimized plan cost` 数值不同是正常的。但差异并不只是是否包含 source 这个常数项；在 fusion、filter 和 boundary 模型下，二者可能代表不同目标，甚至可能对两个计划给出不同排名。

DP 搜索完成后，`calculate_cost` 通常不会重新改变重排、offload 和 fusion 结果。主要例外是 cache 插入时的二次收益校验。

## 12. 特殊控制流

当前联合 DP 只支持线性单路径图。如果物理图不是单路径，会退回原 Cedar 的 `Optimizer._physical_opt`。

如果 `enable_reorder=False`，代码也不会进入上述联合 DP，而是保持原顺序，仅独立为每个算子选择最低估计成本的 backend。此路径不联合考虑 fusion、cache 和显式 stage boundary。

local parallelism 在 DP 完成后由基类逻辑单独选择。除严格 profile resource matching 下的 CPU 可行性约束外，它没有直接进入上述 operator cost 目标。

## 13. 核心假设与局限

当前模型的主要假设包括：

- operator compute 与处理字节量线性相关；
- size ratio 与 selectivity 可以相乘且不随重排顺序改变；
- parallel stage transport 可由固定延迟加双向字节量除以带宽表示；
- fixed latency 可以由 max inflight 完全摊薄；
- fusion 只消除 stage boundary，不减少真实算子 compute；
- additive 目标把所有 stage 工作相加；
- cache 模型面向 cache-hit epoch，不是冷启动加多 epoch 总成本；
- local parallelism 的运行时收益没有直接进入 DP cost；
- 联合 DP 只覆盖线性单路径流水线。

## 14. 总结

默认模式下，真正用于选择计划的目标可以概括为：

```text
plan_cost =
    sum(
        按重排后字节量和记录数缩放的 operator compute cost
    )
    + sum(
        每个 SMP/RAY/TF_RAY stage 的 boundary cost
    )
```

cache 模式允许使用磁盘读取成本替换 deterministic prefix 的累计成本。

当前实现最需要明确区分的是：DP 搜索使用“Amdahl offload cost 加显式
boundary、fusion 不折扣 operator cost”的模型，而最终打印的
`Optimized plan cost` 仍使用旧 `calculate_cost` 语义。两者目前不能视为
同一个数值目标。
