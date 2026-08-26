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

此后 per-data 算子的计算量和 stage boundary 总字节量都按原始输入的 10%
估算；boundary 的单项大小和记录数仍分别保留，不能提前合并成 volume。

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

对于 `SMP`、`RAY` 和 `TF_RAY`，每个物理 stage 单独支付一次输入和输出
boundary 成本。新 profile 的首选数据不是构造字节数组的线性拟合，而是从
baseline profile 的 reservoir 中取出各逻辑边界的真实、合法对象，让 identity
算子通过与正式计划相同的一 actor/process Ray/SMP variant。测量覆盖 driver
队列、task 提交、序列化、传输和结果交付，并使用正式计划按对象大小确定的
Ray submit batch。

一次同类型 identity stage 同时包含该类型的输入、输出边界。对于输入类型为
`A`、输出类型为 `B` 的融合 block，模型使用：

```text
real_boundary_cost =
    0.5 * identity_stage_ms(A) * input_record_ratio
    + 0.5 * identity_stage_ms(B) * output_record_ratio
```

这样，融合仍只保留 block 的首、尾物理边界，内部逻辑边界不会被重复计费。
该实测服务需求进入共享 local-runtime 坐标；算子本身的 backend compute 仍进入
对应的并行 stage 坐标。

若是旧 profile，没有 real-object identity-stage 数据，才回退到下面的构造载荷
带宽/固定延迟模型：

```text
boundary_cost_per_source_record =
    fixed_latency_ms * input_record_ratio / ray_submit_batch_size
    + (input_item_bytes * input_record_ratio
       + output_item_bytes * output_record_ratio)
      / boundary_throughput_bytes_per_sec
      * 1000
```

其中：

- Ray fixed latency 是每次 actor task 的固定成本，只由同一次提交中的 batch
  样本摊销；SMP 每次提交一条样本，因此该除数为 1；
- max inflight 只是队列与背压上限，不代表并行执行能力，不参与固定成本摊销；
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
- fixed latency：0。

若 stage 放在已执行集合 `S` 后，并覆盖 block `B`：

```text
input_item_bytes =
    source_output_size * size_prod(S)

output_item_bytes =
    source_output_size * size_prod(S union B)

input_record_ratio = cardinality_prod(S)

output_record_ratio = cardinality_prod(S union B)
```

Ray 的 `submit_batch_size` 严格使用 `input_item_bytes + output_item_bytes`
计算，与最终物理计划的运行时规则相同；它不使用 `work_prod`。filter 降低的
是期望任务数和总传输量，而不是一条存活记录的大小。把 volume 当作 item
size 会在 filter 后虚构过大的 batch，并系统性低估 task latency。

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

CPU `INPROCESS` block 与 source 共用 local worker 的串行执行通道，因此其
`regular_cost` 累加到 `local_serial`。CPU `RAY`、`TF_RAY` 或 `SMP` block
的 backend compute 更新 `parallel_bottleneck`；real-object identity profile
测得的边界服务需求更新 `local_serial`。旧 profile 的合成带宽回退项仍与
backend compute 一起进入 parallel stage，保持向后兼容。

`PipeExecutionResource.CUDA` block 使用独立的 `gpu_serial` 坐标。正式实验
只有一张 GPU，因此不同 GPU stage 不能被视为具有独立吞吐能力，其服务需求
必须累加。CPU 坐标以单个 local worker 为尺度，而 GPU 是 W 个 worker 共享
的全局资源，所以 GPU compute 还要乘固定 worker 数 W：

```text
INPROCESS:
    local_serial' = local_serial + regular_cost

CPU RAY / TF_RAY / SMP:
    local_serial' = local_serial + real_boundary_cost
    parallel_bottleneck' = max(parallel_bottleneck, compute_cost)

CUDA RAY / TF_RAY:
    stage_cost = compute_cost + boundary_cost
    gpu_serial' = gpu_serial + W * stage_cost
```

CUDA block 不允许使用 `INPROCESS` 或 `SMP`，从而保证物理计划能够显式声明
GPU 资源。物理计划按所有 GPU actor 的全局数量分配 fractional GPU；例如
W=8 且只有一个单 actor GPU stage 时，每个 actor 声明 1/8 GPU，整份计划
合计恰好一张 GPU。

最终吞吐量目标为：

```text
plan_cost = max(local_serial, parallel_bottleneck, gpu_serial)
```

source 成本会进入 `local_serial`。虽然它对 additive 排名是常数，但在
`max(local_serial, parallel_bottleneck)` 中会影响 local/parallel 负载均衡，
不能删除。`inner_ops` 包含最终 output 算子，因此 output 参与 backend 和
fusion 选择，但依赖约束会保证其位置合法。

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

打开 cache 的转移会用 cache read cost 替换此前的两个 prefix 坐标；cache
读取发生在 local lane：

```text
local_serial = cache_read_cost
parallel_bottleneck = 0
gpu_serial = 0
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

同一 mask、cache 状态和 CPU 占用维护服务需求 Pareto 前沿：

```text
(local_serial, parallel_bottleneck, parallel_total_work, gpu_serial)
```

只有当一个状态在所有维度上都不大于另一个状态时才会支配后者。不能只保留
当前 `max(...)` 最小的状态：一个当前 local 较小、parallel 较大的状态，
在后续继续追加 local operator 后可能成为全局最优。

最多 8 个算子的 optimality 实验保留完整精确前沿。更大的正式负载默认
使用 `CEDAR_DP_PARETO_EPSILON=0.10` 的乘法 trimming。每个 prefix 的误差
设为：

```text
epsilon_step = (1 + epsilon_global)^(1 / n) - 1
```

因此经过最多 `n` 次转移后，各维服务需求的最坏乘法误差不超过 10%。设为
`0` 可对任意算子数量恢复精确前沿，但复杂负载的计划生成时间和内存开销会
明显上升。搜索日志同时记录 global/step epsilon、保留状态数和最大前沿。

若两个候选 cost 在极小浮点容差内相同，代码倾向选择包含更多算子的 fusion block。

## 11. `Optimized plan cost` 与 DP cost 的差异

真正决定计划的是日志：

```text
[DpOptimizer] DP objective: throughput_bottleneck
```

其中同时报告 `local_serial` 与 `parallel_bottleneck`。计划物化后，
`calculate_dp_objective_cost` 会严格重放同一组 block、backend、boundary 和
cache 转移；重放值必须与搜索值一致，否则计划生成直接失败。

继承的旧 Cedar `Optimizer.calculate_cost` 仍可能被其他 optimizer 用作报告
指标。这套旧模型保留如下语义：

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

因此旧 Cedar cost 不能用于解释 DP 的计划排名；论文中的 DP 代价准确性实验
应调用 `calculate_dp_objective_cost`。

## 12. 特殊控制流

当前联合 DP 只支持线性单路径图。如果物理图不是单路径，会退回原 Cedar 的 `Optimizer._physical_opt`。

如果 `enable_reorder=False`，代码也不会进入上述联合 DP，而是保持原顺序，仅独立为每个算子选择最低估计成本的 backend。此路径不联合考虑 fusion、cache 和显式 stage boundary。

local parallelism 在 DP 完成后由基类逻辑单独选择。除严格 profile resource matching 下的 CPU 可行性约束外，它没有直接进入上述 operator cost 目标。

## 13. 核心假设与局限

当前模型的主要假设包括：

- operator compute 与处理字节量线性相关；
- size ratio 与 selectivity 可以相乘且不随重排顺序改变；
- parallel stage transport 可由固定延迟加双向字节量除以带宽表示；
- Ray task-level fixed latency 可以由同一 task 的 submit batch 摊销；
- fusion 只消除 stage boundary，不减少真实算子 compute；
- distinct CPU parallel stages 可在稳态流水线中并发；共享同一张卡的 GPU
  stages 则共享一个串行服务容量，其需求累加；
- cache 模型面向 cache-hit epoch，不是冷启动加多 epoch 总成本；
- local parallelism 的运行时收益没有直接进入 DP cost；
- 联合 DP 只覆盖线性单路径流水线。

## 14. 总结

默认模式下，真正用于选择计划的目标可以概括为：

```text
local_serial = source_cost + sum(INPROCESS block cost)

parallel_bottleneck = max(
    每个 CPU SMP/RAY/TF_RAY block 的 compute + boundary cost
)

gpu_serial = W * sum(每个 CUDA block 的 compute + boundary cost)

plan_cost = max(local_serial, parallel_bottleneck, gpu_serial)
```

cache 模式允许使用磁盘读取成本替换 deterministic prefix 的累计成本。

搜索通过多维 Pareto 前沿保持该目标的精确最优子结构。六算子穷举验证会
枚举所有合法重排、fusion 分区和 backend 分配，并检查 DP 与独立 oracle
的全局最优值完全一致。
