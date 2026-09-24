# 设计：把"表示感知的计算模型"接进 profiler 与 DP（已实现）

上一轮的 `design.md` 是方案；本轮按该方案完成实现，本文档记录**实现后的**接口、
公式与状态，以及与原两参数 affine 的区别。

## 1. 特征与公式

对算子 `i`、位置 `p`：

```
elements(p) = source_elements * Π_{j before p} element_ratio[j]        # 计算规模
class(p)    = transition[last]( ... transition[first](source_class) )  # 表示类
compute_i(p) = k_(i, class(p)) * elements(p) + b_(i, class(p))          # 单位: ms / 源记录
bytes(p)     = source_bytes * Π_{j before p} byte_ratio[j]              # 只用于边界/传输
```

- 表示类由 payload 自身决定（`cedar/pipes/common.py:payload_representation_class`）：
  `uint8:3ch`、`float32:1ch`、`PIL:L`、`path`、`text` 等；规划时即可获得。
- 计算规模由 `payload_compute_scale` 给出：图像 `C*H*W`、PIL `W*H*bands`、
  文本/路径 `len`、容器递归求和。
- 传输/边界**完全不变**：仍用序列化字节量与既有的 boundary 模型。

## 2. profiler 实现

`Dataset._profile_operator_compute_model`（`cedar/client/dataset.py`）：

1. 候选 payload = 该次 profiling 流水线真实产出的中间值（reservoir）
   ∪ 把这些值再经过本 feature 任一 callable 一次应用的结果。
   后者用于覆盖"声明顺序从不物化、但合法重排会物化"的表示类
   （例如 uint8 单通道：`Grayscale(uint8:3ch)`）。
2. 对每个算子，用一次试调筛出它接受的 payload，按表示类分组；
   每类取中位大小的 payload，做 0.5× / 2.0× **空间**缩放得到两个层
   （只有空间缩放，绝不跨类套用）。
3. 同一算子的所有 (类 × 层) 点在**同一个交错窗口**内测量，每点累计
   `CEDAR_PROFILE_COMPUTE_TARGET_SEC`（默认 5 s）的实际调用时间，取逐调用平均。
   这是本轮最重要的协议修正：更短的窗口会漏掉 GaussianBlur 的慢模式，
   把均值低估 3–6 倍（见 `audit.md` B2 与 `blur_mean_curve.json`）。
4. 两点线性拟合 `k`、`b`；没有尺寸对比的 payload（文件路径）按 `k = 0` 的同一族常数处理。
5. 同时记录 `class_transition[p][class]` 与 `element_ratio[p]`（用该算子在声明位置的真实输入测得）。

落盘位置：`physical_model.compute_model`（schema_version=1），与旧的
`physical_model.operator_affine` **并存**，旧模型与其历史结果不受影响。

## 3. DP 实现

`cedar/compose/simple_dp_ablation_optimizer.py:_RepresentationComputeMixin`：

- 与 `_dp_r_prod` 同构地构建两张表：
  `element_prod[mask] = Π element_ratio`、`class_state[mask]`（转移沿 mask 逐位应用）。
  两张表都只依赖"已放入前缀的算子集合"，因为本负载的转移是单调的
  （`to_float` 设置元素类型、`grayscale` 减少通道）；`repr_state_is_order_independent()`
  会对整个算子集做穷举一致性检查，一旦未来 recipe 破坏该性质会报错而不是错误合并状态。
- `_dp_compute_work_prod(mask, idx)` 用 `elements(mask)` 与 `class(mask)` 定价；
  `_dp_work_prod`（字节）保持不变，boundary/cache/transport 仍走字节。
- `_dp_compute_cost_denominator`、`_calculate_pipe_cost` 用算子在**声明位置**的特征做锚点，
  保证"profile 位置处预测 = profile 值"，与旧实现的归一化口径一致。
- 缺少 `(算子, 表示类)` 曲线时**显式报错**，不会静默回退到字节模型
  （`RuntimeError: operator X has no compute curve for representation class Y`）。

## 4. 变体

| selector | 类 | 模型 |
| --- | --- | --- |
| 21 / 27 / 29（现状） | `SimpleDpBoundaryOptimizer` 等 | 字节 `k·bytes+b`（基线，未改） |
| 34 | `SimpleDpBoundaryAffineElementsOptimizer` | M3：元素 affine，不分类 |
| 35 | `SimpleDpBoundaryAffineReprProportionalOptimizer` | M4：分类过原点 `k_(i,z)·e` |
| 36 | `SimpleDpBoundaryAffineReprOptimizer` | M5：分类 affine `k_(i,z)·e+b_(i,z)` |
| 37 | `SimpleDpWorkersWidthBoundaryAffineReprOptimizer` | PICO（W×width）配 M5 |

CLI 名（`evaluation/compare_optimizer_perf.py`）：
`simple_dp_repr_elements` / `simple_dp_repr_proportional` /
`simple_dp_repr_affine` / `simple_dp_workers_width_repr_affine`。

## 5. 与原两参数 affine 的差别（用于论文表述）

| 维度 | 旧模型 | 新模型 |
| --- | --- | --- |
| 自变量 | 序列化字节 | 元素数（计算规模） |
| 系数量 | 每算子 1 对 (k,b) | 每 (算子, 表示类) 1 对 (k,b) |
| 重排影响 | 字节量变化即认为工作量变化 | 只有元素数/表示类变化才影响计算；字节只影响边界 |
| 覆盖 | 任何算子 | 只覆盖**测过的**表示类，未覆盖即报错 |

## 6. 验收（已执行/待执行）

- DP vs 穷举：`tmp_analysis/test_repr_dp_optimality.py`（关 fusion/offload/parallelism，
  枚举全部合法顺序，用独立参考实现打分，DP 必须落在 argmin）。
- 缺失类行为：`test_compute_model_profiler.py` + DP 报错路径。
- 成本回放一致性：`tmp_analysis/score_m1_m5_plans.py` 在同一 profile 上重放逐算子预测。
- 字节边界未变：新变体的 `_dp_work_prod` 未被覆盖，边界项仍由
  `_layered_boundary_cost_ms` 计算（与 21/27 共用）。
