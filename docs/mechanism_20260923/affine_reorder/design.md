# 设计：把"元素数 + 表示类"接进 profiler 与 DP（未实现，待批准）

本文件只写依据、改法与验收标准，**不改现网模型**。当前正式模型
（`physical_model.operator_affine` 的字节 `kx+b`）保持不变，改进作为**独立变体**实现。

## 1. 依据（来自本轮证据）

1. `blur_geometry.json`：同样约 6 万字节，`u8 1ch 244×244` = 2.606 ms 而
   `f32 1ch 122×122` = 0.850 ms（3.1×）；元素数相同而 dtype 不同时成本几乎相同。
2. `operator_matrix.json`：在 4 个表示类 × 3 个空间尺度上，`k·元素数 + b` 的
   留出点误差多数 < 8%（Jitter-3ch +30%、极小算子 +23% 是主要例外）。
3. `plan_scoring.json`：以流水线内 p10 为目标，六个计划上
   M4（元素 + 表示类）= 0.92–1.42×，M2（字节 affine）= 0.54–11.13×。
4. `audit.md` A5：这些计划的**统计传播没有误差**，所以改进只需动"成本响应"，
   不需要改选择率/尺寸传播。

## 2. 模型变更

```
compute_price(op, payload) = k_(op, class(payload)) * elements(payload) + b_(op, class(payload))
boundary_price(...)        = 保持字节口径不变（序列化/传输确实与字节相关）
```

- `elements(payload)`：tensor → `numel`；PIL → `W×H×bands`；bytes/str → `len`；
  dict/list → 递归求和。**不引入新维度**，只是把既有 payload 的"工作量"度量换掉。
- `class(payload)`：`{dtype}×{channels}`（tensor）、`PIL{mode}`、`text`、`path` 等，
  由 payload 自身决定，**是规划时可获得的信息**。
- 每个 (算子, 类) 仍是两参数仿射：`k`、`b`。总参数数从 `2×N_op` 增至
  `2×Σ_op(#classes_op)`；本负载下 7 个算子共 4 类、实际可用类 2–4 个，
  即每个算子 4–8 个额外参数。

## 3. profiler 变更（`cedar/client/dataset.py`）

1. 新增 `payload_elements(value)` 与 `payload_class(value)`（放在
   `cedar/pipes/common.py`，供 profiler 与 DP 共用同一实现，避免两处分叉）。
2. `_profile_operator_input_size_affine`：
   - 保留现有的"自己的合法输入"两点拟合 → 写入新的
     `k_ms_per_element` / `b_ms` / `x_reference_elements`（与旧字节字段**并存**，
     旧字段继续给现网模型读）；
   - 新增 `by_class`：对 reservoir 里**本特征其它管道**产出的 payload（同一批记录、
     同一份内容），逐个用一次试调判断该算子是否接受，接受的按类分组；
     每类用"最小/最大两个层"拟合 k、b（与本轮分析脚本同一规则）；
   - 新增 `class_transition[p][in_class] = out_class` 与 `element_ratio[p]`
     （对每类各测一次即可）。
3. 额外剖析成本：每算子每类 2 次计时（每类 5 次 repeat，`target_sec` 不变），
   本负载下约增加 8–16 次小测；相对整份 layered profile 可忽略，
   但必须在 README 里报告（本轮 `protocol.json` 已要求）。
4. 未测到的类**不允许回退**：如果候选计划要求一个没有系数的 (算子, 类)，
   优化器应报错而不是退回字节模型（与 AGENTS.md 的"强制新版本正确"一致）。

## 4. DP 变更（`cedar/compose/my_optimizer.py`、`cedar/compose/dp_optimizer.py`）

现在只有一张表：`_dp_r_prod[mask] = ∏_{i∈mask} byte_ratio[i]`。
需要新增两张与它同构的表（同样的子集递推、同样的 lazy 路径
`_LazyProductTable`，60+ 算子的负载同样适用）：

```
_dp_element_prod[mask] = ∏ element_ratio[i]
_dp_class_state[mask]  = class_transition[last_op]( class_state[mask ^ lsb] )
```

- `class_state` 只需存一个小的离散 id（类数 ≤ 8），`2^n` 个 int8 足够；
  超过子集上限时用与 `_LazyProductTable` 相同的惰性策略。
- `_dp_compute_work_prod` 改为
  `cardinality_prod[mask] * price(op, element_prod[mask], class_state[mask])`；
  `_dp_work_prod`（字节）**保持不变**，继续给 boundary / cache / transport 用。
- `_calculate_pipe_cost` 的本地分支与后端分支都改用元素价格；
  后端测得的 `backend_compute.mean_ms_per_sample` 仍在**它被剖析的那个类**上，
  换类时按该类系数缩放（本轮 M4 的做法）。

## 5. 变体与消融接线

| selector | 模型 | 用途 |
| --- | --- | --- |
| 现网 21/27/29 | 字节 `kx+b` | 不变，正式基线与历史结果 |
| 新 34 `SimpleDpBoundaryAffineElements` | 元素 `kx+b` | M3 消融（只换自变量） |
| 新 35 `SimpleDpBoundaryAffineRepr` | 元素 + 每类系数 | M4（推荐候选） |

两个新变体都复用同一份 layered profile（旧的字节字段仍在里面），
保证"模型形式/额外剖析/统计修复"三者可分离。

## 6. 验收标准（事先固定）

1. **回归**：在同一份 profile 上，变体 34/35 对**声明计划**的预测必须与实测
   （逐调用 p10）相差 < 20%——否则说明接线引入新错误。
2. **计划级**：六个计划（§4.10.4 的四个 + v1/v2）上的预测倍率必须落在
   分析脚本给出的区间附近（M4：0.92–1.42），并对**新构造**的两个顺序保持。
3. **决策级**：对六个计划，比较"模型选择的计划"与"实测最优计划"的后悔值；
   运行间不确定（v1 2.6%、v2 6.5%）以内的差异标记为并列。
4. **不退化**：dp-boundary-affine（字节版）与 PICO 的行为与当前一致
   （同 profile、同数据、同 W 下的计划与 cost 不变）。
5. 若变体 34（只有元素、没有类）在多数计划上并不优于字节版，
   则必须在论文中说明"改进来自**表示类**这一额外维度"，而不是自称仍是两参数 affine。

## 7. 已知风险

- 类的数量会随负载增长（多模态、文本、视频），`class_transition` 的表会变大；
  需要上限与显式的"未覆盖类"错误，而不是静默回退。
- 元素数对文本类负载没有意义（`len(bytes)` 就是元素数），因此该改动对文本负载
  等价于"字节但去掉 pickle 开销"，需要单独验证不产生回归。
- 本轮只证明成本响应改善；**是否带来更好的计划选择与吞吐仍未验证**，
  这正是验收标准 2/3 要回答的问题。
