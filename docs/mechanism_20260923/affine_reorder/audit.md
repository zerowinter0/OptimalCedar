# 审计：affine 迁移到重排计划之前必须核对的口径

每一项标注 **已确认 / 存在差异 / 尚不确定**，并给出代码位置与数据来源。
所有测量都在容器 `optimalcedar-torch201-dev` 内、
`limit_native_threadpools(1)` + `torch.set_num_threads(1)` 下进行。

## A. 剖析侧的测量口径

### A1. affine 的计时窗口 = 算子 callable 本身 —— 已确认

- `Dataset._profile_operator_input_size_affine`（`cedar/client/dataset.py:2623`）对每个算子
  取 `fn = pipe.fn`（或 batcher/reader 的等价 callable），用
  `_time_operator_on_snapshots` / `_time_operator_value` 计时；
  `_time_operator_fresh_snapshot`（`:2384`）把 `pickle.loads` 放在 `perf_counter` **之前**，
  只计 `fn(value)`；batcher 的批内时间再除以 `batch_size`。
- 计时在 profiling **driver** 进程内完成，`evaluation/eval_cedar.py:385-387` 已
  `limit_native_threadpools(1)` + `torch.set_num_threads(1)`。
- 结论：剖析口径 = "单个 payload、单次 callable、单线程"，不含上游拉取、组批、排队。

### A2. 自变量 x = 序列化字节数（含 pickle 开销） —— 已确认

- 拟合点 `x` 取 `len(pickle.dumps(value))`（`dataset.py:2712` 用 `median(len(s)...)`）；
- DP 侧 `_dp_affine_value(p_id, input_size)`（`cedar/compose/my_optimizer.py:1508`）
  与 `baseline.input_sizes/output_sizes` 同源，单位一致；
- 结论：**单位一致，不存在量纲不一致**；问题在"字节数是不是充分的解释变量"。

### A3. 两个拟合点来自空间缩放，而非法表示变化 —— 存在差异（本轮主因）

- 现网 profile 中 9 个算子有 7 个 `source = rescaled_legal_input`
  （`outputs/ultimate_eight_optimizers_fix_20260921/simclrv2/profiles/shared.yaml`）；
- 制造对比度的 `_affine_rescale_payload`（`dataset.py:2446`）对图像做
  **空间重采样**（`factor=0.5/2.0`），因此第二个点是"同一表示类、四分之一的像素"，
  而**不是**"同一空间尺寸、不同的 dtype/通道"；
- 于是斜率 `k` 被解读成"每字节"，实际测的是"每像素"，两者在换 dtype 时脱钩。

### A4. 剖析只在算子在声明计划里见到的那一种表示上做 —— 存在差异

- `snapshots = reservoir.values_for(predecessor_id)`（`dataset.py:2701`）：输入池 =
  该算子的前驱在**声明计划**里产出的 payload；
- 若该算子在候选计划里会被放到 `to_float` 之前/之后，它会拿到不同 dtype 的 payload，
  而剖析从未测过这一种；
- 证据：`operator_affine.operators` 里 6 个算子只有两组点，且都属于同一表示类。

### A4b. 落盘 profile 的合成点在同类 payload 上无法复现 —— 存在差异（原因尚不确定）

- profile 里 Blur 的两个点是 `(59,945 B, 3.41 ms)` 与 `(953,001 B, 48.37 ms)`，
  即 float32 的 122² 与 488² 重采样图像；
- 本轮在同一台机器、同一线程设置下用交错双 block 协议测得同类 payload：
  `f32 1ch 122²`（59,945 B）= **0.850 ms**，`f32 1ch 375×500`（750,425 B）= **7.202 ms**
  （线性外推到 953,001 B ≈ 9.6 ms）。两者相差 **4–5 倍**；
- 已排除"每次调用都重新 unpickle"这一协议差异：`fresh_vs_reuse.json` 显示
  fresh/reuse = 0.98–1.25（Blur 为 1.00 / 0.98）；
- 因此**不能**用本轮数据解释该差异；候选原因是生成 profile 那次运行的机器状态/并发，
  但未复现。**结论：现网 profile 的绝对水平不可直接用作真值，只能用其相对形状。**

### A5. 传播统计（调用数、平均输入大小） —— 已确认（对本工作负载无误差）

- 四个诊断计划 + 两个新验证计划中，逐算子"传播得到的元素数/字节数"与
  "实测统计"一致到 < 1%（`input_metadata.csv` vs `plan_scoring.json` 的 `*p` 列）；
- 这些计划没有 filter，每个记录过一次每个算子，选择性为 1；
- 结论：**本轮的迁移误差不能归因于选择率或尺寸传播**。

### A6. 极小的算子对测量口径敏感 —— 存在差异（次要）

- 例如 `F_to_float` 在 float32 输入上是 no-op（0.001–0.002 ms），在 uint8 输入上是真转换
  （0.03–0.8 ms）；`H_flip` 全量程只有 0.02–0.13 ms；
- 这些算子的绝对误差不影响计划级结论（占比 < 1%），但不能用它们的相对误差论证模型好坏。

## B. 执行侧（被预测对象）的口径

### B1. 捕获方式 = 直接在算子 callable 外层计时 —— 已确认

- 本轮新增诊断钩子 `CEDAR_OP_CAPTURE_DIR`（`cedar/client/op_capture.py`，
  在 `cedar/client/utils.py` 的 worker 内挂载）：对计划里每个 mapper 变体的 `fn`
  包一层，记录入参 `type/dtype/shape/contiguity`、序列化字节数、逐调用墙钟；
  每个管道前若干个 payload 另存 pickle 供离线回放；
- 计时窗口与 A1 相同（只包 callable），不包含上游拉取、排队与序列化；
- 钩子默认关闭，只在显式设置环境变量时启用。

### B2. 流水线内逐调用分布是重尾 —— 尚不确定（未归因）

- 例：Blur 在 PICO 计划里 p10 = 2.56 ms、中位 = 3.49 ms、均值 = 10.62 ms、p90 = 33.7 ms；
  declared 计划里 p10 = 2.59 / 均值 = 10.42；六个计划都是同一形态；
- 独立通道（`CEDAR_RECONCILE_DIR` 的 wall/process trace）给出同量级，
  因此不是捕获钩子引入的；
- **未确认**机制（疑似分配器/页错误/线程池争用）。本轮结论只建立在
  **可复现的 p10/中位**上，均值不作模型正确性的判据，并单独列为未解释上下文项。

### B3. 线程设置会主导微基准 —— 存在差异（已修正）

- 第一版回放脚本没有限制 native 线程池，Blur 因此被测成 1.53 ms，
  而同一 payload 在 1 线程下是 8.86 ms（差 7 倍）；
- 修正后（`limit_native_threadpools(1)` + `torch.set_num_threads(1)`）与剖析口径一致；
- 记录在 `tmp_analysis/replay_captured_inputs.py` 的注释与本文档，避免后人重犯。

### B4. 实测输入表示（本轮关键事实） —— 已确认

`input_metadata.csv` 给出的逐计划表示：

| 计划 | Crop | Grayscale | Jitter | Flip | Blur | to_float |
| --- | --- | --- | --- | --- | --- | --- |
| declared | f32 3ch | f32 3ch | f32 3ch | f32 3ch | **f32 1ch** | u8 3ch |
| pico | u8 3ch | u8 3ch | **u8 1ch** | u8 1ch | **u8 1ch** | u8 1ch |
| cedar | u8 1ch | u8 3ch（大图） | **u8 1ch** | u8 1ch | **u8 1ch** | u8 1ch |
| old-dp | f32 1ch | f32 3ch（大图） | **f32 1ch** | f32 1ch | **f32 3ch（大图）** | u8 3ch |
| v1 | u8 3ch | u8 3ch | **u8 3ch** | u8 3ch | **u8 1ch** | u8 1ch |
| v2 | f32 1ch | f32 3ch（大图） | **f32 1ch** | f32 1ch | **f32 1ch** | u8 3ch |

同一算子在不同计划里的表示类不同，而字节模型把它们映射到同一条直线。

## C. 合法性与语义

### C1. 计划依赖 —— 已确认

`evaluation/pipelines/target_pipeline/simclr/cedar_dataset.py` 只声明了四条约束：
reader 在最前（`fix()`）、`RandomHorizontalFlip.depends_on(["crop"])`、
`Normalize.depends_on(["float"])`、batcher 为 sink（`fix()`）。六个计划都是该 DAG
的线性延伸，都能通过 `Feature._check_physical_plan` 并真实执行（本轮全部实跑成功）。

### C2. 语义等价 —— 存在差异（必须在论文里说明）

- "合法计划" ≠ "同一数据增强语义"：`ColorJitter` 的饱和度/色相调整在 1 通道输入上是
  no-op（实测 27.9 ms → 0.44 ms），`Grayscale` 放到 `RandomResizedCrop` 之前会改变
  下采样与模糊的作用对象；
- 因此重排带来的成本差异**部分是真实的工作量差异**（通道数变化），**部分是增强语义差异**；
- 本轮不据此宣称任务精度等价；论文若要用重排收益，需要另做精度/语义对照。

### C3. 随机性控制 —— 已确认（本轮）

捕获与回放使用固定种子/固定 payload；重排实验只比较同一算子在给定 payload 上的服务时间，
不比较不同顺序的输出分布。
