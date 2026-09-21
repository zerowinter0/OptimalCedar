# 为什么需要 affine（kx+b）：把 unopt 上测到的每算子代价迁移到新 plan（2026-09-21）

**问题**：Cedar 的 cost model 用"单点测量 + 按输入字节等比例外推"（`y = x`）来估计一个算子在
**新 plan** 上的开销。这个估计能迁移吗？

**做法**：四个 pipeline，**同一批 9 个算子、同一执行方式**（全 INPROCESS、无融合、无 offload、W=4、
batch 4、9,469 条），**只有算子顺序不同**——顺序决定了每个算子看到的输入尺寸：

| cell | 顺序（reader → … → batcher） | 说明 | 稳态吞吐 |
| --- | --- | --- | ---: |
| `declared` | F C H J G B N | unoptimized/声明顺序（= Cedar profile 的对象） | 170.3 rec/s |
| `pico` | C G J H B F N | PICO plan 的算子顺序 | 279.2 rec/s |
| `cedar` | G C B H J F N | cedar-opt plan 的算子顺序 | 273.7 rec/s |
| `old-dp` | F N B G J C H | old-dp 融合块的顺序 | 64.0 rec/s |

*仅顺序不同*，吞吐就差 **4.4×**（64.0 ↔ 279.2 rec/s）。每算子数据用 `CEDAR_RECONCILE_DIR` 采集：
输入字节/记录 与 该算子段的 **process-time（CPU）每记录 ms**（用 CPU 时间而非 wall 分段，避免把排队算进
算子开销；wall 分段会包含等待）。

## 1. 每算子的输入尺寸与实测开销

| 算子 | x declared (B) | x pico (B) | t declared (ms) | t pico (ms) |
| --- | ---: | ---: | ---: | ---: |
| T Batcher | 238,144 | 238,144 | 7.942 | 5.079 |
| N Normalize | 238,144 | 238,144 | 0.236 | 0.170 |
| B Blur | 238,144 | **59,536** | 8.387 | 9.150 |
| G Grayscale | 714,432 | **178,608** | 0.350 | 0.285 |
| J Jitter | 714,432 | **59,536** | 8.497 | 0.496 |
| H Flip | 714,432 | **59,536** | 0.129 | 0.081 |
| C Crop | 2,507,430 | **748,943** | 1.986 | 2.110 |
| F to_float | 626,857 | **59,536** | 0.452 | 0.063 |
| R ImageReader | 157 | 157 | 2.372 | 2.571 |

## 2. 把 declared（unopt）profile 迁移到 pico 顺序

两个模型都**锚定同一个 declared 实测点**（只看模型形式差异）：

- **cedar**：`t = t_declared × (x / x_declared)`（过原点等比例，Cedar 的做法）
- **affine**：`t = t_declared × (k·x + b) / (k·x_declared + b)`，k、b 取 profile 的
  `physical_model.operator_affine`（profiler 在多尺寸下拟合）

| 算子 | cedar 估计 | 误差 | affine 估计 | 误差 |
| --- | ---: | ---: | ---: | ---: |
| T Batcher | 7.942 | +56.4% | 7.942 | +56.4% |
| N Normalize | 0.236 | +38.5% | 0.236 | +38.5% |
| B Blur | 2.097 | **−77.1%** | 2.297 | −74.9% |
| G Grayscale | 0.088 | **−69.3%** | 0.153 | −46.2% |
| J Jitter | 0.708 | +42.8% | 1.064 | +114.5% |
| H Flip | 0.011 | **−86.7%** | 0.052 | −35.3% |
| C Crop | 0.593 | **−71.9%** | 1.693 | −19.8% |
| F to_float | 0.043 | −31.5% | 0.043 | −31.5% |
| R ImageReader | 2.372 | −7.7% | 2.372 | −7.7% |

**均值 |误差|：cedar 等比例 53.5%（中位 56.4%，最大 86.7%）；affine 47.2%（中位 38.5%）**。
只看**输入尺寸发生变化的那 6 个 mapper**：cedar ≈ **63%**、affine ≈ **54%**，且 affine 在
Crop（−72% → −20%）、Flip（−87% → −35%）、Grayscale（−69% → −46%）、Blur（−77% → −75%）都更好。

同样四个 in-pipeline 点做 **留一顺序**预测（k、b 只由另外三个顺序拟合）：cedar 等比例 均值 84.0% /
中位 62.6%，affine 均值 67.6% / 中位 46.4% —— affine 同样系统性更好。

## 3. 结论与残留误差

1. **Cedar 的每算子代价无法迁移到新 plan**：仅改顺序（尺寸随之变化）后，Blur/Grayscale/Flip/Crop 的
   估计误差达 **−69% ~ −87%**，而且是**系统性低估**——因为它假设 `cost ∝ 输入字节`（过原点）。
2. **漏掉的正是 affine 的 b（固定分量）**：profile 自己的 affine 拟合显示 9 个算子里有 6 个固定分量占比
   25%–100%（crop 79.8%、normalize 47.8%、flip 35.1%、grayscale 25.1%、jitter 4.6%、blur 3.2%）；
   把 b 加回来，同样一个锚点下 Crop/Flip 的误差就从 −70% 级降到 −20%/−35%。
3. **仍有残差，且原因明确**（不是 affine 形式的错）：
   - **外推越界**：Jitter 在 pico 顺序里只拿到 59,536 B，低于 profiler 实测过的下界（179,026 B），
     于是 kx+b 外推反而高估 115%（Cedar 的等比例在这点恰好更接近）；
   - **同一尺寸、不同 plan 上下文**：Normalize / ImageReader 在两个顺序里输入尺寸完全相同，但实测 CPU
     时间差 1.1–1.8×（上游节奏、缓存/频率状态），任何只以"输入尺寸"为自变量的模型都解释不了 —— 这部分
     需要 plan 上下文项，属于当前模型的已知边界。§5 把这里的残差拆成了两类：**Batcher 那一行原本是
     trace 归因错误（已修，不能当算子代价用）**，Normalize / ImageReader 才是真实的上下文效应。

也就是说：**affine 是必要的（少了 b 会有 70%+ 的系统性偏差），但还不充分（还需要覆盖尺寸范围
与 plan 上下文）**。

## 复现

```bash
# 容器内
bash scripts/run_unopt_order_transfer_20260921.sh              # 四个顺序各 1 次的 trace（约 10 分钟）
bash scripts/run_unopt_order_transfer_repeats_20260921.sh      # 四个顺序各 3 次、round-robin（约 20 分钟）
python -u scripts/analyze_unopt_order_transfer.py              # 上面的表格（有 repeats 目录时自动按重复取均值）
python -u tmp_analysis/batch_trace_probe.py                    # §5.1：单进程探针，看 batcher 窗口现在量的是什么
python -u tmp_analysis/batch_normalize_rows.py outputs/<run>   # §5：逐 cell 打印 pipe 0/1/8 的 size/process/wall
```

产物：`outputs/unopt_order_transfer_20260921/`（单次）与 `outputs/unopt_order_transfer_repeats_20260921/`（3 次重复，含 `analysis.json`）。
注意：reconcile trace 由 MP worker 落盘，所以实验用 **W=4**（W=1 时 pipeline 跑在 driver 进程里，不产生
trace）；四个 cell 都是"全 INPROCESS、无融合、无 offload"，除顺序外完全一致。

## 4. 重复实验、输入一致性与"控制算子"（2026-09-21 追加）

单次运行不足以断言"位置不变的算子开销相同"，因此把四个顺序各跑 **3 次（round-robin）**，每次保留独立
trace；脚本 `scripts/run_unopt_order_transfer_repeats_20260921.sh`，产物在
`outputs/unopt_order_transfer_repeats_20260921/`。

**输入完全一致（已核对）**：四个顺序、每次运行都是 **4 个 worker × 每个 worker 592 个 batch × batch=4
= 9,472 样本**，数据集路径/分片方式/开关都相同；reconcile 的 `input_sizes` 也逐算子一致（差异只来自
trace 抽样，见下）。

**计划级吞吐（rec/s，3 次）**

| 顺序 | r1 | r2 | r3 | 波动 |
| --- | ---: | ---: | ---: | ---: |
| declared | 165.1 | 173.5 | 169.2 | ±2.5% |
| pico | 280.8 | 284.3 | 290.9 | ±1.8% |
| cedar | 277.1 | 272.1 | 275.5 | ±0.9% |
| old-dp | 66.7 | 62.1 | 70.0 | ±6.3% |

**控制算子：ImageReader（四个顺序里位置、输入尺寸、上游都相同）**

| 顺序 | 每记录 process-time | 相对均值 |
| --- | ---: | ---: |
| declared | 2.3768 ± 0.023 | −0.6% |
| pico | 2.3903 ± 0.045 | −0.1% |
| cedar | 2.3884 ± 0.036 | −0.2% |
| old-dp | **2.1513 ± 0.021** | **−9.9%** |

读法：

- 上一次担心的"reader 在 declared/pico 之间差 8%"是**单次运行的噪声**：3 次重复后 declared / pico /
  cedar 的 reader 分别是 2.377 / 2.390 / 2.388 ms，彼此相差 ≤0.6%，落在 run-to-run 噪声内（全体算子的
  平均相对标准差 **2.3%**）。
- 但 **old-dp 顺序下的 reader 稳定快 10%**（2.151 ± 0.021，明显超出噪声）：四个顺序里读者本身的工作、
  输入都相同，唯一区别是**下游流水线的速度**（old-dp 顺序最慢，64–70 rec/s，其余 165–291 rec/s）。也就是说
  即使"位置和尺寸不变"，同一算子的 CPU 时间也会随 plan 上下文（机器负载/频率/缓存）变化约 10% —— 这类
  效应任何只以"输入尺寸"为自变量的模型都覆盖不到，是本次实验给出的方法论边界。

**用重复均值重做迁移对比（declared → pico）**

| 算子 | t pico (ms) | cedar 等比例 | 误差 | affine | 误差 | run-to-run 噪声 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| T Batcher | 5.022 ± 0.24 | 7.711 | +53.5% | 7.711 | +53.5% | 4.8% |
| N Normalize | 0.170 ± 0.00 | 0.240 | +41.3% | 0.240 | +41.3% | 0.6% |
| B Blur | 8.233 ± 0.19 | 2.042 | **−75.2%** | 2.238 | −72.8% | 2.3% |
| G Grayscale | 0.288 ± 0.00 | 0.089 | **−68.9%** | 0.157 | −45.6% | 1.5% |
| J Jitter | 0.492 ± 0.00 | 0.720 | +46.2% | 1.081 | +119.6%（越界外推） | 0.7% |
| H Flip | 0.080 ± 0.00 | 0.011 | **−86.2%** | 0.054 | −32.8% | 1.7% |
| C Crop | 2.090 ± 0.01 | 0.527 | **−74.8%** | 1.688 | **−19.2%** | 0.4% |
| F to_float | 0.062 ± 0.00 | 0.042 | −31.7% | 0.042 | −31.7% | 0.6% |
| R ImageReader | 2.390 ± 0.05 | 2.377 | −0.6% | 2.377 | −0.6% | 1.9% |

- **平均 |误差|：cedar 等比例 53.1%、affine 46.3%；中位 53.5% / 41.3%**；全体算子的平均 run-to-run 噪声只有
  **2.3%** —— 也就是说这些迁移误差（50–86%）**远超噪声**，结论稳健。
- 留一顺序（用四个顺序的重复均值）：cedar 84.1%（中位 61.8%）vs affine 69.3%（中位 47.1%），affine 仍系统性更好。
- 结论不变：**Cedar 的"按输入字节等比例"每算子代价无法迁移到新 plan**（尺寸变化的那几个算子低估 69–86%），
  补上 affine 的固定项 b 能把最坏的几个拉回 −19%/−33% 量级；剩余误差来自 ① 越界外推、② 上面这种约 10% 的
  plan 上下文效应。

## 5. 两个异常行的性质：Batcher 是 trace 归因错误（已修），Normalize 是真实的 plan 上下文效应

§4 的表里有两行看着像"模型失效"（T Batcher 5.02 ↔ 12.55 ms/record，N Normalize 0.170 ↔ 0.240
ms/record，而两者四个顺序的输入尺寸都相同）。逐行核查后，**两行的成因完全不同**。

### 5.1 T Batcher 行 = batcher 的 trace 归因错误（已在代码里修掉）

**现象**：batcher 的 trace 窗口量的是"这批 4 条记录在自己上游被生产出来"的 CPU 跨度，而不是
batcher 自己的 `torch.stack`：7.711（declared）/ 5.022（pico）/ 5.122（cedar）/ 12.550（old-dp）
ms/record，而 batch=4、单条 `(3, 244, 244)` float32 的 stack 单独测只有 **0.089 ms/batch ≈ 0.022
ms/record** —— 差两个数量级。

**根因**：`InProcessBatcherPipeVariant._iter_impl` 历史上在**每次 append** 时复制 trace
（`if x.do_trace: batch_ds.copy_metadata_from(x)`）。source 只按 `TRACE_FREQUENCY_SEC = 0.1 s`
抽样打 trace，所以一个 batch 最终带的是**更早那条记录**的链路；`trace_dict[batcher]` 减去那条记录
在上游的输出时间，差值里就混进了整批组装期间上游的 CPU 时间。

**修复**（`cedar/pipes/batch.py`）：

- 新增 `_emit_batch(batch_ds, last_input)`，batch 只在**最后一个输入**到达后锚定
  （`copy_metadata_from(last_input)`），窗口 = batcher 自己的 stack + 出队开销；
- trace 抽样间隔改为可配置：`CEDAR_TRACE_FREQUENCY_SEC`（默认 0.1 s，保持原 profiling 成本）。
  reconcile / 归因分析类实验用 `0`（每条记录都 trace）——否则"最后一个输入恰好没被抽到"的 batch 会
  整批不带 trace；
- 修复只影响**之后**产生的 trace，已归档的 profile / 结果快照保持不动。

**验证**（同一 harness：W=4、batch=4、全 INPROCESS、9,469 条、同一份 plan）：

| 顺序 | 修复前 T Batcher（3 次均值 ±stdev） | 修复后 T Batcher（r1，单次） |
| --- | ---: | ---: |
| declared | 7.711 ± 0.17 | 0.058 |
| pico | 5.022 ± 0.24 | 0.048 |
| cedar | 5.122 ± 0.08 | 0.044 |
| old-dp | 12.550 ± 0.55 | 0.066 |

（单位 ms/record。修复前 = `outputs/unopt_order_transfer_repeats_20260921`；修复后 =
`outputs/unopt_order_transfer_repeats_traceall_20260921` 的 r1 四个 cell。单进程探针
`tmp_analysis/batch_trace_probe.py` 独立给出每 4 条记录 0.167–0.202 ms，即 0.042–0.051 ms/record，
与表中一致，且该次每个样本的 `trace_order` 都完整。）

→ **§1 / §2 / §4 里 T Batcher 那一行不能当算子代价使用**：它测的是上游生产整批数据的跨度。修复后这行
四个顺序都是 ≈0.05 ms/record、建模输入尺寸也相同（238,144 B/record），两个模型都能预测，不再提供区分度。

### 5.2 N Normalize 行 = 真实的 plan 上下文效应

| 顺序 | 输入 (B/record) | process (ms/record) | wall (ms/record) |
| --- | ---: | ---: | ---: |
| declared | 238,144 | 0.2423 | 0.2005 |
| pico | 238,144 | 0.1724 | 0.1554 |
| cedar | 238,144 | 0.1338 | 0.1201 |
| old-dp | 2,644,349 | 0.9977 | 0.9760 |

- declared / pico / cedar 三列**输入尺寸完全相同**、batch 也相同，process-time 却差 **1.8×**；
- 修复 batcher 归因前后这两行的值几乎不变（0.240 / 0.170 / 0.139 → 0.242 / 0.172 / 0.134），说明它跟
  5.1 的 bug 无关：Normalize 在 batcher **之前**，它的 trace 锚点一直是自己那条记录的上游；
- wall 与 process 同量级（process/wall = 1.02–1.21，多线程算子的 CPU 时间本就略高于 wall），**差异不是
  排队**；同样尺寸的 payload 在各顺序里的形状/内存布局、上游节奏都不同，算子自身在同一尺寸下的成本因此
  相差 1.8×。无论具体机制是布局还是缓存/频率状态，都落在"只以输入字节为自变量"的模型之外；
- 同类效应在 ImageReader 上也出现过（旧运行里 old-dp 的 reader 快 10%），但修复后的 r1 里四个顺序的
  reader 是 2.06 / 1.99 / 2.05 / 2.05 ms（≤3%），所以 10% 这一档还需要更多重复才能定量。

**对结论的影响**：迁移结论建立在 `B Blur / G Grayscale / H Flip / C Crop / J Jitter` 这些**输入尺寸随
顺序改变**的算子上（低估 −69% ~ −86%），它们不受 5.1 影响；**T Batcher 行作废**，**N Normalize 行是
上下文边界**而不是 affine 形式的问题。

### 5.3 Reader 的输出尺寸差异（declared 646.6 kB vs pico 678.8 kB）也是抽样伪影

§4 的另一个疑点：`R ImageReader` 在四个顺序里输入完全相同（157 B/record），但它的**输出**尺寸却随顺序
不同——三次重复的均值分别是 declared **646.6 kB**、pico **678.8 kB**、cedar 629.7 kB、old-dp 564.3 kB
（每 record）。核查后同样是测量伪影，不是数据不同：

- reader 的输出尺寸 = **解码后像素字节数**（`get_sizeof_data(PIL.Image)` → `W×H×bands`），而 imagenette2/train
  的尺寸分布是**重尾**的：9,469 张图的均值 675.9 kB、标准差 **1,339 kB**（CV 1.98，最大一张解码后 38 MB）；
- reconcile 的 `output_sizes` 只是**被采样到的那部分记录**的均值，而 trace 是**按时间**抽的（默认每
  0.1 s 一条），所以每个顺序抽到的记录集合都不同：n ≈ 440 时的标准误 **≈64 kB（9.4%）**，
  646.6 ↔ 678.8 的 32 kB 差异远在噪声内；
- **直接验证**：把 trace 改成每条记录都打（`CEDAR_TRACE_FREQUENCY_SEC=0`，即 §5.1 之后 reconcile
  实验的默认设置）后，declared / pico / cedar 三个顺序的 reader 输出尺寸**完全相同**——
  均值 666.1 kB，连四个 worker 的分项都逐一对上（639.0 / 702.4 / 689.4 / 633.7 kB）；old-dp 是 661.1 kB，
  只因为那次运行的 old-dp cell 被提前中止（2,252 / 2,368 条）。

→ 结论：**不采样时该算子的尺寸在每个顺序里都一样**；只要用按时间抽样的 trace，重尾分布算子的"每记录
尺寸/成本"就会带上约 10% 的抽样抖动，跨顺序比较时不能把它当成数据差异。复现：
`tmp_analysis/probe_reader_size_sampling.py`（数据集尺寸分布 + 步长抽样仿真）。

## 6. 附：新 profile 里 simclrv2 各算子的 affine 拟合原始数据

数据来源：**大规模正式实验那一份新的 layered profile** ——
`outputs/ultimate_eight_optimizers_20260920/simclrv2/profiles/shared.yaml` 的
`physical_model.operator_affine` 段。该 profile 由正式实验的 profiling 阶段产生
（simclrv2 输入 189,380 条 = 9,469 × 20 epochs，`CEDAR_LAYERED_ADAPTIVE_PROFILE=1`，
`profile_protocol = dual_legacy_whole_pipeline_plus_adaptive_layered`），与修复后重跑的 root
`outputs/ultimate_eight_optimizers_fix_20260921/simclrv2/profiles/shared.yaml` **逐字节相同**
（md5 `7758769e1a6778059bf2521974aab03e`），也就是正式结果表里 `simple-dp-opt (new profile)`
消费的那一份；`old-dp-opt (legacy profile)` 消费的是同一文件里的 `baseline`/`offloads` 旧字段。
拟合字段：`method: two_stratum_operator_affine_fit`、`min_contrast: 1.1`、`schema_version: 1`。
拟合协议（`cedar/client/dataset.py`）：每个算子取**合法输入的最小与最大两个
尺寸层**，各测一次"单记录成本 t"，用 `t = k·x + b` 的两点解
`k = (t_hi − t_lo)/(x_hi − x_lo)`、`b = t_lo − k·x_lo`（都截断在 ≥0）；
`fixed_fraction = b/(k·x_ref + b)` 是"参考尺寸处不随 payload 收缩的那部分成本占比"（截断在 0.95）。
`source = legal_inputs` 表示两个点就是算子在真实流水线里见过的两个不同尺寸；`rescaled_legal_input`
表示算子自身没有尺寸差（同一层），于是把它自己的合法输入**缩放到最小/最大层的尺寸**上做反事实测量。

**有尺寸差的 7 个算子（原始观测点 + 拟合系数）**

| 算子 | 点 lo (B → ms/record) | 点 hi (B → ms/record) | k (ms/byte) | b (ms) |
| --- | --- | --- | ---: | ---: |
| N Normalize | 59,945 → 0.068691 | 953,001 → 0.289146 | 2.46854e-07 | 0.0538935 |
| B Blur | 59,945 → 3.412598 | 953,001 → 48.373382 | 5.03449e-05 | 0.394676 |
| G Grayscale | 179,026 → 0.089917 | 2,858,153 → 0.665809 | 2.14955e-07 | 0.0514349 |
| J Jitter | 179,026 → 2.457084 | 2,858,153 → 33.333920 | 1.15250e-05 | 0.393816 |
| H Flip | 179,026 → 0.040905 | 2,858,153 → 0.234878 | 7.24018e-08 | 0.0279428 |
| C Crop | 1,500,421 → 1.625097 | 2,250,422 → 1.736740 | 1.48857e-07 | 1.40175 |
| F to_float | 226,058.5 → 0.073043 | 562,921 → 0.219711 | 4.35396e-07 | 0 |

**参考尺寸、模型代价与固定分量占比**

| 算子 | 点来源 | x_ref (B) | k·x_ref + b (ms) | b 占比 |
| --- | --- | ---: | ---: | ---: |
| N Normalize | rescaled_legal_input | 238,144.00 | 0.112680 | 0.478287 |
| B Blur | rescaled_legal_input | 238,144.00 | 12.384003 | 0.031870 |
| G Grayscale | rescaled_legal_input | 714,432.00 | 0.205006 | 0.250895 |
| J Jitter | rescaled_legal_input | 714,432.00 | 8.627617 | 0.045646 |
| H Flip | rescaled_legal_input | 714,432.00 | 0.079669 | 0.350736 |
| C Crop | legal_inputs | 2,390,209.52 | 1.757548 | 0.797559 |
| F to_float | legal_inputs | 597,552.38 | 0.260172 | 0.0 |

**k = 0 的两个算子**（payload 无尺寸差，按同一族的常数项记录，`status = constant_measured`）

| 算子 | measurement | x_ref (B) | k·x_ref + b = b (ms) | b 占比 |
| --- | --- | ---: | ---: | ---: |
| T Batcher | batcher | 238,144.00 | 0.0254309 | 1.0 |
| R ImageReader | image_reader | 159.09 | 1.931472 | 1.0 |

- p_id 9（dummy source）不在拟合范围内，profile 记
  `unfitted_reasons = {"9": "not_a_single_input_stage"}`，其余算子全部有拟合值（`unfitted_operators = []`）。
- 这些点是把**不同尺寸的 payload 单独喂给算子**测出来的；放回真实流水线（上游节奏、内存布局、是否融合
  都不同）绝对值会有出入 —— 所以 §2 / §4 只用 profile 的 k/b **形状**，锚点仍是同一份 declared 实测。
- 原始数字随时可以从上面的 yaml 路径重读（`physical_model.operator_affine.operators`）。
