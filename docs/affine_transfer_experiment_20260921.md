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
   - **同一尺寸、不同 plan 上下文**：Batcher / Normalize / ImageReader 在两个顺序里输入尺寸完全相同，
     但实测 CPU 时间差 1.1–1.6×（batch 组装等待、上游节奏变化），任何只以"输入尺寸"为自变量的模型都
     解释不了 —— 这部分需要 plan 上下文项，属于当前模型的已知边界。

也就是说：**affine 是必要的（少了 b 会有 70%+ 的系统性偏差），但还不充分（还需要覆盖尺寸范围
与 plan 上下文）**。

## 复现

```bash
# 容器内
bash scripts/run_unopt_order_transfer_20260921.sh              # 四个顺序各 1 次的 trace（约 10 分钟）
bash scripts/run_unopt_order_transfer_repeats_20260921.sh      # 四个顺序各 3 次、round-robin（约 20 分钟）
python -u scripts/analyze_unopt_order_transfer.py              # 上面的表格（有 repeats 目录时自动按重复取均值）
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
