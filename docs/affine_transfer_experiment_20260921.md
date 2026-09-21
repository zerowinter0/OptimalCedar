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
bash scripts/run_unopt_order_transfer_20260921.sh      # 四个顺序的 trace（约 10 分钟）
python -u scripts/analyze_unopt_order_transfer.py      # 上面的表格
```

产物：`outputs/unopt_order_transfer_20260921/{plans,logs,results,reconcile_*,analysis.json}`。
注意：reconcile trace 由 MP worker 落盘，所以实验用 **W=4**（W=1 时 pipeline 跑在 driver 进程里，不产生
trace）；四个 cell 都是"全 INPROCESS、无融合、无 offload"，除顺序外完全一致。
