# §3.2 补充轮：扩展范围与补齐表示类（2026-09-27）

commit `b791b00d096547e8272bb66bbc2ea994d74b8253`；本轮结果 `outputs/ch32_scale_affine_ext_20260927/`，
旧数据 `outputs/ch32_scale_affine_20260927/` 原样未动；快照 `docs/ch32_scale_affine_ext_20260927/`。

## 1. 归一化定义（冻结，不改）

```
x = 张量原生字节数 / 参考输入原生字节数
y = 算子平均计算时间  / 参考输入平均计算时间
```

参考点沿用第一轮的每个算子单一参考输入（**冻结**，不因新数据改变）：

| 算子 | 参考 cell | 参考原生字节 | 参考均值 (ms) |
| --- | --- | ---: | ---: |
| blur | blur:float32:1ch:t4 | 200704 | 6.9701 |
| jitter | jitter:float32:3ch:t4 | 602112 | 7.1120 |
| crop | crop:float32:3ch:t6 | 1769472 | 1.7531 |
| grayscale | grayscale:float32:3ch:t4 | 602112 | 0.1674 |
| flip | flip:float32:3ch:t4 | 602112 | 0.0743 |

所有表示类共用同一个参考点，因此类之间的字节/耗时差异被原样保留（不做逐类归一化）。

## 2. 本轮新增测量

| 目标 | 新增训练尺寸 | 新增验证尺寸 | 归一化 x 覆盖 |
| --- | --- | --- | --- |
| jitter uint8:1ch | 4 | 4 | 0.435–2.002 |
| crop uint8:1ch | 4 | 4 | 0.113–0.549 |
| flip uint8:1ch | 8 | 4 | 0.007–1.741 |
| grayscale uint8:1ch | 8 | 4 | 0.007–1.741 |
| grayscale float32:1ch | 8 | 4 | 0.027–6.966 |

另有 8 个第一轮 cell 原样重测作为漂移对照（见 §3）。

## 3. 新旧一致性

判据：每个对照 cell 在第二轮与第三轮（复核）各重测一次，任一批次与**冻结的第一轮值**相比满足 |Δ| ≤ max(2×合并块间标准差, 第一轮均值的 5%) 即通过；不通过的批次只从该 cell 剔除。
结果：8 个对照里 6 个两批全通过，共剔除 2 个批次- cell 组合。

| cell | 第一轮 | 第二轮 | Δ2 | 第三轮(复核) | Δ3 | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| blur:float32:1ch:t4 | 6.9701 | 7.0895 | +1.7% | 6.9908 | +0.3% | round2 and recheck both kept |
| blur:uint8:3ch:t7 | 49.5281 | 48.4844 | -2.1% | 50.9492 | +2.9% | round2 and recheck both kept |
| crop:float32:3ch:t6 | 1.7531 | 1.7763 | +1.3% | 1.7652 | +0.7% | round2 and recheck both kept |
| crop:uint8:1ch:t7 | 0.6826 | 0.6993 | +2.4% | 0.6936 | +1.6% | round2 and recheck both kept |
| flip:float32:3ch:t4 | 0.0743 | 0.0735 | -1.1% | 0.0805 | +8.3% | recheck dropped (round2 agrees with round1) |
| grayscale:float32:3ch:t4 | 0.1674 | 0.1839 | +9.9% | 0.1740 | +4.0% | round2 dropped (recheck agrees with round1) |
| jitter:float32:3ch:t4 | 7.1120 | 7.4543 | +4.8% | 7.1795 | +0.9% | round2 and recheck both kept |
| jitter:uint8:1ch:t7 | 1.5991 | 1.6353 | +2.3% | 1.5607 | -2.4% | round2 and recheck both kept |

另外，扩展尺寸里刻意保留了与原范围**同尺寸**的第二个 cell（独立 id、独立测量），
用于在最贴近的点上再检查一次一致性：

| 算子/类 | 尺寸 | 原 cell | 扩展 cell | 相对差 |
| --- | ---: | --- | --- | ---: |
| crop uint8:1ch | 448×448 | t7 0.6918 ms | xt0 0.7114 ms | +2.8% |
| jitter uint8:1ch | 512×512 | t7 1.5984 ms | xt0 1.6572 ms | +3.7% |

只有通过检查的批次才进入 `merged_measurements.csv`；被剔除的组合记在 `consistency.json`。
合并口径：同一 cell 若多个批次都通过，取全部 block 的算术平均（`batch` 列保留来源）。
**跨批次的绝对水平存在 ≤10% 的漂移**（见上表：flip +8.3%、grayscale +9.9% 的批次被判为超差），
而本图要展示的类间差异是 4–17×，因此批次漂移不影响类间结论，但不能用来比较 10% 量级的差异。

## 4. 扩展后的关键拟合

每类的全范围模型 = M5（按表示类分开的 kx+b）；局部模型 = 同一类内只用 x≤2 的训练点，
并在该类 x≤2 的独立验证点上评估。

| 算子 | 类 | M5 全范围（kx+b） | M3 局部 x≤2（比例） | M4 局部 x≤2（kx+b） |
| --- | --- | ---: | ---: | ---: |
| jitter | float32:1ch | 7.4% | — | — |
| jitter | float32:3ch | 22.7% | 17.5% | 5.6% |
| jitter | uint8:1ch | 30.0% | 34.7% | 34.7% |
| jitter | uint8:3ch | 23.5% | — | — |
| crop | float32:1ch | 3.0% | — | — |
| crop | float32:3ch | 0.8% | — | — |
| crop | uint8:1ch | 1.7% | — | — |
| crop | uint8:3ch | 1.1% | — | — |
| flip | float32:1ch | 10.5% | — | — |
| flip | float32:3ch | 9.4% | 50.2% | 9.7% |
| flip | uint8:1ch | 8.7% | — | — |
| flip | uint8:3ch | 3.8% | — | — |
| grayscale | float32:1ch | 22.3% | 46.6% | 7.8% |
| grayscale | float32:3ch | 12.6% | 39.6% | 7.6% |
| grayscale | uint8:1ch | 17.1% | — | — |
| grayscale | uint8:3ch | 7.2% | — | — |

（局部模型只在 x≤2 的训练点上拟合、在 x≤2 的独立验证点上评估；适用范围写明在 `fits.csv` 的
`fit_scope` 与 `constraint` 字段，不能外推使用。）

### 4.1 大尺寸是否主导最小二乘（ColorJitter float32:3ch）

全范围拟合 k=6.258e-05、b=0；x>2 的 2 个训练点贡献了**最小二乘目标的 47%**，窗口内（x≤2）因此出现系统性负残差：

| 训练点 x | 实测 (ms) | 全范围预测 (ms) | 残差 | 目标占比 |
| ---: | ---: | ---: | ---: | ---: |
| 0.08 | 1.510 | 0.769 | +0.741 | 1.4% |
| 0.18 | 2.023 | 1.730 | +0.293 | 0.2% |
| 0.41 | 3.244 | 3.893 | -0.649 | 1.1% |
| 0.73 | 5.372 | 6.921 | -1.548 | 6.2% |
| 1.00 | 7.249 | 9.420 | -2.171 | 12.2% |
| 1.65 | 12.065 | 15.572 | -3.507 | 31.9% |
| 2.94 | 24.985 | 27.683 | -2.698 | 18.9% |
| 5.22 | 52.504 | 49.214 | +3.290 | 28.1% |

排除 x>2 的点后，同一类的局部仿射给出 b=0.673 ms、验证 MAPE 5.6%（全范围 22.7%）。
窗口内的偏差不是块间波动造成的，而是全范围最小二乘被大尺寸点拉高斜率。

### 4.2 扩展段的非单调：ColorJitter uint8:1ch

该类的扩展点在 x≥1.5 处没有随规模单调增长，且同一 cell 的块间中位数摆动很大：

| cell | x | 逐块中位数 (ms) | 逐块均值 (ms) | 均值/中位数 |
| --- | ---: | --- | --- | ---: |
| xv2 | 1.53 | 6.66 / 6.47 / 6.28 | 10.00 / 7.47 / 6.39 | 1.23 |
| xt2 | 1.74 | 7.35 / 13.01 / 7.53 | 8.69 / 12.15 / 10.35 | 1.12 |
| xv3 | 1.97 | 10.27 / 9.47 / 8.87 | 10.35 / 9.66 / 9.45 | 1.03 |
| xt3 | 2.00 | 8.05 / 8.21 / 7.95 | 8.08 / 8.49 / 8.49 | 1.04 |

因此 x≳1.5 这段**不能**用仿射或比例模型描述，也不能当作真实尺度响应；
该范围标记为未覆盖。局部 x≤2 模型同样没有改善（34.7% vs 全范围 30.0%），
因为问题不是截距，而是大 cell 的测量分布。这一负结果按原样保留：不删点、不为贴合曲线调参。

## 5. 执行路径差异（如实记录）

| 表示类 | 实际执行路径 | 依据 |
| --- | --- | --- |
| ColorJitter 1ch | saturation 与 hue 分支直接 `return img`，只有 brightness/contrast 生效 | `torchvision/transforms/functional_tensor.py::adjust_saturation/adjust_hue` |
| Grayscale 1ch | `rgb_to_grayscale` 对单通道走 `img.clone()`，即恒等但有一次拷贝 | 同上 `rgb_to_grayscale` |
| Grayscale 3ch | 真实加权求和（0.2989/0.587/0.114） | 同上 |

## 6. 不支持或未覆盖的类别

| 算子 | 表示类 | 状态 | 已测 cell 数 | 尺寸 | 说明 |
| --- | --- | --- | ---: | --- | --- |
| blur | float32:3ch | measured | 12 | 64 80 96 144 160 192 224 256 288 384 448 512 | measured in round 1 |
| blur | float32:1ch | measured | 12 | 64 80 96 144 160 192 224 256 288 384 448 512 | measured across rounds |
| blur | uint8:3ch | measured | 12 | 64 80 96 144 160 192 224 256 288 384 448 512 | measured across rounds |
| blur | uint8:1ch | measured | 12 | 64 80 96 144 160 192 224 256 288 384 448 512 | measured in round 1 |
| jitter | float32:3ch | measured | 14 | 64 80 96 128 144 160 192 224 256 288 384 448 512 | measured across rounds |
| jitter | float32:1ch | measured | 14 | 64 80 96 128 144 160 192 224 256 288 384 448 512 | measured in round 1 |
| jitter | uint8:3ch | measured | 12 | 64 80 96 144 160 192 224 256 288 384 448 512 | measured in round 1 |
| jitter | uint8:1ch | measured | 20 | 64 80 96 144 160 192 224 256 288 384 448 512 640 736 832 960 1024 1088 1098 | measured across rounds |
| crop | float32:3ch | measured | 12 | 96 112 128 160 192 224 256 288 320 384 416 448 | measured across rounds |
| crop | float32:1ch | measured | 12 | 96 112 128 160 192 224 256 288 320 384 416 448 | measured in round 1 |
| crop | uint8:3ch | measured | 12 | 96 112 128 160 192 224 256 288 320 384 416 448 | measured in round 1 |
| crop | uint8:1ch | measured | 20 | 96 112 128 160 192 224 256 288 320 384 416 448 576 672 768 832 896 960 986 | measured across rounds |
| grayscale | float32:3ch | measured | 12 | 64 80 96 144 160 192 224 256 288 384 448 512 | measured across rounds |
| grayscale | float32:1ch | measured | 12 | 64 96 144 192 224 320 384 448 640 704 832 1024 | measured in round 2; torchvision rgb_to_grayscale clones a 1-channel input (identity path), so the curve is a memory copy |
| grayscale | uint8:3ch | measured | 12 | 64 80 96 144 160 192 224 256 288 384 448 512 | measured in round 1 |
| grayscale | uint8:1ch | measured | 12 | 64 96 144 192 224 320 384 448 640 704 832 1024 | measured in round 2; same clone path as float32:1ch |
| flip | float32:3ch | measured | 12 | 64 80 96 144 160 192 224 256 288 384 448 512 | measured across rounds |
| flip | float32:1ch | measured | 12 | 64 80 96 144 160 192 224 256 288 384 448 512 | measured in round 1 |
| flip | uint8:3ch | measured | 12 | 64 80 96 144 160 192 224 256 288 384 448 512 | measured in round 1 |
| flip | uint8:1ch | measured | 12 | 64 96 144 192 224 320 384 448 640 704 832 1024 | measured in round 2 |

- 五个算子 × 四种表示类全部有测量。

## 7. 文件与复现

- `measurements.csv`：本轮新测量（沿用第一轮字段 + `origin`）。
- `merged_measurements.csv`：通过一致性检查后的两轮合并数据（含 `batch` 列）。
- `fits.csv` / `predictions.csv`：全范围与局部模型、逐点预测与误差。
- `figure_data.json`：每个算子/类的原生字节、元素数、耗时、归一化坐标、参考点与拟合系数。
- `coverage.csv`、`consistency.json`、`env.json`、`raw_calls.csv.gz`（逐调用原始耗时）。
- `verify.json`：从 `predictions.csv` 重新推导全部指标并与 `fits.csv` 对账（0 差异才算通过）。

```bash
python -m tmp_analysis.ch32_scale_affine_ext measure   # 远程 Ray actor，3 blocks
python -m tmp_analysis.ch32_scale_affine_ext merge     # 一致性检查 + 合并
python -m tmp_analysis.ch32_scale_affine_ext fit       # 拟合/验证/交付表
python -m tmp_analysis.ch32_scale_affine_ext manifest  # 校验和 + docs 快照
```