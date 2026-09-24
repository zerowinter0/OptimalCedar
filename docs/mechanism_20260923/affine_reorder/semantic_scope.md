# 语义范围：哪些重排可以用于"优化收益"的正式验证

本轮把候选计划分成两套清单。**只有清单 A 的加速可以写进"优化收益"**；
清单 B 的差异里混着任务行为改变（例如 1 通道上的 ColorJitter 丢掉色相/饱和度），
只能作为表示与成本响应的诊断。

## 1. 系统允许的顺序

`evaluation/pipelines/target_pipeline/simclr/cedar_dataset.py` 只声明四条约束：

- reader 最先（`ImageReaderPipe(...).fix()`）；
- `RandomHorizontalFlip.depends_on(["crop"])`；
- `Normalize.depends_on(["float"])`；
- `BatcherPipe(...).fix()` 为 sink。

因此系统的"合法顺序"= 这些边之外任意线性延伸，包含把 `RandomResizedCrop`、
`ColorJitter`、`Grayscale`、`GaussianBlur` 任意重排——**这在语义上并不等价**。

## 2. 本轮结论：该负载上不存在"语义等价的重排"

原计划（下面的清单 A）是"只移动 `to_float`，增强算子顺序不变"。**独立检查否定了这个假设**：
`to_float` 是 `x.to(torch.float32)`，**不做 /255 归一化**；而 torchvision 对 float 图像
按 [0,1] 解释（`ColorJitter` 等会 clamp 到 1）。于是：

| 计划 | ColorJitter 输入 | 输出范围（4 张真实图） | 与 uint8 路径的平均绝对差 |
| --- | --- | --- | --- |
| cast 在最前（declared） | float32，值域 0..255 | **[0.00, 1.00]**（被 clamp） | 85–130（0..255 尺度） |
| cast 在最后 | uint8，值域 0..255 | [0, 235]（正常） | — |

数据：`outputs/affine_reorder_diagnosis_20260924/dtype_semantics.json`（4 条记录）
与 `plan_output_differences.json`（32 条记录，逐算子）。

→ **移动 `to_float` 会显著改变增强结果**，所以本负载上：

- 没有任何重排与声明计划在任务语义上等价；
- 重排带来的成本下降（例如把 cast 后移，Blur/Jitter 走 uint8 路径更便宜）
  **不能当作等价优化收益**，只能作为"成本模型必须具备的预测能力"的证据；
- 训练质量对照需要现有训练评估入口；本轮没有该协议，**该项列为待定**，
  并且按照"必须改变任务语义才能获得收益则停止扩大实验"的要求，
  本轮不再扩大吞吐/质量实验。

## 3. 清单 A（已作废，仅保留设计记录）

SimCLRv2 的增强链是 `crop → flip → jitter → grayscale → blur`，`to_float` 是纯 cast。
把 cast 放在链上任意位置，**增强算子的相对顺序完全不变**，改变的只是这些算子
计算时的元素类型（cast 在前 = float32，cast 在后 = uint8）。

| 计划 | 顺序（pipe id） | cast 位置 | 语义 |
| --- | --- | --- | --- |
| `a_f0`（= declared） | 9 8 7 6 5 4 3 2 1 0 | 最前 | 参照 |
| `a_f1` | 9 8 6 7 5 4 3 2 1 0 | crop 后 | 同增强顺序 |
| `a_f2` | 9 8 6 5 7 4 3 2 1 0 | flip 后 | 同增强顺序 |
| `a_f3` | 9 8 6 5 4 7 3 2 1 0 | jitter 后 | 同增强顺序 |
| `a_f4` | 9 8 6 5 4 3 7 2 1 0 | grayscale 后 | 同增强顺序 |
| `a_f5`（= v1） | 9 8 6 5 4 3 2 7 1 0 | blur 后 | 同增强顺序 |

含义：这 6 个计划处理的数据、算子参数、算子顺序一致；差异是
**每个算子计算时的元素类型**，因此字节量随之变化（float32 是 uint8 的 4 倍），
而元素数不变。这正是本轮成本模型要解释的对象。

**待定项（不发明阈值）**：如上，float32 与 uint8 路径的差异远大于量化误差
（jitter 被 clamp 到 1.0），因此"训练质量等价"必须先由现有训练评估入口回答。

## 4. 清单 B：成本模型的验证计划（本轮实际使用）

| 计划 | 与 A 的差别 | 为什么不能当优化收益 |
| --- | --- | --- |
| PICO 顺序 9 8 6 3 4 5 2 7 1 0 | grayscale 提前到 jitter/blur 之前 | 1 通道上的 ColorJitter 丢掉色相/饱和度；blur 作用对象改变 |
| cedar 顺序 9 8 5 6 3 4 2 7 1 0 | flip 在 crop 前、grayscale 提前 | flip 与 crop 交换会改变几何基（flip 后 crop 与 crop 后 flip 裁剪区域不同） |
| old-dp 顺序 9 7 6 5 4 3 2 1 0 | 全部 float32 + 无 grayscale 前置变化 | 实际是"blur 在 grayscale 后不变"但输入是 3 通道大图：工作量本身不同 |
| v2 顺序 9 8 7 3 6 5 4 2 1 0 | grayscale 在 crop 前 | 缩放对象从三通道变为单通道，且 jitter 作用于单通道 |

清单 B 的价值：它们让同一个算子拿到**不同表示类**的输入（u8 1ch、f32 3ch 大图等），
因此是检验 `k_(i,z)·e+b_(i,z)` 是否比字节模型更准的最好压力测试。
论文里必须同时标注"这些加速不构成等价优化收益"。

## 4. 输出一致性检查（确定性部分）

固定每个算子每次调用的随机种子后，逐算子比较输出：

- `crop/flip/jitter/grayscale/blur` 在 A 清单各计划里拿到的是同一批记录、
  同一组参数；差异仅来自计算时的 dtype（float32 vs uint8 量化）。
- 本轮给出**数值差异的度量**（最大绝对差、平均绝对差、以及量化步长上界 1/255），
  不声称逐位相等；随机算子的"等价"标准是：同一记录、同一种子、同一算子参数下
  输出分布一致（而不是输出逐位相同）。

复现：`tmp_analysis/compare_plan_outputs.py`（A 清单逐记录比较 + 数值差统计）。
