# PICO 表示感知计算模型：实现、消融与验证（2026-09-24）

本文件是"把计算模型从字节换成元素数 + 表示类"这一轮的交付说明。
结果目录：`outputs/affine_repr_model_20260924/`（模型与计划证据）、
`outputs/affine_repr_profile_20260924/`（新 profile）、
`outputs/stage_b_repr_20260924/`（端到端消融）。

## 1. 模型

```
compute_i(p) = k_(i, class(p)) * elements(p) + b_(i, class(p))     [ms/源记录]
bytes(p)     仍用于 boundary / cache / transport（未改动）
```

- 表示类 `class(p)`：payload 的 dtype/通道/容器（`uint8:3ch`、`float32:1ch`、`path`…），
  规划期可确定；元素数 `elements(p)`：图像 `C*H*W`、PIL `W*H*bands`、文本 `len`。
- 与旧模型的关系：**同样的两参数 affine**，换成"每 (算子, 表示类) 一对"，
  自变量从序列化字节换成元素数。旧的字节模型与旧 optimizer 全部保留。

## 2. 实现落点

| 位置 | 内容 |
| --- | --- |
| `cedar/pipes/common.py` | `payload_compute_scale`、`payload_representation_class` |
| `cedar/client/dataset.py` | `_profile_operator_compute_model`：按表示类拟合；`_time_operator_grid_mean_ms`：逐算子交错长窗口均值 |
| `cedar/compose/simple_dp_ablation_optimizer.py` | `_RepresentationComputeMixin`：元素/类传播、按位置定价、缺失类显式处理；变体 34/35/36/37/38 |
| `evaluation/compare_optimizer_perf.py` | CLI 名 `simple_dp_repr_elements` / `_repr_proportional` / `_repr_affine` / `_workers_width_repr_affine` |
| `scripts/run_repr_profile_20260924.sh` | 生成新 profile（含远端 Ray runtime_env 快照） |
| `scripts/run_stage_b_pico_repr_20260924.sh` | 端到端消融 |

关键实现点（详见 `audit.md` D1–D4）：

1. 每点 5 s 实测预算、同一算子所有点交错测量——短窗口会把 Blur 均值低估 3–6 倍；
2. 表示类状态按特征声明顺序递推（不按位序），否则非法前缀会污染状态；
3. 未测表示类不给回退：惩罚价 + `assert_plan_covered(plan)`，最终计划若落在未测类上直接报错；
4. batcher 的 element_ratio 按 `records_per_call` 归一。

## 3. 消融（同一 profile、同一测量、11 个计划 × 3 次完整运行）

目标 = **逐算子在完整运行中的平均自身服务时间之和**（ms/源记录）。

| 计划 | 实测 | M1 字节比例 | M2 字节 affine | M3 元素 affine | M4 表示感知(过原点) | M5 表示感知 affine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| a_f0（cast 最前） | 29.788 | 28.175 | 20.537 | 26.002 | 24.338 | 26.002 |
| a_f1 | 25.984 | 24.969 | 20.155 | 25.832 | 24.267 | 26.107 |
| a_f2 | 25.601 | 24.865 | 20.114 | 25.832 | 24.219 | 26.078 |
| a_f3 | 24.974 | 15.445 | 13.889 | 25.832 | 25.432 | 28.017 |
| a_f4 | 23.295 | 14.951 | 13.720 | 25.779 | 25.408 | 28.010 |
| a_f5（cast 最后） | 21.824 | 7.138 | 7.388 | 25.779 | 22.752 | 25.354 |
| pico | 13.953 | 5.021 | 5.995 | 18.215 | 10.569 | 12.779 |
| cedar | 13.136 | 4.662 | 6.031 | 18.398 | 10.693 | 11.803 |
| old-dp | 72.438 | 117.356 | 92.967 | 130.848 | 50.542 | 51.507 |
| v2 | 18.864 | 18.274 | 15.112 | 18.621 | 13.380 | 14.380 |

误差（相对实测的比值，跨 11 个计划）：

| 模型 | 比值范围 | MAPE | RMSE |
| --- | --- | ---: | ---: |
| M1 字节比例 | 0.32–1.62 | 37.7% | 46.8% |
| M2 字节 affine | 0.33–1.28 | 41.2% | 44.5% |
| M3 元素 affine（不分类） | 0.87–1.81 | 19.5% | 30.0% |
| M4 表示感知过原点 | 0.70–1.09 | 13.6% | 17.1% |
| M5 表示感知 affine | 0.71–1.20 | **13.5%** | **15.8%** |

分步结论：

- **M2→M3（换自变量）**：MAPE 41.2% → 19.5%，是最大的一步；
- **M3→M5（加表示类）**：19.5% → 13.5%；
- **M4→M5（加截距 b）**：13.6% → 13.5%（RMSE 17.1% → 15.8%）→ **b 在表示类正确之后
  几乎没有独立贡献**；本轮的贡献应表述为"表示感知"，而不是"截距"。

## 4. 一致性验收

- **DP vs 穷举**：关闭 fusion/offload/parallelism 后，1260 个合法顺序全部枚举，
  用独立参考实现打分；DP 选中的顺序落在 argmin（目标 11.717960，共 4 个并列最优）。
  命令：`python -u tmp_analysis/test_repr_dp_optimality.py <profile>`（PASS）。
- **缺失类**：DP 构造候选时会遇到违反依赖的掩码；实现给惩罚价并记录，
  最终计划由 `assert_plan_covered` 校验，未测类不会被静默回退。
- **边界未动**：`_dp_work_prod`（字节）未覆盖；新变体只改计算项。

## 5. 成本

| 项 | 数值 | 说明 |
| --- | ---: | --- |
| profiling：旧 layered profile | ≈11 min | 2026-09-19 campaign（simclrv2） |
| profiling：加入表示感知后 | ≈21 min | 同一脚本、同一环境；差 ≈10 min |
| 其中表示感知阶段 | ≈10 min | 9 个算子 / 33 条类曲线，默认 5 s/点；`CEDAR_PROFILE_COMPUTE_TARGET_SEC` 可调 |
| 规划（DP） | 见阶段 B | 表规模：`element_prod`/`class_state` 各 2^n（本负载 n=9） |

## 5b. 端到端消融（阶段 B，1 epoch / 9,469 条，同 profile/资源，round-robin，每格 3 次）

| optimizer | 计算模型 | 选择的计划 | W | 吞吐（均值，min–max） | 优化时间 |
| --- | --- | --- | ---: | ---: | ---: |
| `simple_dp_boundary` | 字节 affine | 单条 INPROCESS FusedPipe{6,3,2,4,5,7,1} | 1 | 74.0 /s（73.7–74.3） | 2.7 s |
| `simple_dp_repr_affine` | M5 | FusedPipe{3,6} + RAY blur + FusedPipe{5,7,1,4} | 64 | 1348.7 /s（1340.6–1353.7） | 26.1 s |
| `simple_dp_workers_width_boundary`（PICO 现状） | 字节 affine | 单条 FusedPipe{6,3,2,4,5,7,1} | 64 | 2404.0 /s（2355.3–2468.3） | 275.5 s |
| `simple_dp_workers_width_repr_affine`（新 PICO） | M5 | 单条 FusedPipe{3,6,2,5,7,1,4} | 64 | 2319.9 /s（2260.4–2371.7） | 288.8 s |

- 粗搜索变体（`simple_dp_boundary`）：字节模型选 W=1（74 /s），表示感知选 W=64 + Ray 阶段（1349 /s）——
  差异来自**模型诱导的 W 决策**，不是算子变快。
- 完整 PICO（W×width）：两者都选 W=64、单条 INPROCESS 融合，吞吐 2404 vs 2320 /s，
  **区间重叠 → 不可区分**。因此本轮**没有**证明新模型在完整 PICO 上带来吞吐提升；
  它证明的是成本估计更准与固定物理配置下的选择更接近实测最优。
- 两种吞吐差都**不是**等价优化收益（见 §6 与 `semantic_scope.md`），数据量也只有 campaign 的 1/20。

## 6. 语义范围（本轮的重要负面结论）

`to_float` 是 `x.to(torch.float32)`，不归一化到 [0,1]；torchvision 对 float 图像按 [0,1]
解释，因此**把 cast 前后移动会让 ColorJitter 的输出被 clamp 到 [0,1]**（实测 32 条记录：
jitter/grayscale/blur 平均绝对差 101–104，而 crop/flip 只有 0.24 的量化差）。

→ SimCLRv2 上**不存在语义等价的重排**；重排的成本差异可用于验证成本模型，
**不能**作为等价优化收益。详见 `semantic_scope.md` 与 `dtype_semantics.json`。

## 7. 交付物

| 文件 | 内容 |
| --- | --- |
| `plan_summary.csv` | 每个计划的实测均值/p10、各模型预测与比值 |
| `operator_results.csv` | 逐计划逐算子：表示类、元素数、实测均值、M1–M5 |
| `predictions.csv` | 长表：计划 × 算子 × 模型 |
| `measurements.csv` | 每次完整运行的 p10/中位/均值之和 |
| `figure_data.json` | 计划表、模型汇总、选择与后悔值、逐轮数据 |
| `model_summary.json` | MAPE/RMSE 与选择结果 |
