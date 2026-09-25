# Final W-only PICO：交付与完成矩阵

commit: `b182e58fee560112459bf24acaad369b0d3eed1c`

## 身份

最终 PICO = `pico_final`（selector 39，`SimpleDpWorkersBoundaryAffineReprOptimizer`）：
联合搜索顺序/融合/后端/缓存与 W；**不搜索 stage width**（每个并行 stage 固定 width=1）；
计算项 = `k_(算子, 表示类)·元素数 + b_(算子, 表示类)`；边界项沿用字节模型。详见 `identity.json`。

## 完成矩阵（由已完成 cell 自动生成）

| 负载 | cells | optimizers |
| --- | ---: | --- |
| llava_pretrain | main_fast | pico_final, plumber_optimizer, raydata_optimizer |
| simclrv2 | ablation_fast, dp_repeat, main_fast, staged_fast | optimizer, pico_byte_proportional, pico_final, pico_final_no_boundary, plumber_optimizer, raydata_optimizer, simple_dp_workers_boundary, staged_final |
| simclrv2_cache | ablation_fast, main_fast | optimizer, pico_byte_proportional, pico_final, pico_final_no_boundary, plumber_optimizer, raydata_optimizer, simple_dp_workers_boundary |

## 吞吐（均值 ± 范围，samples/s）

| 负载 | cell | optimizer | 轮数 | 均值 | 范围 |
| --- | --- | --- | ---: | ---: | --- |
| llava_pretrain | main_fast | pico_final | 1 | 17.2 | 17.2–17.2 |
| llava_pretrain | main_fast | plumber_optimizer | 1 | 17.6 | 17.6–17.6 |
| llava_pretrain | main_fast | raydata_optimizer | 1 | 15.8 | 15.8–15.8 |
| simclrv2 | ablation_fast | pico_byte_proportional | 1 | 2408.6 | 2408.6–2408.6 |
| simclrv2 | ablation_fast | pico_final | 1 | 2432.9 | 2432.9–2432.9 |
| simclrv2 | ablation_fast | pico_final_no_boundary | 1 | 2420.3 | 2420.3–2420.3 |
| simclrv2 | ablation_fast | simple_dp_workers_boundary | 1 | 2447.5 | 2447.5–2447.5 |
| simclrv2 | dp_repeat | optimizer | 2 | 686.3 | 590.3–782.2 |
| simclrv2 | dp_repeat | pico_final | 2 | 2429.4 | 2422.0–2436.7 |
| simclrv2 | main_fast | optimizer | 1 | 713.2 | 713.2–713.2 |
| simclrv2 | main_fast | pico_final | 1 | 2424.0 | 2424.0–2424.0 |
| simclrv2 | main_fast | plumber_optimizer | 1 | 160.4 | 160.4–160.4 |
| simclrv2 | main_fast | raydata_optimizer | 1 | 49.8 | 49.8–49.8 |
| simclrv2 | staged_fast | pico_final | 1 | 2444.4 | 2444.4–2444.4 |
| simclrv2 | staged_fast | staged_final | 1 | 2475.0 | 2475.0–2475.0 |
| simclrv2_cache | ablation_fast | pico_byte_proportional | 1 | 2433.9 | 2433.9–2433.9 |
| simclrv2_cache | ablation_fast | pico_final | 1 | 2430.6 | 2430.6–2430.6 |
| simclrv2_cache | ablation_fast | pico_final_no_boundary | 1 | 2364.9 | 2364.9–2364.9 |
| simclrv2_cache | ablation_fast | simple_dp_workers_boundary | 1 | 2508.7 | 2508.7–2508.7 |
| simclrv2_cache | main_fast | optimizer | 1 | 671.8 | 671.8–671.8 |
| simclrv2_cache | main_fast | pico_final | 1 | 2450.7 | 2450.7–2450.7 |
| simclrv2_cache | main_fast | plumber_optimizer | 1 | 140.2 | 140.2–140.2 |
| simclrv2_cache | main_fast | raydata_optimizer | 1 | 101.9 | 101.9–101.9 |

## 计算模型（第三章，来自 `outputs/affine_repr_model_20260924`）

| 模型 | 计划数 | 比值范围 | MAPE | RMSE |
| --- | ---: | --- | ---: | ---: |
| M1 | 11 | 0.32–1.62 | 37.7% | 46.8% |
| M2 | 11 | 0.33–1.28 | 41.2% | 44.5% |
| M3 | 11 | 0.87–1.81 | 19.5% | 30.0% |
| M4 | 11 | 0.70–1.09 | 13.6% | 17.1% |
| M5 | 11 | 0.71–1.20 | 13.5% | 15.8% |

## W 证据（`w_scaling.csv`，含来源与局限）

| 负载 | W | optimizer | 吞吐 /s | 来源 |
| --- | ---: | --- | ---: | --- |
| simclrv2 | 1 | simple_dp_boundary | 74.0 | outputs/stage_b_repr_20260924/results_cheap.json |
| simclrv2 | 64 | simple_dp_repr_affine | 1348.7 | outputs/stage_b_repr_20260924/results_cheap.json |
| simclrv2 | 64 | simple_dp_workers_width_boundary | 2404.0 | outputs/stage_b_repr_20260924/results_pico.json |

局限：W=1 与 W=64 两组来自**不同计划结构**的完整运行（stage B），不是固定结构 W 扫描；固定结构扫描脚本已修好但本轮未跑成，写作时按上表标注。

## 规划成本与 DP 状态数（`planning.csv`）

| 负载 | cell | 算子数 | mask 数 | 最大状态数 | DP 秒 |
| --- | --- | ---: | ---: | ---: | ---: |
| llava_pretrain | main_fast | 16 | 462 | 182488 | 653.4 |
| simclrv2 | ablation_fast | 9 | 18 | 1342 | 2.6 |
| simclrv2 | dp_repeat | 9 | 18 | 369 | 2.5 |
| simclrv2 | main_fast | 9 | 18 | 369 | 2.3 |
| simclrv2 | main_mean | 9 | 18 | 369 | 2.3 |
| simclrv2 | staged_fast | 9 | 18 | 369 | 2.4 |
| simclrv2_cache | ablation_fast | 9 | 18 | 1489 | 2.6 |
| simclrv2_cache | main_fast | 9 | 18 | 497 | 2.3 |

## 失败与负结果（必须保留）

- 未完成 cell（原样保留，不冒充成功）：ablation_fast.failed.json、main_fast.failed.json、main_fast.failed.json
- 语义：移动 `to_float` 会改变 torchvision 的值域解释（float 图像 clamp 到 [0,1]），
  SimCLRv2 上不存在语义等价重排；重排吞吐差不得写成等价优化收益（`semantic_scope.md`）。
- 完整 PICO 上字节 affine 与表示感知模型的吞吐在运行噪声内不可区分
  （2404 vs 2320 samples/s，区间重叠）→ 本轮不宣称吞吐提升。
- 联合 oracle（顺序×融合×后端×W 的独立枚举）只完成顺序×W 轴与
  `state_sufficiency`；plan-replay 入口的联合枚举被阻塞，原因记录在 `oracle_results.json`。

## 复现

```bash
bash scripts/pico_final_all_20260924.sh        # profiles -> smoke -> campaign -> W scaling -> assembly
python -u tmp_analysis/assemble_final_delivery.py   # 任何时候重跑都可以重建本目录的表格
```