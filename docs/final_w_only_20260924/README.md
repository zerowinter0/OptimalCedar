# Final W-only PICO：交付与完成矩阵

commit: `315d51c3cc4eddd5edd556a746cfe8e72d371994`

## 身份

最终 PICO = `pico_final`（selector 39，`SimpleDpWorkersBoundaryAffineReprOptimizer`）：
联合搜索顺序/融合/后端/缓存与 W；**不搜索 stage width**（每个并行 stage 固定 width=1）；
计算项 = `k_(算子, 表示类)·元素数 + b_(算子, 表示类)`；边界项沿用字节模型。详见 `identity.json`。

## 完成矩阵（由已完成 cell 自动生成）

| 负载 | cells | optimizers |
| --- | ---: | --- |
| coco | main_fast | optimizer, pico_final |
| commonvoice | ablation_fast, main_fast | optimizer, pico_byte_proportional, pico_final, plumber_optimizer |
| llava_pretrain | main_fast | pico_final, plumber_optimizer, raydata_optimizer |
| simclrv2 | ablation_fast, dp_repeat, main_fast, staged_fast, w_cell_W1, w_cell_W16, w_cell_W4, w_cell_W64 | optimizer, pico_byte_proportional, pico_final, pico_final_no_boundary, plumber_optimizer, raydata_optimizer, simple_dp_workers_boundary, staged_final |
| simclrv2_cache | ablation_fast, main_fast | optimizer, pico_byte_proportional, pico_final, pico_final_no_boundary, plumber_optimizer, raydata_optimizer, simple_dp_workers_boundary |

## 吞吐（均值 ± 范围，samples/s）

| 负载 | cell | optimizer | 轮数 | 均值 | 范围 |
| --- | --- | --- | ---: | ---: | --- |
| coco | main_fast | optimizer | 1 | 30.5 | 30.5–30.5 |
| coco | main_fast | pico_final | 1 | 220.2 | 220.2–220.2 |
| commonvoice | ablation_fast | pico_byte_proportional | 1 | 714.4 | 714.4–714.4 |
| commonvoice | ablation_fast | pico_final | 1 | 714.3 | 714.3–714.3 |
| commonvoice | main_fast | optimizer | 1 | 194.6 | 194.6–194.6 |
| commonvoice | main_fast | pico_final | 1 | 709.2 | 709.2–709.2 |
| commonvoice | main_fast | plumber_optimizer | 1 | 145.5 | 145.5–145.5 |
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
| simclrv2 | w_cell_W1 | pico_final | 1 | 91.2 | 91.2–91.2 |
| simclrv2 | w_cell_W16 | pico_final | 1 | 1145.1 | 1145.1–1145.1 |
| simclrv2 | w_cell_W4 | pico_final | 1 | 342.2 | 342.2–342.2 |
| simclrv2 | w_cell_W64 | pico_final | 1 | 2272.2 | 2272.2–2272.2 |
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
| simclrv2 | 1 | pico_final | 91.2 | final PICO restricted to this single W candidate (structure re-optimised at that W); one complete run |
| simclrv2 | 16 | pico_final | 1145.1 | final PICO restricted to this single W candidate (structure re-optimised at that W); one complete run |
| simclrv2 | 4 | pico_final | 342.2 | final PICO restricted to this single W candidate (structure re-optimised at that W); one complete run |
| simclrv2 | 64 | pico_final | 2272.2 | final PICO restricted to this single W candidate (structure re-optimised at that W); one complete run |
| simclrv2 | 1 | simple_dp_boundary | 74.0 | complete run of the same plan family at this W (confounded with the plan structure; not a fixed-structure sweep) |
| simclrv2 | 64 | simple_dp_repr_affine | 1348.7 | complete run of the same plan family at this W (confounded with the plan structure; not a fixed-structure sweep) |
| simclrv2 | 64 | simple_dp_workers_width_boundary | 2404.0 | complete run of the same plan family at this W (confounded with the plan structure; not a fixed-structure sweep) |

局限：W=1 与 W=64 两组来自**不同计划结构**的完整运行（stage B），不是固定结构 W 扫描；固定结构扫描脚本已修好但本轮未跑成，写作时按上表标注。

## 规划成本与 DP 状态数（`planning.csv`）

| 负载 | cell | 算子数 | mask 数 | 最大状态数 | DP 秒 |
| --- | --- | ---: | ---: | ---: | ---: |
| coco | main_fast | 6 | 2 | 100 | 0.0 |
| commonvoice | ablation_fast | 7 | 1 | 120 | 0.0 |
| commonvoice | main_fast | 7 | 1 | 101 | 0.0 |
| llava_pretrain | main_fast | 16 | 462 | 182488 | 653.4 |
| simclrv2 | ablation_fast | 9 | 18 | 1342 | 2.6 |
| simclrv2 | dp_repeat | 9 | 18 | 369 | 2.5 |
| simclrv2 | main_fast | 9 | 18 | 369 | 2.3 |
| simclrv2 | main_mean | 9 | 18 | 369 | 2.3 |
| simclrv2 | staged_fast | 9 | 18 | 369 | 2.4 |
| simclrv2 | w_cell_W1 | 9 | 18 | 369 | 1.6 |
| simclrv2 | w_cell_W16 | 9 | 18 | 338 | 1.7 |
| simclrv2 | w_cell_W4 | 9 | 18 | 369 | 1.5 |
| simclrv2 | w_cell_W64 | 9 | 18 | 39 | 1.2 |
| simclrv2_cache | ablation_fast | 9 | 18 | 1489 | 2.6 |
| simclrv2_cache | main_fast | 9 | 18 | 497 | 2.3 |

## 2026-09-27 补测：CommonVoice 与 COCO（`scripts/pico_final_missing_cells_20260927.sh`）

这两个负载在此前所有轮次里都没跑出可用数据（CommonVoice 的 profile 只拟合出 1 条曲线、
COCO 的 profile 直接没产出）。根因与修复：

1. `payload_compute_scale` / `payload_representation_class` 没有 ndarray 分支，CommonVoice 的音频和 COCO 的框这两类 payload 无法拟合任何表示曲线（修复后分别得到 `np:<dtype>:<ndim>d` 类）；
2. COCO 的反事实上采样会在全分辨率图上爆内存/超时，新增`CEDAR_PROFILE_COMPUTE_MAX_ELEMENTS`（本轮 2e6）跳过超限反事实并记录；
3. CommonVoice 的脚本把数据集指向了 `cv-corpus-15.0-delta-2023-09-08/en/clips`（40,571 个 delta 片段，与标准 300k 训练集片段名零重叠），而项目标准协议用 `datasets/commonvoice/cv15_en_train_300000`；已改为标准数据集重跑，错误数据集下的产物归档在 `commonvoice/archive_wrong_dataset_20260927/`。

| 负载 | cell | optimizer | 数据量 | 稳态吞吐 (rec/s) | 优化+启动 (s) |
| --- | --- | --- | ---: | ---: | ---: |
| coco | main_fast | optimizer | 20000 | 30.5 | 18.9 |
| coco | main_fast | pico_final | 20000 | 220.2 | 10.4 |
| commonvoice | ablation_fast | pico_final | 100000 | 714.3 | 22.6 |
| commonvoice | ablation_fast | pico_byte_proportional | 100000 | 714.4 | 22.7 |
| commonvoice | main_fast | optimizer | 100000 | 194.6 | 21.6 |
| commonvoice | main_fast | pico_final | 100000 | 709.2 | 22.3 |
| commonvoice | main_fast | plumber_optimizer | 100000 | 145.5 | 2.4 |

结论（1 轮，属快速协议）：CommonVoice 上最终 PICO 709.2 / 714.3 rec/s，Cedar 194.6、Plumber 145.5；COCO 上最终 PICO 220.2、Cedar 30.5。
同负载内两个 cell 的 PICO 复现差 0.7%（CommonVoice），远小于与基线的 3.6–7.2 倍差距，因此这两个负载上的排序不依赖单轮噪声。
COCO 的早期低吞吐是 64 个 worker 的启动瞬态，稳态段才计入 `perf_time_sec`。

## 失败与负结果（必须保留）

- 失败/未完成 cell（原样保留，不冒充成功）：wikitext103/main_fast.failed.json
- `wikitext103`：最终 PICO 无法估价——profile 只覆盖 9 个算子中的 5 个（算子 2/3/4/5 被 torchtext 变换以 `TypeError: Input type not supported` 拒绝，池内含 float32:2d/int64:1d/list/text 四种表示仍不可测），优化器按设计直接报错而不是回退到字节模型。文本负载结论仍只能引用历史反例。
- `llava_pretrain`：计划覆盖校验未接入（需要该负载自己的 feature builder），只能报告吞吐，不能报告全覆盖验证。
- 语义：移动 `to_float` 会改变 torchvision 的值域解释（float 图像 clamp 到 [0,1]），
  SimCLRv2 上不存在语义等价重排；重排吞吐差不得写成等价优化收益（`semantic_scope.md`）。
- 完整 PICO 上字节 affine 与表示感知模型的吞吐落在同一范围的运行噪声内（simclrv2 2447.5 vs 2432.9 samples/s，极差 1.6%；cache 5.4%）→ 不宣称吞吐提升，
  只主张计划成本估计更准（第三章 MAPE 41.2% → 13.5%）。
- 联合 oracle 的覆盖范围：顺序×融合×W 轴在 INPROCESS 子空间全枚举（576 候选，DP = 枚举 = 3.4772573，`oracle_results.csv`）；
  含 offload 的联合枚举仍只是探针（DP 找到 SMP 计划 3.0125），不宣称全空间最优。
- 固定结构的 W 扫描仍缺：`w_scaling.csv` 是每个 W 重新联合优化的部署口径曲线。

## 复现

```bash
bash scripts/pico_final_all_20260924.sh        # profiles -> smoke -> campaign -> W scaling -> assembly
python -u tmp_analysis/assemble_final_delivery.py   # 任何时候重跑都可以重建本目录的表格
```