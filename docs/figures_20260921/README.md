# 正式实验结果图（2026-09-21）

由 `scripts/make_experiment_figures_20260921.py` 从 `outputs/figure_data_20260921/figure_data.json`
（`scripts/collect_figure_data_20260921.py` 采集）直接生成，每个图都有 `.png`（300 dpi 预览）、
`.pdf` 与 `.svg`（论文排版用）。

数据来源：正式放大 campaign `outputs/ultimate_eight_optimizers_fix_20260921`（simclrv2 189,380 条、
commonvoice 300,000、coco 50,000、llava_pretrain 43,940 条过过滤、stackexchange 7,238 条），
llava 的 PICO 取 W-only 运行 `outputs/pico_w_only_20260921`。口径与注意事项见
`docs/figure_data_20260921.md`。

## fig1_throughput

每个负载 × 8 个 optimizer 的**稳态吞吐**（records/s，log 轴，柱顶数字为实测值）：
`unopt / plumber / raydata / cedar / cedar-dp / PICO-Resource / PICO-Resource-Op / PICO`。
红叉 = 该 cell 没有结果（unopt 在 commonvoice/coco 执行超时；cedar 在 llava/stackexchange 因已知优化
超时跳过；PICO 在 stackexchange 规划超时）。

要点：PICO 在 5 个有结果的负载里 4 个第一（simclrv2 2501.5、commonvoice 743.2、coco 294.3、
llava 29.4 rec/s），simclrv2-cache 上 PICO-Resource-Op 5502.4 > PICO 5399.4。

## fig2_optimization_time

同名 8 个 optimizer 的**优化/构建时间**（`setup_time_sec`，log 轴）。PICO 带斜纹。
红叉 = 无 cell：`unopt@commonvoice/coco`（执行超时）、`cedar@llava/stackexchange`（已知 Cedar 优化超时）、
`PICO@stackexchange`（规划超 2 h 上限）。

要点：PICO 的规划时间显著更高（simclrv2 267 s、simclrv2-cache 418 s、llava 的 W-only 1,938 s），
其余 optimizer 都在 100 s 以内；cedar 在 simclrv2 上只要 22 s，但在 llava/stackexchange 上跑不出来。

## fig3_cost_rank

每个负载一个面板：**实测吞吐排名**（x，1 = 最快）vs **cost model 给出的 cost 排名**
（y，1 = 最便宜），三个模型 `cedar`（圆）、`plumber`（方，已按 plan 的 W 折算）、`PICO`（三角，
`S/W`）。虚线 = 完全一致；点越靠近虚线，模型排序越贴近真实。

要点：PICO 在 simclrv2 / simclrv2-cache / commonvoice / coco 上几乎在对角线上（ρ = 0.98 / 0.99 /
0.96 / 1.00），llava 0.61，stackexchange 0.37；cedar 的等比例模型在 commonvoice 上甚至负相关
（ρ = −0.16）。

## fig4_rank_agreement

把 fig3 的结论压成一张汇总图：每个负载上三个模型的 Spearman ρ（cost 排名 vs 速度排名）。

要点：PICO 在 5/6 个负载上最好（0.98 / 0.99 / 0.95 / 1.00 / 0.61），只在 stackexchange 上被 plumber
反超（0.37 vs 0.72，因为 plumber 在这条文本流水线上的瓶颈模型恰好贴合）。

## 复现

```bash
# 容器内
python -u scripts/collect_figure_data_20260921.py        # 采集数据 → outputs/figure_data_20260921/
python -u scripts/make_experiment_figures_20260921.py    # 出图 → outputs/figures_20260921/
```

渲染产物同时归档在本目录（`.png` / `.pdf` / `.svg`）。
