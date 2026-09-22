# 正式实验结果图（2026-09-21）

由 `scripts/make_experiment_figures_20260921.py` 从 `outputs/figure_data_20260921/figure_data.json`
（`scripts/collect_figure_data_20260921.py` 采集）直接生成，每个图都有 `.png`（300 dpi 预览）、
`.pdf` 与 `.svg`（论文排版用）。

数据来源：正式放大 campaign `outputs/ultimate_eight_optimizers_fix_20260921`（simclrv2 189,380 条、
commonvoice 300,000、coco 50,000、llava_pretrain 43,940 条过过滤），llava 的 PICO 取 W-only 运行
`outputs/pico_w_only_20260921`。口径与注意事项见 `docs/experiments.md` §5。

**图中不含 stackexchange**（按要求去掉）；它的吞吐/优化时间/cost 数据仍完整保留在
`docs/experiments.md` §5 与 `outputs/figure_data_20260921/figure_data.json`。

## fig1_throughput

每个负载 × 8 个 optimizer 的**稳态吞吐**（records/s；**每个负载一个面板、各自独立线性纵轴**——
负载之间跨度 3.1–5{,}502 rec/s，共用一根线性轴会把小柱子压平；柱顶数字为实测值）：
`unopt / plumber / raydata / cedar / cedar-dp / PICO-Resource / PICO-Resource-Op / PICO`。
红叉 = 该 cell 没有结果（unopt 在 commonvoice/coco 执行超时；cedar 在 llava/stackexchange 因已知优化
超时跳过）。

要点：PICO 在 5 个有结果的负载里 4 个第一（simclrv2 2501.5、commonvoice 743.2、coco 294.3、
llava 29.4 rec/s），simclrv2-cache 上 PICO-Resource-Op 5502.4 > PICO 5399.4。

## fig2_optimization_time

同名 8 个 optimizer 的**优化/构建时间**（`setup_time_sec`），**纵轴是等差（线性）**，PICO 带斜纹。
由于 llava 的 PICO 要 1,938 s，单用一根线性轴会把其余柱子压平，所以画成上下两块、都是线性轴：
上面板 0–2,000 s（全量），下面板 0–460 s（放大，柱顶数字只在放大面板标全）。
红叉 = 无 cell：`unopt@commonvoice/coco`（执行超时）、`cedar@llava_pretrain`（已知 Cedar 优化超时）。

要点：PICO 的规划时间显著更高（simclrv2 267 s、simclrv2-cache 418 s、llava 的 W-only 1,938 s），
其余 optimizer 都在 100 s 以内；cedar 在 simclrv2 上只要 22 s，但在 llava 上跑不出来。

## fig3_cost_rank

每个负载一个面板：**实测吞吐排名**（x，1 = 最快）vs **cost model 给出的 cost 排名**
（y，1 = 最便宜），三个模型 `cedar`（圆）、`plumber`（方，已按 plan 的 W 折算）、`PICO`（三角，
`S/W`）。虚线 = 完全一致；点越靠近虚线，模型排序越贴近真实。

要点：PICO 在 simclrv2 / simclrv2-cache / commonvoice / coco 上几乎在对角线上（ρ = 0.98 / 0.99 /
0.96 / 1.00），llava 0.61；cedar 的等比例模型在 commonvoice 上甚至负相关（ρ = −0.16）。

## fig4_rank_agreement

把 fig3 的结论压成一张汇总图：每个负载上三个模型的 Spearman ρ（cost 排名 vs 速度排名）。

要点：去掉 stackexchange 后 PICO 在**全部 5 个负载**上最好（0.98 / 0.99 / 0.95 / 1.00 / 0.61），
plumber 次之（0.83 / 0.92 / 0.63 / 0.44 / 0.11），cedar 的等比例模型最差（0.41 / 0.25 / −0.16 /
0.13 / 0.09）。

## 复现

```bash
# 容器内
python -u scripts/collect_figure_data_20260921.py        # 采集数据 → outputs/figure_data_20260921/
python -u scripts/make_experiment_figures_20260921.py    # 出图 → outputs/figures_20260921/
```

渲染产物同时归档在本目录（`.png` / `.pdf` / `.svg`）。
