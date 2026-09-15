# 正式矩阵结果快照（simple-DP 单列）

来源：outputs/pico_formal_20260915（9 负载 × 7 planner，full-pass 排空测量，每格 1 次）。
生成：python tmp_analysis/paper_table.py outputs/pico_formal_20260915

### 表 1：PICO vs 外部系统（吞吐，源记录/秒，越高越好）

| 负载 | 源记录 | Cedar | DJ-Cedar | Pecan-Cedar | Plumber | Ray-Data | **PICO** | 最优外部 | PICO 加速比 |
|---|---|---|---|---|---|---|---|---|---|
| simclr | 9469 | 455 | 158 | 458 | 884 | 174 | **1737** | Plumber 884 | **1.96×** |
| blip | 1000 | 194 | 198 | 178 | 1939 | 195 | **2079** | Plumber 1939 | **1.07×** |
| clip | 1000 | 198 | 181 | 184 | 901 | 194 | **905** | Plumber 901 | **1.00×** |
| dino | 1000 | — | 88 | 83 | 1038 | 83 | **812** | Plumber 1038 | **0.78×** |
| alpaca_cot | 74771 | 2136 | 2045 | 2004 | 521 | 1267 | **2056** | Cedar 2136 | **0.96×** |
| pile_hackernews | 100000 | — | 51 | 47 | 52 | — | — | Plumber 52 | — |
| pile_pubmed_abstracts | 100000 | — | 276 | 281 | 189 | 162 | **283** | Pecan-Cedar 281 | **1.01×** |

### 表 2：ablation 单列（同一联合搜索 + Cedar 原始 cost model）

| 负载 | simple-DP (ablation) | 最优外部 | ablation 加速比 | PICO 加速比 |
|---|---|---|---|---|
| simclr | 1695 | Plumber 884 | 1.92× | 1.02× |
| blip | 1867 | Plumber 1939 | 0.96× | 1.11× |
| clip | 981 | Plumber 901 | 1.09× | 0.92× |
| dino | 943 | Plumber 1038 | 0.91× | 0.86× |
| alpaca_cot | 1343 | Cedar 2136 | 0.63× | 1.53× |
| pile_hackernews | 37 | Plumber 52 | 0.71× | — |
| pile_pubmed_abstracts | — | Pecan-Cedar 281 | — | — |

## 尚未完成的格子

- pile_hackernews：PICO / Ray-Data 两格在建 Ray actor 时超时失败；Cedar 那格自身 reorder 超过 900 s 弃权（perf=inf）。
- pile_pubmed_abstracts：simple-DP（ablation）正在运行。
- pile_uspto_backgrounds / bloom_oscar：尚未开始。
