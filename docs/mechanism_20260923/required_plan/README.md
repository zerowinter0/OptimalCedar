# 构造计划：Cedar cost 更低、吞吐更高、顺序更差（SimCLRv2，2026-09-23）

目标：找出（并构造）一个 SimCLRv2 plan，同时满足

1. 最终 Cedar cost < cedar plan 的 8.4753（`Optimizer.calculate_cost`，每副本/每记录）
2. 实测稳态吞吐 > cedar plan（同场对照 1226.9 rec/s；campaign 参考 1187.7）
3. 算子顺序与 cedar plan（G→C→[B,H,J RAY]→F→N）不同
4. **仅 reorder（无 fuse、全 local）的 Cedar cost 明显更高**（要求 > 11，cedar 为 10.5481）

## 构造方法

1. 用 `scripts/score_simclrv2_orders.py` 扫顺序：把 `to_float`(7) 提前会让后续算子按 float32 尺寸计价
   （模型里 4× 字节），reorder 代价可到 16.8–18.4。
2. 用 `scripts/enumerate_simclrv2_structures.py` 对给定顺序枚举"分段 × 每段后端"
   （INPROCESS/SMP/RAY 的单段或融合块），全部用真实 `calculate_cost` 定价。
3. 筛选 `final < 8.4753` 且结构合理的组合（1 个本地融合块 + 1 个 2–4 成员 SMP 块），
   用 `scripts/build_simclrv2_plan.py` 生成可执行 plan，再用
   `scripts/run_fixed_plan_throughput.py` 实测（189,380 条 = imagenette2 train ×20 epoch，
   batch 4，W=64，同一 profile，2 轮）。

## 结果

cedar 同场对照（同一份 campaign plan 文件，2 轮）：1226.9 rec/s（1206.2 / 1247.6）。

| 计划 | 顺序 | 结构 | reorder-only cost | 最终 cost | 吞吐 (rec/s) | 四条 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| **CFGBHJN_w64** | C F G B H J N | `Fused{6,7,3}[INPROCESS] → Fused{2,5,4,1}[SMP w=1]` | **17.1171** | **8.1422** | **2010.9**（2007.0/2014.8） | ✓ |
| FCGBJHN_w64 | F C G B J H N | `Fused{7,6,3}[INPROCESS] → Fused{2,4,5,1}[SMP]` | 18.4425 | 8.1526 | 1990.2（1977.4/2003.1） | ✓ |
| FCGBHJN_w64 | F C G B H J N | `Fused{7,6,3}[INPROCESS] → Fused{2,5,4,1}[SMP]` | 18.4425 | 8.1526 | 1969.5（1960.9/1978.2） | ✓ |
| GCFBHJN_w64 | G C F B H J N | `Fused{3,6,7}[INPROCESS] → Fused{2,5,4,1}[SMP]` | 16.7665 | 8.1449 | 1953.7（1963.3/1944.1） | ✓ |
| cedar_reference_w64 | G C B H J F N | `Fused{2,5,4}[RAY w=1]` | 10.5481 | 8.4753 | 1226.9 | 参考 |

推荐 **CFGBHJN_w64**：reorder-only 比 cedar 高 **62%**（17.1171 vs 10.5481，且 > 11），
最终 cost 低 **3.9%**（8.1422 vs 8.4753），实测吞吐高 **64%**（2010.9 vs 1226.9 同场；campaign 参考 1187.7 → +69%）。

## 为什么这不是自相矛盾

- reorder 代价只由**顺序 + 每算子输入尺寸**决定；把 `to_float` 提前让 5 个下游算子按 float32 计价，
  所以 cedar 认为这个顺序明显更贵。
- 最终代价里，B/H/J/N 被放进 SMP 融合块：Cedar 对这些成员的 offload 计价做 Amdahl 反演并**clip 到 0**，
  融合块再乘 I/O 折扣，于是"更贵的顺序"反而得到更低的最终估价。
- 实测吞吐由结构决定：本地融合块 + 单 SMP 块（与 campaign 里 `dp-boundary` 的形态同族，
  其实测 1937.5 rec/s）远快于 cedar plan 的 `RAY w=1` 融合块。

## 产物

- `plans/*.yaml`：4 个构造 plan + cedar 参考（`plans/cedar_reference_w64.yaml`）
- `scores/*.json`：每个 plan 的最终 cost 与 reorder-only cost
- `throughput/*.json` + `logs/`：逐轮实测（189,380 条/轮，2 轮）
- `required_plan_summary.json`：汇总与四条布尔判定

复现：

```bash
# 容器内，先 source env/bin/activate
python -u scripts/score_simclrv2_orders.py --order CFGBHJN FCGBHJN
python -u scripts/enumerate_simclrv2_structures.py --order CFGBHJN --top 10
python -u scripts/build_simclrv2_plan.py --order CFGBHJN \
    --structure INPROCESS:6,7,3 SMP:2,5,4,1 --workers 64 \
    --out outputs/simclrv2_required_plan_20260923/plans/CFGBHJN_w64.yaml
python -u scripts/run_fixed_plan_throughput.py --plan outputs/.../plans/CFGBHJN_w64.yaml ...
```
