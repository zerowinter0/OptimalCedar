# 主张 → 证据（最终 W-only 交付）

| 主张 | 实验 | 原始路径 | 关键数值 | 可用范围 |
| --- | --- | --- | --- | --- |
| 字节不是计算规模的充分统计量 | 表示/几何受控对照 | `docs/mechanism_20260923/affine_reorder/blur_geometry.json` | 同约 6 万字节：0.850 vs 2.606 ms | 仅该负载的算子族 |
| 元素+表示类显著降低计划成本误差 | M1–M5 消融（11 计划 × 3 轮） | `outputs/affine_repr_model_20260924/` | MAPE 41.2% → 13.5% | 计划计算量口径，非吞吐 |
| 截距 b 无独立贡献 | M4 vs M5 | 同上 | 13.6% → 13.5% | 同负载 |
| DP 在顺序×W 空间最优 | 1260 顺序独立参考 | `outputs/affine_reorder_diagnosis_20260924/dp_optimality_test.json` | 命中 argmin (11.717960) | 该受限空间 |
| 表示状态可由算子集合决定 | 前缀一致性 + 实测 payload | `outputs/pico_final_w_only_20260924/state_sufficiency.json` | 71 个 mask 无冲突 | SimCLRv2 DAG |
| 最终 PICO 的吞吐 | 主对比（每 cell 3 轮） | `throughput.csv` | 见 README 表 | 合法计划，非等价语义 |