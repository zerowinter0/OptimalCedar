# 执行计划与预计工作量（最终 W-only PICO）

## 阶段与 cell 计数

| 阶段 | 内容 | cell 数 | 1 cell 预计 | 总预计 |
| --- | --- | ---: | ---: | ---: |
| 0 | smoke gate（simclrv2，900 样本，5 optimizer，1 轮） | 1 | ≈10 min | ≈10 min |
| 1 | profile 重生成（simclrv2_cache / commonvoice / coco / llava_pretrain / wikitext103；simclrv2 已有） | 5 | 20–60 min | 2–5 h |
| 2a | 主对比 `main_mean`（pico_final, cedar, plumber, raydata, unopt；3 轮） | 6×5×3 = 90 | COCO/CommonVoice 100–1800 s、其余数分钟 | 10–30 h |
| 2b | 模型消融 `ablation_model`（pico_final, no-boundary, byte-proportional, byte-affine；3 轮） | 6×4×3 = 72 | 同上 | 8–24 h |
| 2c | staged vs joint `staged_joint`（pico_final, staged_final, staged byte-affine；3 轮） | 6×3×3 = 54 | 同上 | 6–18 h |
| 3 | W scaling（simclrv2，W∈{1,4,16,64}，3 轮） | 4×3 = 12 | 2–10 min | 1–2 h |
| 4 | 组装（`assemble_final_delivery.py`） | — | <5 min | <5 min |

合计约 **230 个执行 cell**，按本机 + 远端 Ray 的实测速度估计 **30–70 小时**。
所有 cell 都可续跑：结果文件存在即跳过。

## 续跑与停止规则

- 每个 stage 先检查结果文件再执行；`bash scripts/pico_final_all_20260924.sh` 可重复调用。
- 规划上限 3600 s/cell、执行上限 7200 s/cell（可用 `PLAN_TIMEOUT`/`CELL_TIMEOUT` 覆盖），
  超时按原样记 failed，不放宽。
- 缺少带 `compute_model` profile 的 workload 会被跳过并写明原因，不用字节模型冒充最终 PICO。
- 组装脚本可在任何时刻重跑，从已完成 cell 重建
  `throughput.csv`、`ablation.csv`、`search_comparison.csv`、`planning.csv`、
  `ranking.csv`、`README.md`、`claim_evidence.md`、`MANIFEST.json`。
- 本轮的语义结论（SimCLRv2 无语义等价重排）已冻结，不因吞吐结果改变；
  吞吐一律标注为"合法计划"口径。

## 已知未覆盖项（写在交付里，不用它冒充完成）

1. **联合 oracle（顺序×融合×后端×W 的独立枚举）**：顺序×W 轴与
   `state_sufficiency.json` 已完成；plan-replay 入口对人为构造的计划抛
   `DP objective scoring requires one linear source`，联合枚举被阻塞（原始错误见
   `oracle_results.json`）。
2. **Stage E 的合成规模曲线（n=4…16，稀疏/中等/稠密依赖）**：未跑；
   现有可替代证据是各负载真实算子数下的 DP 状态与规划时间（`planning.csv`，来自 cell 日志）。
3. **StackExchange**：本轮未纳入（profile 与执行都需要额外输入准备）；如需附表需单独补跑。
4. **训练质量对照**：按最终决定不做；语义差异只报告、不外推。
