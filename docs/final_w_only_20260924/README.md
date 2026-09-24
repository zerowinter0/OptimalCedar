# 最终 W-only PICO：身份、完成矩阵与交付

本轮把 PICO 冻结为**只搜索 W**的最终配置，并给出第三章所需的证据链。
远端 campaign 在 tmux session `pico_final` 中运行（`scripts/pico_final_all_20260924.sh`），
结束后会自动重建本目录的表格与报告（`tmp_analysis/assemble_final_delivery.py`）。

## 1. 最终 PICO 是什么

| 项 | 值 |
| --- | --- |
| 名称 | `pico_final` |
| selector | 39 |
| 类 | `SimpleDpWorkersBoundaryAffineReprOptimizer` |
| 搜索 | 顺序 / 融合 / 后端 / 缓存 **联合**；W 由 W-conditioned DP 选择；**stage width 不搜索**（每并行 stage 固定 1，`_dp_candidate_parallelisms → (1,)`） |
| W 候选 | `1..min(replica_cap, local_budget//(1+reserve), ray_budget)`；每主机 64 CPU；不可行切片跳过；不缩小候选 |
| 计算模型 | `k_(算子, 表示类)·元素数 + b_(算子, 表示类)`（元素数 = `C·H·W` / `W·H·bands` / `len`） |
| 边界模型 | 沿用 Cedar/layered 字节边界项（未改） |
| 目标 | `DpObjectiveCost`：`score/W = 串行计算/W + 字节服务`，转移再加边界项 |
| profile | `physical_model.compute_model`（schema 1）+ 旧 `operator_affine`（基线读旧字段） |
| 基线 | `optimizer`=Cedar 原生分阶段；`plumber_optimizer`/`raydata_optimizer`=在 Cedar 优化器接口内的策略实现；`unopti`=声明计划直跑 |

完整字段与命令见 `identity.json`。消融身份：
`pico_final_no_boundary`(40)、`pico_byte_proportional`(41)、`simple_dp_workers_boundary`(25，
同 W-only 搜索 + 字节 affine)、`staged_final`(42，staged 搜索 + 同一最终模型)。

## 2. 完成矩阵

| 阶段 | 状态 | 证据位置 |
| --- | --- | --- |
| A1 冻结身份 | **完成** | `identity.json`；smoke gate 通过（900 样本：pico_final 1128.5 samples/s，5 个 optimizer 全部完成） |
| A2 状态充分性 | **完成** | `state_sufficiency.json`：SimCLR DAG 71 个可达 mask、1260 个合法前缀，无冲突；1260 顺序独立参考 argmin 命中 |
| A2 联合 oracle（顺序×融合×后端×W） | **部分**：顺序×W 轴完成；联合轴被 plan-replay 入口阻塞 | `oracle_results.json`（原始错误：`DP objective scoring requires one linear source`） |
| A3 运行器/基线审计 | **完成** | 见 §4；同 harness、同 profile、同资源上限、round-robin |
| B 主实验（6 负载 × 5 optimizer × 3 轮） | **运行中**（tmux `pico_final`） | `throughput.csv`，逐 cell 日志在 `outputs/pico_final_w_only_20260924/<workload>/logs/` |
| C1 模型消融（同 W-only 搜索） | **运行中** | `ablation.csv` |
| C2 staged vs joint | **运行中**（同一最终模型，selectors 39 vs 42） | `search_comparison.csv` |
| C3 W 实用性 | **运行中**（simclrv2：W∈{1,4,16,64}，固定结构，3 轮） | `w_scaling.csv`（由 `w_scaling/results_W*.json` 汇总） |
| D1 计算模型 | **完成** | `outputs/affine_repr_model_20260924/`：11 计划 × 3 轮，M1–M5 |
| D2 融合/边界 | **完成（复用）** | `fusion_components.csv`（真实块 U/P/F 三轮均值 121.539/51.558/28.314，成员计算 6.825/6.687/6.564，余项 114.714/44.872/21.749） |
| D3 完整计划排序 | **部分**：11 计划已评分；主实验的候选并集排序随 campaign 产出 | `ranking.csv`、`plan_predictions.csv` |
| E 规划成本/规模 | **部分**：真实负载的规划时间与 DP 状态数随 campaign 产出（`planning.csv`）；合成 n=4…16 规模曲线未跑 | `planning.csv`、`execution_plan.md` §未覆盖 |

## 3. 主要结果（截至冻结前已完成部分）

**计算模型（第三章核心）**——目标 = 完整运行中逐算子平均自身服务时间之和，11 个计划 × 3 轮：

| 模型 | 比值范围 | MAPE | RMSE |
| --- | --- | ---: | ---: |
| M1 字节比例 | 0.32–1.62 | 37.7% | 46.8% |
| M2 字节 affine | 0.33–1.28 | 41.2% | 44.5% |
| M3 元素 affine（不分类） | 0.87–1.81 | 19.5% | 30.0% |
| M4 表示感知过原点 | 0.70–1.09 | 13.6% | 17.1% |
| **M5 表示感知 affine（最终）** | 0.71–1.20 | **13.5%** | **15.8%** |

换自变量贡献最大（41.2%→19.5%），表示类次之（→13.5%），**截距 b 无独立贡献**（13.6%→13.5%）。

**已知的端到端事实（阶段 B 前身，1 epoch）**：
完整 PICO 下字节 affine 与表示感知模型的吞吐**不可区分**（2404 vs 2320 samples/s，
区间 2355–2468 与 2260–2372 重叠）；粗搜索变体 `simple_dp_boundary` 则从 74.0 /s
（W=1）跳到 1348.7 /s（W=64 + Ray blur）。后者是**模型诱导的 W 决策**，不是算子变快。

## 4. 基线与运行器审计

- 全部 cell 用同一个 harness（`evaluation/compare_optimizer_perf.py`）与同一份按负载冻结的
  profile；native 策略与 Cedar 原生优化器在结果表中用 `identity` 列区分，不混称。
- 资源：`CPU_BUDGET=64`、`RAY_CPU_BUDGET=64`、远端 Ray `cedar_remote`、线程数 1；
  每条结果记录实际 W、backend、stage width、融合分组。
- 计数：每条结果记录 `num_samples`（实际处理条数）；LLaVA 以源记录为单位计数，
  图像数另列（campaign 日志中的数据集统计）。
- 失败/超时按原样保留（`*.failed.json`），不当作成功复用；smoke gate 通过后才启动长实验。

## 5. 失败与负结果（必须随论文保留）

1. **语义**：`to_float` 是纯 cast，移动它会让 torchvision 把 float 图像 clamp 到 [0,1]
   （32 条记录实测 jitter/grayscale/blur 平均绝对差 101–104）。SimCLRv2 上**没有语义等价的重排**，
   因此所有重排吞吐差都只能写成"合法计划"口径，不能写成等价优化收益。
2. **完整 PICO 上没有吞吐提升**（2404 vs 2320，区间重叠）。
3. **联合 oracle 未完成**（plan-replay 入口阻塞）。
4. **合成规模曲线（n=4…16）未跑**。
5. **StackExchange 未纳入本轮**（需额外输入准备）。

## 6. 复现

## 5b. 7 小时预算下的协议放宽（用户批准，逐项记录）

为了把总时长压进 7 小时，`scripts/pico_final_fast_20260925.sh` 相对冻结协议做了以下放宽，
**每一项都改变了对"轮次/数据量"的主张强度，写作时必须按此措辞**：

| 放宽 | 内容 | 影响 |
| --- | --- | --- |
| 数据量 | simclrv2 / simclrv2_cache 8 epochs = **75,752** 条（原 189,380）；commonvoice **100,000**（原 300,000）；coco **20,000**；llava_pretrain **10,000** | 稳态吞吐与数据量无关，仍可比；但"大数据规模"只能写"够稳态"，不能写原规模 |
| 轮次 | 慢基线（plumber/raydata）**1 轮**；DP 对（pico_final/cedar）主 cell 1 轮 + 额外 repeat 1 次 | 只有 1–2 个样本，只能给点值；不做显著性声明 |
| 去项 | 所有负载去掉 `unopti`；commonvoice 去掉 `raydata` | 与最弱基线的对比缺失，README 里标注 |
| llava | **不跑 cedar**（稳定超时），只跑 pico_final / plumber / raydata，超时上限 1200 s | Cedar 在 llava 上的对比缺失，写明原因 |
| profile | coco / llava 用缩短窗口（compute 曲线 1.5 s/点、adaptive 2–10 s） | 这两份 profile 的曲线精度低于其他负载，标注 |

计划总时长 ≈ **3 小时**（预算 7 小时），此后自动组装；若中途某 cell 超时按原样记录。

```bash
# 容器内，先 source env/bin/activate
bash scripts/pico_final_all_20260924.sh          # profiles → smoke → campaign → W scaling → assembly
python -u tmp_analysis/assemble_final_delivery.py  # 任何时候重跑都会重建 CSV/JSON/README/MANIFEST
python -u tmp_analysis/state_sufficiency.py outputs/affine_repr_profile_20260924/shared.yaml
python -u tmp_analysis/oracle_joint_small.py outputs/affine_repr_profile_20260924/shared.yaml
```

进度查看：`docker exec optimalcedar-torch201-dev tmux capture-pane -pt pico_final | tail`
（或 `tail -f outputs/pico_final_w_only_20260924/campaign.log`）。
