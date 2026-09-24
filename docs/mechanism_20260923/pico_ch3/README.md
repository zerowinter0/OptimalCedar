# PICO 第三章（Fine-Grained Cost Modeling）实验依据（2026-09-24）

统一结果目录。回答三个问题：

* **3.1** 整体输入缩放（比例规则）与整体融合折扣（固定 ρ）的局限；
* **3.2** 固定计算与输入相关计算的分离（affine 是否比比例规则更准、能否迁移到完整重排计划）；
* **3.3** 算子计算与执行边界成本的分离（分项模型是否比整体折扣稳定）。

**纪律**：先剖析并冻结模型，再执行/读取待预测计划；不使用目标计划执行时间校准其预测；
串行服务时间与并行吞吐分开报告；重复以完整运行（3 轮交错）为单位；失败/超时/零截断原样保留。

## 0. 审计结论（本节先于其它结果）

1. **折扣重算（修正先前手写常数）**：Cedar 的 `_calculate_cost_fused` 对 3 个等大小
   （in = out = s）成员计 `IO_U = s + 2s + 2s + s = 6s`、`IO_F = 2s` → **ρ = 1/3**。
   此前 `fusion_discount_harness.py` 手写 0.5（按 4s/2s）是错的；本次全部按真实字节重算，
   **原始测量不变**，只更正派生预测与误差。部分融合按组算：融合 A+B 的 ρ = 0.5，
   未融合的 C 保留其自身贡献，不把局部折扣乘到整块。
2. **模型身份**：本报告的 PICO = `SimpleDpWorkersWidthBoundaryOptimizer`（selector 27，
   `simple-dp-boundary-affine-W-width`）。组成逐项记录在 `audit.json::model_identity`：
   本地 affine 锚点（`MyOptimizer._dp_affine_value`）、共同执行修正（`_dp_co_run_factor`）、
   后端隔算锚点（`offloads.<BACKEND>.<pipe>.backend_compute`）、宽度缩放
   （`_dp_pipe_cost_at_parallelism`）、固定+字节边界（`_unscaled_boundary_transfer_ms`
   + `physical_model.boundary`）、W 与切片（`_worker_resource_groups`）、加性目标
   （`_SharedCommunicationObjective` = local_serial + ray_serial，系统代价 = 目标/W）。
   **不含** max-lane/overlap/GPU 模型；Cedar 原生估价与扩展计划回放的区别也在该节列出
   （materialized fused 定价与 INPROCESS 早退属于扩展路径，从不当作原版行为）。
3. **测量窗口**：批级求和后再相减、每记录只除一次记录数，逐批断言
   `计算 + 其他开销 = 端到端`；“其他开销”不称为网络时间；Ray `get` 含 actor 计算，
   不与成员计算相加；插桩与无插桩结果分列。**输出一致性检查的覆盖**：fusion harness
   每个 cell 只比对 **1 批（4 条记录）**，机制实验同样每配置 1 批，重排实验**没有**逐位比对
   ——因此不能宣称“全量一致”。

## 1. 3.1 固定 ρ 的局限（无插桩主实验 + 修正后的规则）

受控三算子块（A→B→C，同 shape/dtype/布局，float32 1×244×244 = 238,144 B/条；每个算子把同一
确定性变换重复 K 次并返回最后一次结果，输出逐位一致）。**主测量关闭成员计时**，U=3 个 Ray 阶段、
F=1 个融合 Ray 阶段，3 轮交错、每轮 120 批、单批在飞、所有 actor 绑同一物理核、单线程。

| K | 无插桩 T_U | 无插桩 T_F | 实测保留 T_F/T_U | **ρ（修正后 1/3）** | 规则 T_U×ρ | 规则误差 | 插桩保留 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32.388 | 14.533 | 0.449 | 0.3333 | 10.796 | **−25.7%** | 0.430 |
| 4 | 45.246 | 20.848 | 0.461 | 0.3333 | 15.082 | **−27.7%** | 0.423 |
| 16 | 71.214 | 46.418 | 0.652 | 0.3333 | 23.738 | **−48.9%** | 0.543 |
| 64 | 170.597 | 163.513 | 0.958 | 0.3333 | 56.866 | **−65.2%** | 0.907 |

单位 ms/记录；T_U/T_F 为 3 轮均值，逐轮值见 `measurements.csv`。
**ρ 固定不变，实测保留从 0.449 涨到 0.958**，而规则误差从 −26% 扩大到 −65%
（即固定折扣严重高估融合收益）。插桩对照：相同 cell 的插桩运行给出 0.430/0.423/0.543/0.907，
与无插桩结论方向一致（差异集中在重档，见 `figure_data.json` 的 `instrument_comparison`）。

## 2. 3.3 分项模型（计算 + 固定/字节边界）比整体折扣稳定

分项模型：`compute = 隔算锚点 × (k·x+b)/(k·x_ref+b) × co_run`；
`boundary = fixed_latency + (in+out)/throughput`（每个 Ray 阶段一次）。参数来源：
合成块 = 独立剖析（单 actor、CPU 绑核、单线程，K 档各 192 次调用，锚点
0.564/2.397/8.486/32.684 ms/记录）；真实块 = 冻结 profile
（fixed 7.827 ms、94.58 MB/s、R²=0.996）。

**受控块（与上面的实测对照）**

| K | 实测 T_F | Cedar 固定 ρ 规则 | 误差 | 分项模型 | 误差 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 14.533 | 10.796 | −25.7% | 14.553 | **+0.1%** |
| 4 | 20.848 | 15.082 | −27.7% | 20.053 | **−3.8%** |
| 16 | 46.419 | 23.738 | −48.9% | 38.320 | **−17.4%** |
| 64 | 163.513 | 56.866 | −65.2% | 110.915 | **−32.2%** |

**真实块 `to_float → RandomResizedCrop → RandomHorizontalFlip`**（声明顺序，全 Ray，3 轮）

| 组织 | 实测（串行） | 分项模型 | 误差 | Cedar 固定 ρ 规则 | 误差 | actor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| U（3 阶段） | 121.283 | 106.906 | −11.9% | — | — | 3 |
| P（融合 7+6） | 51.110 | 48.536 | **−5.0%** | — | — | 2 |
| F（融合 7+6+5） | 28.404 | 25.602 | −9.9% | 21.156 | **−25.5%** | 1 |

真实块 ρ（真实字节 7,521,267 → 1,311,984）= **0.1744**。分项模型在 4 个计算档与真实块上都比
固定折扣稳定；在高计算档仍系统性低估（隔算锚点低于同上下文中的成员计算），这一点单独报告。

## 3. 3.2 affine 与比例规则的独立验证

数据：既有的全本地、无融合、无卸载、逐记录 trace 的四种顺序计划（W=4；
`unopt_order_transfer_repeats_traceall_20260921`）。**实验一**：以声明顺序的实测为共同锚点，
比较比例规则 `t_ref·(x/x_ref)` 与 affine 规则 `t_ref·(k·x+b)/(k·x_ref+b)`。
**实验二**：直接用冻结 profile 的实际 PICO 计算（affine 绝对拟合，无锚点）。

| 计划 | 实测总计算 | 比例规则 | affine 规则 | PICO（绝对） | MAPE 比例 | MAPE affine | MAPE PICO |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| declared | 23.288 | 23.288 | 23.288 | 25.454 | 0%（锚点） | 0%（锚点） | 24.7% |
| PICO 顺序 | 13.994 | 6.082 | 7.876 | 8.190 | 64.1% | **53.4%** | 64.9% |
| Cedar 顺序 | 14.057 | 6.012 | 7.989 | 8.229 | 62.3% | 70.5% | 77.0% |
| old-dp 顺序 | 59.523 | 124.376 | 120.792 | 149.226 | 147.5% | 156.0% | 181.2% |

（单位 ms/记录，MAPE 取 6 个尺寸变化算子。）

**要点（含负面结果）**：

* affine 相对比例规则**只在 PICO 顺序上更好**（53.4% vs 64.1%），在 Cedar 顺序上更差（70.5% vs 62.3%），
  在 old-dp 顺序上接近（156% vs 147.5%）→ “affine 普遍优于比例规则”**不成立**，只能说在部分顺序上更好。
* 两者都把重排带来的变化**过度修正**（预测 6–8 ms vs 实测 14 ms；old-dp 预测 121–149 vs 实测 59.5），
  即模型把每个算子当独立过程，缺少流水线上下文项。
* 排序：实测 PICO(13.994) < Cedar(14.057) < declared(23.288) < old-dp(59.523)。
  affine 规则给出一致排序；**比例规则把 PICO/Cedar 前两名弄反**；PICO 绝对预测排序正确但误差更大。

## 4. 3.4 完整流水线（W=1，块外计划不变，4,000 条/轮 ×3 轮）

| 计划 | 实测吞吐 | 实测加速 | PICO 预测加速 | 误差 | Cedar 预测加速 | 误差 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| U | 17.35 rec/s | 1.000 | 1.000 | — | 1.000 | — |
| P | 46.99 rec/s | 2.709 | 1.812 | −33.1% | 1.765 | −34.8% |
| F | 52.64 rec/s | 3.034 | 2.662 | **−12.3%** | 1.838 | **−39.4%** |

两者都给出正确排序（U < P < F）；PICO 的幅度在 F 上明显更准（−12% vs −39%），P 上两者都低估约 1/3。
PICO 目标是排序分数（系统代价 = 目标/W），不是墙上时钟吞吐预测；该比较只用于排序与幅度检验。
块外模型贡献在三个计划中相同（Cedar 21.0886），但**这不能证明实测差异全部来自块内**
（融合同时改变 actor 数与重叠结构）；本轮未做块外直接测量。

## 5. 交付文件

| 文件 | 内容 |
| --- | --- |
| `audit.json` | 折扣重算、PICO 模型身份（含文件/函数位置）、测量窗口与一致性覆盖 |
| `protocol.json` | 执行前确定的 cell、参数、重复与停止规则；profile 校验值 |
| `predictions.csv` | 所有模型预测（含锚点、单位与来源） |
| `measurements.csv` | 每轮/每 cell 实测值与跨轮标准差 |
| `operator_results.csv` | 逐算子：调用数、输入字节、参考成本、比例/affine/PICO 预测、实测、误差 |
| `reorder_summary.csv` / `reorder_analysis.json` | 每个重排计划的总量与排序 |
| `boundary_results.csv` | 边界路径、固定/字节分项、每记录预测 |
| `summary.csv` | 各模型误差汇总（A/B/C/D 四组） |
| `pipeline_scoring.csv` / `.json` | 完整流水线 PICO/Cedar 代价、预测与实测加速、排序 |
| `component_predictions.json` | 分项模型（合成块按 K、真实块按组织） |
| `profile/synthetic_anchor.json` | 独立剖析得到的每 K 档计算锚点 |
| `expA1_main/` | 无插桩主实验逐批/逐轮数据（`expA1_raw.csv`、`expA1_summary.csv`） |
| `figure_data.json` | 第三章各图表直接数据 |
| `MANIFEST.json` | 大文件路径、行数、sha256 |

复现（容器内，先 `source env/bin/activate`）：

```bash
RUN=outputs/pico_ch3_20260924
# 1) 无插桩主实验（3 轮交错）
python -u scripts/fusion_discount_harness.py run --run-dir $RUN/expA1_main --label expA1 \
    --instrument none --levels 1 4 16 64 --rounds 3 --batches 120 --warmup 40 \
    --local-cpu 12 --remote-cpu 8
# 2) 审计与折扣重算
python -u scripts/pico_ch3_audit.py --run-dir $RUN --profile <profile> --out $RUN/audit.json
# 3) 独立剖析 + 分项预测
python -u scripts/pico_ch3_component_model.py profile-synthetic --out $RUN/profile/synthetic_anchor.json
python -u scripts/pico_ch3_component_model.py predict --run-dir $RUN --profile <profile> \
    --anchor $RUN/profile/synthetic_anchor.json --out $RUN/component_predictions.json
# 4) 完整流水线 PICO/Cedar 预测
python -u scripts/pico_ch3_pipeline_scoring.py --run-dir $RUN \
    --fusion-run outputs/fusion_discount_20260923 --profile <profile>
# 5) 重排迁移验证（复用既有 trace）
python -u scripts/pico_ch3_reorder_analysis.py --run-dir $RUN \
    --trace-run outputs/unopt_order_transfer_repeats_traceall_20260921 --profile <profile>
# 6) 汇总交付
python -u scripts/pico_ch3_expA_analysis.py --run-dir $RUN \
    --instrumented-run outputs/fusion_discount_20260923/expA \
    --instrumented-off-run outputs/fusion_discount_20260923/expA_instrument_off --audit $RUN/audit.json
python -u scripts/pico_ch3_assemble.py --run-dir $RUN
python -u scripts/sync_pico_ch3_artifacts.py --run-dir $RUN --dest docs/mechanism_20260923/pico_ch3
```

## 6. 未完成 / 明确不做

* 未做 W=64 与多负载扩展（按指示本轮不需要）。
* 未做块外工作的直接测量；因此“误差全部来自块内”的推断被明确保留为未验证。
* 重排实验没有逐位输出比对（只比逐算子时间），已在该文件与 `audit.json` 中声明。
* 成员计算在高计算档的系统性低估（隔算锚点 vs 同上下文）仍未定位到具体机制。
