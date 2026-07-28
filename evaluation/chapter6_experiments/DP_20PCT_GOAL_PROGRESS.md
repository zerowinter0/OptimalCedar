# DP 20%/30% 目标：总体任务、当前进展与后续工作

最后更新：2026-07-26（Asia/Shanghai）

## 1. 总体任务与成功判据

目标是在不降低正式实验参数、不删去失败负载、不按结果挑选负载的前提下，
从 Data-Juicer 的官方 recipe 中扩展与 Cedar 优化问题高度相关的负载，并改进、
验证 selection-aware 联合 DP optimizer。

最终判据为：在完整负载集合中，至少 30% 的负载上，`dp_optimizer` 的纯执行
时间比其余 in-scope optimizer 中最快者至少快 20%，即
`best_other_time / dp_time >= 1.20`。优化时间单独报告，不计入纯执行时间。

对手集合固定为：

- `optimizer`（原 Cedar）；
- `dj_optimizer`；
- `dp_cedar_optimizer`；
- `pecan_optimizer`。

按用户要求，`dp_two_stage_optimizer` 完全排除，不参与运行、最佳对手选择或
最终计数。

## 2. 冻结的实验范围与公平性约束

### 2.1 负载分母

现有正式结果包含 10 个负载：`coco`、`commonvoice`、
`commonvoice_cache`、`llava_pretrain`、`redpajama_c4`、`simclrv2`、
`simclrv2_cache`、`stackexchange`、`wikitext103`、
`wikitext103_cache`。

在查看新增负载的 optimizer 性能结果前，另行注册了 6 个 Data-Juicer
候选：`pile_europarl`、`redpajama_code`、`pile_hackernews`、
`pile_pubmed_abstracts`、`pile_freelaw`、`pile_uspto_backgrounds`。
失败、超时或低于 20% 的候选仍保留在分母中。因此最终分母固定为 16，
30% 需要至少 `ceil(0.30 * 16) = 5` 个胜出负载。

### 2.2 正式配置

- `W=8`，`CPU_BUDGET=64`；
- profile 中每个 Ray actor / SMP process 的宽度均为 1，每个 stage 10 秒；
- 每个负载的五个 optimizer 使用同一 profile；
- 新候选输出 20,000 个 retained samples，cache 关闭；
- 每个 optimizer 运行 3 次，round-robin 轮换顺序；
- optimizer 计划生成上限 300 秒；
- 单次正式执行上限 3,600 秒；
- 正式执行过程中若少于 20,000 个输出即源数据耗尽，则该负载失败；
- DP 任一次正式重复失败、超时或耗尽源数据后，该候选已经不可能满足“三次
  成功重复”的门槛，允许 fail-fast，但仍作为失败项保留在分母中；
- 只有三个重复均有效时才比较均值；三个竞争者重复均超时时，可按 3,600 秒
  给出保守的 censored speedup 下界；部分超时或缺失结果不能建立胜出结论。

完整冻结协议见 `DATA_JUICER_CANDIDATE_PROTOCOL.md`。

## 3. 已完成结果

### 3.1 原有 10 个负载

目前只有两个负载达到门槛：

| 负载 | DP 相对最佳对手 speedup | 结论 |
|---|---:|---|
| `commonvoice` | 1.989x | PASS |
| `simclrv2_cache` | 1.462x | PASS |

其余 8 个均低于 1.20x。原有集合为 2/10。

### 3.2 `redpajama_code`

selection-aware profile 与五 optimizer、三次 round-robin 正式矩阵均已完成：

| optimizer | 执行时间 mean±sd (s) | 优化时间 (s) |
|---|---:|---:|
| 原 Cedar | 551.985±2.758 | 19.426 |
| Data-Juicer | 459.747±1.036 | 3.140 |
| DP-Cedar | 553.566±2.095 | 3.896 |
| DP | 507.168±0.742 | 20.915 |
| Pecan | 466.046±1.537 | 3.126 |

最佳对手为 Data-Juicer，DP speedup 为 `459.747 / 507.168 = 0.906x`，
因此 FAIL，并永久保留在分母中。

该结果还暴露了 bottleneck objective 对 Cedar stage 理想重叠的错误假设。
在任何四个 held-out 扩展负载生成 profile 或 optimizer 结果之前，DP 的最终
held-out 配置已冻结回 selection-aware exact additive objective。Code 结果没有
被删除或重新解释。

### 3.3 `pile_europarl`

第一轮 Data-Juicer 正式执行达到 3,600 秒上限。随后 DP-Cedar 的部分运行与
辅助验证有资源重叠，已明确标为 `invalid_interference`，其部分吞吐和推算时间
均不作为证据。其余缺失 cell 不做插值，EuroParl 作为 FAIL 保留在分母中。

### 3.4 四个 held-out 扩展负载的准备状态

| 负载 | 下载 | 串行源可行性检查 | 正式 optimizer 矩阵 |
|---|---|---|---|
| `pile_hackernews` | 已完成并有 SHA-256 | 20,000/20,000，已完成 | 未开始 |
| `pile_pubmed_abstracts` | 已完成并有 SHA-256 | 20,000/20,000，已完成 | 未开始 |
| `pile_freelaw` | 已完成并有 SHA-256 | 45,706 秒后停止；记录为 `benchmarkable_timeout` | 未开始 |
| `pile_uspto_backgrounds` | 未完成 | 未开始 | 未开始 |

FreeLaw 的串行检查读取了 715,300,864 字节。该检查按官方顺序在单进程中执行
12 个长文本算子（包括语言识别和 perplexity），不是正式系统基准。它远超
1 小时并不能证明 W=8、CPU=64 的并行正式计划不可行。因此规则已修正为：
串行准备检查最多 3,600 秒；超时记录为 `benchmarkable_timeout` 并进入正式
矩阵，由正式执行的相同 3,600 秒上限和 20,000 输出校验裁决。该修正不改变
任何正式数据、资源或 optimizer 参数，也不能把超时自动计为 DP 胜出。

## 4. DP 与实验基础设施已完成的改进

1. Profile 现在记录 filter 的 `input_counts`、`output_counts`、selectivity 和
   observation source，并使用已有 profiling passes 中覆盖率最高的观测，
   不额外增加计时 pass。
2. DP cost 同时传播 size、cardinality 与 volume，能够评估过滤顺序对后续
   昂贵算子的影响。
3. exact DP 的 Pareto dominance 纳入 stage CPU 使用量；外层扩展只枚举合法
   next mask 的真子集，把弱依赖枚举从接近 4^n 降至 3^n transitions。
4. 实现并用独立 exhaustive oracle 验证过 additive 与 bottleneck objective；
   bottleneck 模式保留用于研究，但四个 held-out 正式负载冻结使用 additive。
5. 已完成的验证包括：17 个 DP/cache/fusion 聚焦测试、13 个 analyzer 测试、
   5 个 exhaustive oracle case（共 2,045,952 个合法计划精确匹配），以及
   12-op 弱依赖压力测试（83.77 秒，低于 300 秒优化上限）。
6. 候选 runner 固定五 optimizer、共享 profile、三次 round-robin、计划/执行
   超时和 fail-fast 规则；analyzer 显式忽略 two-stage，并要求全部六个注册
   候选存在才能生成最终报告。
7. 准备脚本会对已下载数据按 metadata、大小和 SHA-256 重新校验后复用，避免
   恢复任务时重复下载；也会复用已经完成的可行性证据。

主要实现/说明文件：

- `cedar/compose/dp_optimizer.py`
- `cedar/compose/my_optimizer.py`
- `cedar/client/dataset.py`
- `SELECTIVITY_AWARE_DP_DESIGN.md`
- `DATA_JUICER_CANDIDATE_PROTOCOL.md`
- `DATA_JUICER_RECIPE_AUDIT.md`
- `run_datajuicer_candidate_matrix.sh`
- `analyze_dp_20pct_goal.py`

## 5. 当前门槛状态

已知 PASS 为 2 个。已知新增失败为 EuroParl 和 Code，因此在最终 16 个负载中，
仍必须让四个 held-out 负载中的至少 3 个达到 1.20x，才能得到至少 5/16
（31.25%）并完成目标。

当前不能宣称目标完成。HackerNews 和 PubMed 的源过滤率较弱，存在不能产生
20% 差距的风险；必须接受正式结果，不能事后删除或只报告有利候选。

## 6. Current runtime status and logs

The original profile chain was stopped after FreeLaw had spent 24,932 seconds
waiting for the first ten-sample Ray batch at terminal feature 0. Its profile
failure is recorded with a 352,096,256-byte source offset in:

evaluation/chapter6_experiments/formal_results/datajuicer_candidate_runs/20260725T072333Z_selectivity/pile_freelaw/status/profile.json

The fixed resume chain started at 2026-07-26 22:48 Asia/Shanghai with container
PID 502183. It reused the validated HackerNews and PubMed profiles from profile
run 20260726T041133Z, reused the recorded FreeLaw failure, and is currently
generating an isolated USPTO profile under a real 3,600-second wall-clock bound.
Its log is:

evaluation/chapter6_experiments/formal_results/datajuicer_extension_profile_fix_20260726T1448Z.nohup

The completed Code report remains at:

evaluation/chapter6_experiments/formal_results/datajuicer_candidate_runs/20260725T072333Z_code_selectivity/dp_20pct_report.md

The resume path requires both a successful completion record in the original
profile-run log and validation of the resource signature and selectivity schema;
it does not rerun Code or silently reuse an unverified profile.

## 7. 后续工作

1. 后台启动恢复链，复用 HackerNews、PubMed、FreeLaw 已验证下载和已有检查；
2. 下载并冻结 USPTO 100,000-row 数据，记录 revision、大小和 SHA-256；
3. 对 USPTO 做最多 3,600 秒的串行准备检查；
4. 为全部四个 held-out 负载重新生成 selection-aware 共享 formal profile；
5. 在隔离资源下依次执行五 optimizer、三次 round-robin 正式矩阵；
6. 对超时、源耗尽、计划失败和不完整重复按冻结规则标注，不做性能插值；
7. 合并现有 10 个负载、EuroParl、Code 和四个 held-out 负载，生成最终 16
   负载报告：
   `formal_results/dp_20pct_goal_latest.md` 和
   `formal_results/dp_20pct_goal_latest.json`；
8. 审计是否至少 5/16 PASS，并检查每个 PASS 都有三次有效 DP 重复和有效最佳
   对手证据；
9. 若不足 5/16，不对 held-out 结果继续调参。任何下一批候选必须先完整注册
   再测量，并把既有失败继续留在累计分母；若修改算法，必须用新的未见验证
   batch 检验，不能在同一结果上反复选择配置；
10. 达标后整理论文表格、图、优化时间与纯吞吐分离结果，并完成最终复现审计。

## 8. 最终完成所需证据

只有同时满足以下条件，任务才可标为完成：

- 固定分母中的每个负载均有成功、失败、超时、不可行或无效原因之一；
- 全部有效 cell 符合同一 W=8/CPU=64/profile/样本数/重复次数协议；
- DP 在至少 5 个负载上相对有效最佳对手达到至少 1.20x；
- 每个胜出结论有三次正式重复或符合冻结规则的严格 censored lower bound；
- 优化时间与纯执行时间分开报告；
- two-stage 未进入任何最佳对手或计数；
- 最终 JSON、Markdown、原始日志、计划、profile、metadata 和 status 文件均可
  追溯。
