# 第六章实验脚本说明

本目录用于组织 PICO/OptimalReorder 论文第六章实验。脚本只新增实验编排层，不修改 `cedar/` 和已有 `evaluation/` 主入口；实际运行仍复用当前仓库的 `eval_cedar.py`、`compare_optimizer_perf.py`、`benchmark_cedar_reorder_time.py` 以及各 workload 的 `cedar_dataset.py`。

## 实验矩阵

建议第六章至少完成以下实验：

1. **优化开销与规模扩展**：用独立算子合成流水线比较 Cedar 原始枚举重排与 DP 重排/联合 DP 的优化时间，展示阶乘候选空间与 DP 状态空间的差异。
2. **真实 workload 的计划质量**：在 BLOOM/OSCAR、LLaVA-pretrain、Wikitext103、SimCLRv2 等 workload 上比较 modeled cost：原 Cedar staged、DP reorder-only、two-stage DP、PICO unified DP。
3. **端到端运行性能**：用相同 profile 生成物理计划后真实执行，报告 throughput、time per sample、优化时间是否排除、cache warmup 是否排除。
4. **联合优化消融**：固定 unified DP，分别关闭 reorder、fusion、offload、cache、local parallelism，证明收益来自联合搜索而非单一局部技巧。
5. **缓存/多 epoch 收益**：比较 full DP 与 no-cache，在多 epoch 设置下单独报告 warmup 与 steady-state epoch 性能，支撑 cache-after-fusion 的实现语义。
6. **profile 样本量敏感性**：用不同 profile sample count 重新 profile 并比较计划 cost，检查优化结果是否对少量 profiling 噪声过敏。
7. **可选兼容性基线**：如需要和 Cedar VLDB 原系统整体对比，可继续使用 `evaluation/run_*.sh` 跑 PyTorch/tf.data/Ray/FastFlow/Plumber；这部分不属于 PICO 优化器核心贡献，建议作为补充或附录。

## 目录内容

- `experiments.env.example`：服务器环境变量模板。
- `run_all.sh`：主实验一键入口。
- `run_profile_workloads.sh`：为 selected workloads 生成 profile。
- `run_optimizer_overhead.sh`：E1，合成重排开销 + 真实 workload plan-only 开销。
- `run_plan_quality.sh`：E2，modeled cost 对比。
- `run_runtime.sh`：E3，端到端吞吐对比。
- `run_ablation.sh`：E4，DP 功能消融。
- `run_cache_epoch.sh`：E5，多 epoch cache 对比。
- `run_profile_sensitivity.sh`：E6，profile 样本量敏感性。
- `run_dp_ablation.py`：消融实验执行器。
- `summarize_results.py`：将 JSON/CSV 汇总成论文表格友好的 CSV 和 Markdown。

## 服务器迁移步骤

在服务器上复制整个仓库后：

```bash
cd /OptimalCedar
source env/bin/activate
pip install -e .
cp evaluation/chapter6_experiments/experiments.env.example \
  evaluation/chapter6_experiments/experiments.env
```

编辑 `evaluation/chapter6_experiments/experiments.env`，至少确认：

- `CH6_OUT_DIR`：结果输出目录，建议放到大容量磁盘或 `/tmp/pico_chapter6_results`。
- `CH6_WORKLOADS`：先用 `bloom_oscar` smoke，确认后扩展到 `bloom_oscar llava_pretrain wikitext103 simclrv2`。
- `BLOOM_DATASET_PATH`、`LLAVA_DATASET_PATH`、`LLAVA_IMAGE_ROOT`：真实数据路径。
- `CH6_USE_RAY`、`CH6_RAY_IP`、`CH6_DISABLE_OFFLOAD`：Ray/offload 设置。当前主对比脚本复用 `compare_optimizer_perf.py`，更适合在 Ray head/本机 Ray 环境上运行；如果从登录节点连接远端 Ray，需要先用小样本确认该脚本的 Ray 行为。
- `CH6_FULL_DATA_RUN`：正式实验设为 `1`；调试时保持 `0` 并用 `CH6_DATA_NUM_TOTAL_SAMPLES` 控制小样本运行。

运行主实验：

```bash
cd /OptimalCedar
source env/bin/activate
bash evaluation/chapter6_experiments/run_all.sh
```

只跑某一类实验：

```bash
bash evaluation/chapter6_experiments/run_optimizer_overhead.sh
bash evaluation/chapter6_experiments/run_plan_quality.sh
bash evaluation/chapter6_experiments/run_runtime.sh
bash evaluation/chapter6_experiments/run_ablation.sh
```

重新汇总已有结果：

```bash
python evaluation/chapter6_experiments/summarize_results.py \
  --results_dir /tmp/pico_chapter6_results \
  --output_dir /tmp/pico_chapter6_results/summary
```

## 输出结构

默认输出到 `CH6_OUT_DIR`：

- `logs/`：每个实验的完整命令与日志。
- `profiles/`：脚本生成的 workload profile。
- `plans/`：消融和 cache 实验保存的物理计划。
- `raw/`：原始 JSON/CSV 结果。
- `summary/optimizer_comparison_summary.csv`：plan quality/runtime/ablation 汇总。
- `summary/synthetic_overhead_summary.csv`：合成重排开销汇总。
- `summary/chapter6_summary.md`：便于快速检查的 Markdown 表格。

## 推荐执行策略

先做 smoke：

```bash
CH6_WORKLOADS=bloom_oscar \
CH6_REPEATS=1 \
CH6_PROFILE_SAMPLES=20 \
CH6_DATA_NUM_TOTAL_SAMPLES=20 \
CH6_FULL_DATA_RUN=0 \
bash evaluation/chapter6_experiments/run_all.sh
```

确认 profile、Ray 和数据路径无误后，再做正式实验：

```bash
CH6_WORKLOADS="bloom_oscar llava_pretrain wikitext103 simclrv2" \
CH6_REPEATS=3 \
CH6_PROFILE_SAMPLES=200 \
CH6_FULL_DATA_RUN=1 \
bash evaluation/chapter6_experiments/run_all.sh
```

如果 `CH6_USE_RAY=1` 且启用 offload，profile 必须包含对应 Ray/offload 统计；否则 `compare_optimizer_perf.py` 会拒绝运行，这是预期的保护机制。
