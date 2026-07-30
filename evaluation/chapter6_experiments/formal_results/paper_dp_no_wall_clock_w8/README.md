# DP optimizer without wall-clock latency correction

This directory is the formal W=8 ablation after removing the DP optimizer's
wall-clock latency correction and direct worker-timing substitution. The cost
model now uses Cedar's Amdahl-inferred per-variant operator cost unchanged.

Only COCO, SimCLR-v2, and SimCLR-v2-cache are rerun because these are the paper
workloads whose prior DP plans used the removed correction. Every workload uses
`W=8`, a 64-CPU budget, one epoch, and three measured executions. Only
`dp_optimizer` is rerun; all unchanged optimizer measurements are retained.

- COCO: 50,000 samples from `train2017`, matching the enlarged original chart.
- SimCLR-v2: all 9,469 files, cache disabled.
- SimCLR-v2-cache: all 9,469 files, with a separate cache warmup when selected.

The copied profiles under `profiles/` are identical to the profiles used by the
original chart. They are deliberately not regenerated, so the comparison
isolates the cost-model change. Plans, metadata, warmup records, and three JSON
measurements per workload are stored under `workloads/`. Runtime logs and cache
contents are intentionally untracked.

Run inside `optimalcedar-torch201-dev` after activating `env`:

```bash
nohup bash evaluation/chapter6_experiments/run_paper_dp_no_wall_clock_w8.sh \
  > evaluation/chapter6_experiments/formal_results/paper_dp_no_wall_clock_w8/runner.out 2>&1 &
```

After all measurements succeed, the same command automatically redraws the
paper figures in `formal_results/paper_figures_optimizer_w8_no_wall_clock/`.
The original result and figure directories are not modified.
