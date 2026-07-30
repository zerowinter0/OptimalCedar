# Latest optimizer figures with DP-Cedar baseline

This is a source-tracked synthesis of the newest valid W=8 artifacts. Every
successful plotted cell contains three round-robin measured executions:

- COCO, CommonVoice, and CommonVoice-cache: `evaluation/chapter6_experiments/formal_results/scaled_reuse_plan_runs/coco_cv_enlarged_w8_formal_20260727`
  (enlarged inputs, three round-robin repetitions).
- LLaVA, RP-C4, StackExchange, SimCLR(v2), and WikiText-103 variants:
  `evaluation/chapter6_experiments/formal_results/paper_optimizer_w8` (fixed W=8, three round-robin repetitions).
- RP-Code and the Pile pipelines: `evaluation/chapter6_experiments/formal_results/dp_20pct_goal_latest.json`
  (20,000 outputs, three round-robin repetitions).

14 workloads have a valid DP-Cedar execution baseline. Pile EuroParl is
excluded because its DP-Cedar run was invalidated by interference; Pile FreeLaw
is excluded because it has no valid plans. Invalid workloads are not plotted.
The per-workload x-axis is ordered by increasing logical Cedar operator count
(excluding the source); ties retain the suite order.
Cedar has valid execution plans on 9/14
workloads and optimizer-timeout outcomes on the other valid-baseline pipelines.

Headline values:

- DP geomean speedup over DP-Cedar across all 14 valid-baseline workloads:
  **1.329x**.
- On the common 9 workloads where Cedar also completes,
  DP achieves **1.048x** over DP-Cedar.

Use the per-workload figures as the primary paper evidence. Error bars are
propagated sample standard deviations of the normalized ratio.

## Reproduction

Run inside the project container after `source env/bin/activate`:

```bash
python evaluation/chapter6_experiments/plot_latest_optimizer_dp_cedar_baseline.py \
  --candidate-report evaluation/chapter6_experiments/formal_results/dp_20pct_goal_latest.json \
  --scaled-run evaluation/chapter6_experiments/formal_results/scaled_reuse_plan_runs/coco_cv_enlarged_w8_formal_20260727 \
  --paper-matrix evaluation/chapter6_experiments/formal_results/paper_optimizer_w8 \
  --output-dir evaluation/chapter6_experiments/formal_results/paper_figures_optimizer_w8_three_repeat
```
