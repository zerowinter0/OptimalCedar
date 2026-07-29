# Latest optimizer figures with DP-Cedar baseline

This is a source-tracked synthesis of the newest valid artifacts, not a claim
that every workload used one identical repetition protocol:

- COCO, CommonVoice, and CommonVoice-cache: `/workspace/OptimalCedar/evaluation/chapter6_experiments/formal_results/scaled_reuse_plan_runs/coco_cv_enlarged_w8_formal_20260727`
  (enlarged inputs, three round-robin repetitions).
- LLaVA, RP-C4, StackExchange, SimCLR(v2), and WikiText-103 variants:
  `/workspace/OptimalCedar/evaluation/chapter6_experiments/formal_results/cross_system_w8_latest.json` (latest completed formal cell; one run).
- RP-Code and the Pile pipelines: `/workspace/OptimalCedar/evaluation/chapter6_experiments/formal_results/dp_20pct_goal_latest.json`
  (20,000 outputs, three round-robin repetitions).

Fourteen workloads have a valid DP-Cedar execution baseline. Pile EuroParl is
excluded because its DP-Cedar run was invalidated by interference; Pile FreeLaw
is excluded because it has no valid plans. Invalid workloads are not plotted.
The per-workload x-axis is ordered by increasing logical Cedar operator count
(excluding the source); ties retain the suite order.
Cedar has valid execution plans on 9/14
workloads and optimizer-timeout outcomes on the other valid-baseline pipelines.

Headline values:

- DP geomean speedup over DP-Cedar across all 14 valid-baseline workloads:
  **1.354x**.
- On the common 9 workloads where Cedar also completes,
  DP achieves **1.139x** over DP-Cedar.

Use the per-workload figures as the primary paper evidence. The aggregate is
preliminary because the source suite mixes one-run legacy cells with newer
three-repeat cells. Error bars are shown only where three repeated executions
exist; they are propagated sample standard deviations of the normalized ratio.
