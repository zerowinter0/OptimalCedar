# Formal seven-optimizer matrix

This directory is the canonical source for the optimizer execution and
optimization-time figures. It compares Cedar, DJ, Pecan, DJ-TS, Pecan-TS,
Simple-DP, and PICO under one protocol.

- `matrix/` contains the physical plans, terminal status records, three-round
  execution results, plan-only timing records, and per-workload metadata.
- `profiles/` contains the exact shared profiles used by every optimizer.

All workloads use `W=8`, `CPU_BUDGET=64`, cache disabled, and a unified
one-hour limit covering plan generation plus the first execution. A timeout in
the first attempt suppresses the remaining repetitions. Successful cells have
three round-robin executions. GenerateVideo uses 5,000 outputs; the other
sample counts are recorded in each workload's `metadata.txt`.

The plotted table is generated as
`optimizer/figures/formal_seven_optimizer_data.{json,tsv}`. Reproduce it from
the repository root inside the project container with:

```bash
python evaluation/chapter6_experiments/plot_formal_seven_optimizer_matrix.py \
  --matrix-root evaluation/chapter6_experiments/formal_results/paper_artifacts/optimizer/data/formal_seven/matrix \
  --output-dir evaluation/chapter6_experiments/formal_results/paper_artifacts/optimizer/figures
```
