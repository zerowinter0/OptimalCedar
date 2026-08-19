# Formal seven-optimizer figures

The canonical optimizer figures use the completed formal matrix archived in
`../data/formal_seven/`.  The matrix compares Cedar, DJ, Pecan, DJ-TS,
Pecan-TS, Simple-DP, and PICO with `W=8`, `CPU_BUDGET=64`, three round-robin
executions, and one unified one-hour limit covering optimization plus
execution.

`formal_seven_optimizer_execution.*` reports absolute execution time. Each
workload has an independent linear y-axis so that differences remain visible
without implying a shared scale. Bars and error bars are the mean and sample
standard deviation of the three successful executions. A red hatched `TO`
bar denotes a task that did not finish within one hour; its plotted height is
only a visual sentinel and is not a measured runtime.

`formal_seven_optimizer_overhead.*` reports optimizer time on a log2 scale.
The accompanying JSON and TSV files contain the exact plotted values and
status for every cell.

## Reproduction

Run inside the project container after `source env/bin/activate`:

```bash
python evaluation/chapter6_experiments/plot_formal_seven_optimizer_matrix.py \
  --matrix-root evaluation/chapter6_experiments/formal_results/paper_artifacts/optimizer/data/formal_seven/matrix \
  --output-dir evaluation/chapter6_experiments/formal_results/paper_artifacts/optimizer/figures
```

Files whose names begin with `latest_optimizer_` are retained only as
historical artifacts and are not the current paper figures.
