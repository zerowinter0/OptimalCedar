# Formal optimizer results (W=8)

This is the single canonical archive behind the current optimizer figures.
It contains only the five plotted optimizers: DP-Cedar (baseline), Data-Juicer,
Pecan, DP (ours), and Cedar. The obsolete two-stage optimizer is excluded.

## Layout

- `data/enlarged_core/`: COCO, CommonVoice, and CommonVoice-cache measurements.
- `data/standard_core/`: the other seven Cedar workload measurements.
- `data/dp_no_wall_clock/`: replacement DP measurements for COCO and the two
  SimCLR-v2 workloads after removal of the wall-clock cost correction.
- `data/data_pipeline_matrix.json`: RP-Code and Pile aggregate measurements.
- `profiles/`: the exact profiles used to create the plans.
- `figures/`: current PDF, SVG, PNG, and source TSV figure artifacts.
- `MANIFEST.tsv`: size and SHA-256 checksum of every archived file.

All plotted successful executions contain three measurements. Experiments use
W=8 and a 64-CPU budget. COCO uses 50,000 `train2017` samples; SimCLR-v2 uses
all 9,469 files. Cache workloads are measured only after a separate warmup.

## Reproduce the figures

From the repository root inside `optimalcedar-torch201-dev`:

```bash
source env/bin/activate
python evaluation/chapter6_experiments/plot_latest_optimizer_dp_cedar_baseline.py \
  --candidate-report evaluation/chapter6_experiments/formal_results/paper_optimizer_w8/data/data_pipeline_matrix.json \
  --scaled-run evaluation/chapter6_experiments/formal_results/paper_optimizer_w8/data/enlarged_core \
  --paper-matrix evaluation/chapter6_experiments/formal_results/paper_optimizer_w8/data/standard_core \
  --dp-replacement-matrix evaluation/chapter6_experiments/formal_results/paper_optimizer_w8/data/dp_no_wall_clock \
  --output-dir evaluation/chapter6_experiments/formal_results/paper_optimizer_w8/figures
```

Runtime logs and materialized caches are deliberately omitted: neither is an
input to the figures, and cache contents account for nearly all of the former
`formal_results` disk usage.

Some immutable raw JSON/YAML records contain their original absolute
`source` or `profiled_stats` path. Those strings are provenance captured at
measurement time, not live dependencies; the referenced profiles and all
values consumed by the plotting script are present in this archive.

The measurement entry point is
`evaluation/chapter6_experiments/run_paper_dp_no_wall_clock_w8.sh`. By default
it writes a new scratch run outside `formal_results`, so rerunning experiments
does not modify this canonical archive.
