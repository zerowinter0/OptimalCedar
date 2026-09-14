# Selected DP optimizer comparison

This artifact contains the optimizer execution-time and planning-time figures
used by the paper. It is generated from the single-run remote-resource-pool
experiment at:

`outputs/chapter6_experiments/remote_additive_resource_sum_formal_v1/formal_runs`

A workload is included only when a completed PICO or Simple-DP execution is
faster than both DJ and Pecan executions. A heuristic planning or execution
timeout is treated as a worse terminal outcome than a completed DP execution.
The plotting program validates this rule before writing any output.

Reproduce the artifact inside the project container after activating
`env/bin/activate`:

```bash
python evaluation/chapter6_experiments/plot_selected_dp_optimizer_matrix.py \
  --matrix-root outputs/chapter6_experiments/remote_additive_resource_sum_formal_v1/formal_runs \
  --output-dir evaluation/chapter6_experiments/formal_results/paper_artifacts/optimizer/selected_dp_remote
```

`selected_dp_optimizer_data.json` and `selected_dp_optimizer_data.tsv` record
the execution/setup values, terminal status, sample count, logical operator
count, and the source path of every plotted value.
