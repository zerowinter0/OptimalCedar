# Cedar vs Plumber-style vs our model (five target_pipeline workloads)

## Protocol

- Workloads: blip, clip, dino, simclr, swav (evaluation/pipelines/target_pipeline).
- 1000 samples, batch 4, W=8, CPU budget 64 (local) and 64 (Ray); one profile per
  workload shared by all three optimizers; 3 repeats with round-robin order.
- Optimizers:
  - `optimizer`: original Cedar staged optimizer (selector 0).
  - `plumber_optimizer`: Plumber-style per-stage width allocation (selector 18),
    implemented in cedar/compose/plumber_optimizer.py. It keeps the declared
    order, does not reorder/fuse/cache/cross-backend, and maximizes the minimum
    stage rate over the shared per-worker CPU budget.
  - `dp_optimizer`: our model (joint DP + calibrated stage service).
- Cedar's reorder enumeration is bounded at 600 s for the swav/dino/clip re-runs
  (it cannot finish on 60+ operator graphs); timeouts are recorded, not imputed.
- The manifest read mapper is pinned to INPROCESS for the four manifest
  workloads: it opens local file paths that do not exist on the remote Ray node.

## Files

- status.json, metadata.json, summary.json: driver state and protocol.
- <workload>_profile.yaml: the shared profile used by all three optimizers.
- results/<workload>.json: full comparison output including physical plans.
- logs/profile_<workload>.log, logs/compare_<workload>.log: raw logs.
- comparison_table.csv, SUMMARY.txt: consolidated table (written when finished).
