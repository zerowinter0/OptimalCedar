# Chapter 6 experiment suite

This directory contains only the scripts and data used by the current paper.
Run every Python or shell entry point inside `optimalcedar-torch201-dev` after
`source env/bin/activate`.

## Canonical results

`formal_results/paper_artifacts/` is the single authoritative archive. It
contains five paper figures and their exact inputs:

1. optimizer execution performance;
2. cross-system absolute execution time;
3. optimizer planning overhead;
4. DP scalability and exhaustive optimality validation;
5. cost-model ranking and plan-selection accuracy.

The archive contains profiles, physical plans, three-round JSON measurements,
status records, source tables, figures, and a SHA-256 manifest. It intentionally
contains no runtime cache, raw log, failed repair, or superseded figure.

See `formal_results/paper_artifacts/README.md` for the figure-to-data map and
exact plotting commands. Verify the archive with:

```bash
python evaluation/chapter6_experiments/verify_paper_artifacts.py
```

## Retained experiment entry points

- `run_formal_profiles.sh`: generate shared ten-second baseline/Ray/SMP
  profiles.
- `run_formal_plan_and_matrix.sh`: materialize W=8 plans and execute three
  round-robin repetitions.
- `run_datajuicer_candidate_matrix.sh`: RP-Code and Pile workload matrix.
- `run_coco_commonvoice_enlarged_w8.sh`: bounded enlarged COCO/CommonVoice
  protocol; delegates to `run_scaled_reuse_plan_matrix.sh`.
- `run_paper_dp_no_wall_clock_w8.sh`: current PICO measurements affected by
  removal of the wall-clock correction.
- `run_paper_cross_system_w8.sh`: native-system experiment and final absolute
  execution-time figure.
- `run_dp_algorithm_validation.sh`: synthetic search scalability and
  exhaustive optimality verification.
- `run_general_video_refine_w8.sh`: complete MSR-VTT general-video-refine
  profile and six-optimizer W=8 matrix. Its pre-measurement protocol is
  `GENERAL_VIDEO_REFINE_PROTOCOL.md`.

Long experiments write to a scratch directory and should be launched with
`nohup`. They do not overwrite `paper_artifacts` unless results are explicitly
reviewed and promoted.

## Figure and analysis entry points

- `plot_latest_optimizer_dp_cedar_baseline.py`
- `plot_paper_cross_system_absolute.py`
- `plot_search_validation.py`
- `analyze_formal_cost_model_accuracy.py`
- `plot_formal_cost_model_accuracy.py`
- `verify_paper_artifacts.py`

`plot_paper_cross_system_speedup.py` remains only as the shared parser used by
the absolute-time plot; no speedup figure is produced for the paper.

## Protocol documents

- `DATA_JUICER_CANDIDATE_PROTOCOL.md`: candidate workload fairness and timeout
  policy.
- `SCALED_REUSE_PLAN_PROTOCOL.md`: enlarged-input plan-reuse protocol.
- `SELECTIVITY_AWARE_DP_DESIGN.md`: selectivity and semantic-dependency design.
- `DATA_JUICER_RECIPE_AUDIT.md`: operator mapping audit.
- `GENERAL_VIDEO_REFINE_PROTOCOL.md`: frozen video-text data, resource,
  correctness, timeout, and measurement protocol.

All formal executions use local `W=8`, a 64-CPU budget, the same frozen profile
for every optimizer on a workload, and three round-robin repetitions. Cache
workloads warm each plan independently and exclude warmup from reported time.
