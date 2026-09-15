# What a stage width actually buys (2026-09-15)

Follow-up to `PREDICTION_AUDIT_20260914.md`, which listed "interference per
pool per workload" as one of the missing profile axes.  This note measures it
for the text pipeline that the DP was getting wrong (`alpaca_cot`) and turns
it into three model changes.

## Measurement protocol

Every number below comes from `tmp_analysis/run_plan_busy.sh`, which
materialises one plan file, runs it with `--num_total_samples 0` (a drained
full pass: the source is read to EOF and the pipeline is allowed to finish)
and reads the per-record counter logs written by the injected
`patchsite/sitecustomize.py`:

* `records` = calls of the pipeline's terminal operator (and, for pipelines
  whose operators are not Data-Juicer operators, records leaving the pipe, via
  the `PipeVariant.__iter__` counter),
* `busy` = span between the first and the last counted call, which excludes
  actor start-up and is the only part of the run the model is asked to predict,
* `rate = records / busy`, `cores = Σ(per-call time) / busy`.

The harness' own `perf_time_sec` is *not* used: with a deep in-flight window it
measures the time to fill the window rather than to drain it (`issued=5945,
completed=1500` in the counter logs while `perf=0.04 s`).

## Width is not throughput (single fused Ray stage, alpaca_cot)

All eight operators are fused into one Ray stage; the actors per worker `a`,
the worker count `W` and the submit batch (the batch the harness derives for
the workload, `ceil(N/(W·a·3))`) are the only differences.

2 000 records:

| W | a | actors | records/s | ms/record | cores busy |
|---|---|---|---|---|---|
| 8 | 1 | 8 | 395 | 2.53 | 7.9 |
| 8 | 2 | 16 | 470 | 2.13 | 9.5 |
| 8 | 4 | 32 | 762 | 1.31 | 16.8 |
| 8 | 7 | 55 | 1012 | 0.99 | 29.5 |
| 16 | 2 | 31 | 804 | 1.25 | 17.4 |
| 32 | 1 | 32 | 1139 | 0.88 | 25.5 |
| 32 | 2 | 64 | 1215 | 0.82 | 44.7 |

20 000 records:

| W | a | actors | records/s | ms/record |
|---|---|---|---|---|
| 8 | 1 | 8 | 601 | 1.67 |
| 8 | 4 | 32 | 1131 | 0.88 |
| 16 | 2 | 32 | 1418 | 0.71 |
| 8 | 7 | 55 | 1337 | 0.75 |
| 32 | 1 | 32 | 1995 | 0.50 |

Two conclusions the model was getting wrong:

1. **The marginal actor is worth far less than the first one.**  With `W=8`
   the plan reaches 395/470/762/1012 rec/s at `a=1/2/4/7`, where ideal
   per-actor scaling predicts 404/797/1462/1903.  One worker submits to all its
   actors from a single driver loop, so the realized fan-out is close to
   `sqrt(a)`, not `a`.
2. **The same actor count is not the same plan.**  At 32 actors, `W=32,a=1`
   (1139 rec/s) beats `W=16,a=2` (804) and `W=8,a=4` (762) at 2k, and
   `W=32,a=1` (1995) beats `W=16,a=2` (1418) and `W=8,a=4` (1131) at 20k.  The
   worker count is a first-class decision, so a model that treats
   `actors = W·a` as the only width coordinate cannot rank these plans.

The per-operator interference is visible in the same logs: `FlaggedWordsFilter`
averages 19.8 ms/call with 8 actors, 21.9 ms with 32 and 28.9 ms with 55, i.e.
the host is oversubscribed (64 logical CPUs, 32 physical cores) once the plan
puts ~50 heavy processes on it.

## Model changes

1. `DpOptimizer._dp_service_parallelism` divides a parallel stage's service by
   `a^0.5` (`CEDAR_DP_FANOUT_EXPONENT`, default 0.5) instead of by `a`, after
   the existing measured-width cap.  `a=1` — every baseline plan and every
   plan that maximizes workers — is unchanged, so this only removes the
   model's invented reward for stacking actors behind a worker.
2. `DpOptimizer._dp_worker_contention_factor` now reads the calibrated
   `physical_model.worker_contention` points through a *running maximum*.  The
   published alpaca points (1.202 at W=2, 1.851 at W=4, 1.197 at W=8, 1.814 at
   W=16) are not monotone, and interpolating them literally made the worker
   search prefer W=8 with 7 actors over W=32 with 1 actor, although the latter
   measures 1139 against 1012 rec/s.
3. `DpOptimizer._dp_candidate_parallelisms` enumerates a width ladder
   (`CEDAR_DP_WIDTH_LADDER`, default
   `1,2,3,4,6,8,12,16,24,32,48,64,96,128`, `all` restores the old
   every-integer search).  Service is flat past the widest measured point and
   sub-linear below it, so the integer widths in between only made the search
   spend its budget; alpaca planning fell from >25 min (worker search hitting
   its 900 s budget) to 253 s end-to-end with the DP itself at 130 s.  For
   `SMP` the ladder starts at 2: a one-process stage runs on the same host as
   the workers, adds a queue hand-off per record and no parallelism, so the
   search no longer offers it (pricing such a stage when a *baseline* declares
   one is unaffected).  The DP's clip plan moved from 872 rec/s (one SMP(1)
   operator) to 1203 rec/s (all in-process) with that change, matching
   Plumber's plan for the same workload row for row.
4. `DpOptimizer._dp_select_worker_count` gives each worker count a fair share
   of the search budget instead of "whatever is left".  On dino (18
   operators, >30k structural transitions) the first candidate, W=1, consumed
   the whole 900 s budget, so the worker search reported no plan at all and
   the harness skipped the workload; a fair share makes every worker count
   return at least its greedy incumbent, and the deadline path already
   degrades to that incumbent rather than failing.

## Effect on alpaca_cot (2 000 records, drained protocol)

| plan | W | Ray stage | records/s |
|---|---|---|---|
| PICO before | 8 | `[parse, flag]` with 7 actors + `SMP(1)` of 6 ops | 982 |
| cedar / dj / pecan | 32 | all 8 ops, 1 actor | 1117 / 1070 / 1108 |
| **PICO after** | 32 | `[flag]`, 1 actor, rest in-process | **1143** |

The DP now selects the shape every baseline uses (one actor per worker, one
wide worker count) instead of bundling seven actors behind eight workers, and
the model's own ranking follows execution instead of contradicting it.

The other two media workloads behave the same way: on clip the DP's plan is
identical to Plumber's (all in-process, W=32) and measures 1203 against 1302
rec/s, while the recorded cedar/dj/pecan plans declare `submit_batch_size: 1`
on their Ray stage and reach only 105-109 rec/s; on blip the DP's plan is
again all-in-process with W=32 and measures 2342 against 2410 (Plumber) and
2421 (simple-dp).  Those three "winners" are the same plan, and the 3-8%
spread is run-to-run variance of the 1 000-record pass.

## Still open

* The model is still ~1.3x optimistic on the absolute rate of a 32-actor plan
  (predicts 1018 rec/s, measures 1237 at 2k) and it does not yet know that
  `W=32,a=1` beats `W=16,a=2` by more than the `sqrt(a)` correction implies at
  large data sizes; the fan-out exponent is one number fitted on one workload.
* The profile's width sweep is collected with a single local worker, so it
  measures the "one worker, w actors" line only.  A sweep over
  `(W, a) ∈ {1,4,16,32} x {1,2,4}` on the two heaviest operators of each
  workload would replace the `sqrt(a)` fit with a measured surface.
