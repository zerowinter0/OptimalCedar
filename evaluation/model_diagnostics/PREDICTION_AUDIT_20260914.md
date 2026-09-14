# Does the cost model predict a plan's throughput?

58 measured `(workload, plan)` pairs across nine workloads, all at the same
profile per workload, one run per cell
(`tmp_analysis/prediction_error_report.py`,
`outputs/prediction_errors_20260914.json`).

Predicted throughput is derived from the score the model gives the *same*
materialized plan: `predicted = 1000 * W / lane_ms`, where `lane_ms` is the
per-record per-worker score (`pico_plan_costs_by_feature`) and `W` is the
worker count of that plan.  Measured throughput is `num_samples / perf_time`.

## Overall

| statistic | value |
|---|---|
| plans | 58 |
| median \|log2(predicted/measured)\| | 1.19 (≈2.3×) |
| p90 | 3.57 (≈12×) |
| max | 5.27 (≈38×) |
| within ±25% | 8 / 58 |

## Per workload

| workload | n | median \|log2\| | within 1.25× | worst plan |
|---|---|---|---|---|
| simclr | 8 | 0.72 | 2/8 | dj_optimizer ×2.51 |
| blip | 9 | 0.44 | 3/9 | raydata_optimizer ×0.17 |
| clip | 9 | 1.38 | 0/9 | raydata_optimizer ×0.20 |
| dino | 6 | 0.39 | 2/6 | raydata_optimizer ×0.43 |
| alpaca_cot | 9 | 4.35 | 1/9 | pecan_optimizer ×0.03 |
| pile_hackernews | 6 | 1.19 | 0/6 | plumber_optimizer ×0.17 |
| pile_pubmed_abstracts | 5 | 1.50 | 0/5 | dj_optimizer ×0.09 |
| pile_uspto_backgrounds | 4 | 3.36 | 0/4 | pecan_optimizer ×0.08 |
| bloom_oscar | 2 | 2.15 | 0/2 | pecan_optimizer ×0.23 |

Two error families, opposite in sign:

* **too pessimistic** on plans whose work sits on a wide fused parallel stage
  (all text workloads: predictions 3–38× below measurement);
* **too optimistic** on plain local chains at large `W` (clip ×2.4–3.3,
  simclr `dj_optimizer` ×2.5, pile_hackernews `plumber` ×5.7 in the other
  direction).

That both signs occur means the missing pieces are structural, not a single
scale factor: no coefficient makes the text plans 20× faster and the clip
plans 2.5× slower at the same time.

## Deep dive 1 — alpaca_cot, `pecan_optimizer` plan (prediction 0.03×)

Chain of measurements for the same op (`FilterPipe_FlaggedWordsFilter`):

| measurement | value |
|---|---|
| direct micro-benchmark, `op(sample)` per record, 2000 real records | 17.9 ms/record |
| profile `cm_model` (byte-linear fit, r²=0.97) | 18.1 ms/record |
| profile isolated RAY stage, width 1 | 18.1 ms/record |
| profile isolated RAY stage, width 8 | 3.36 ms/record |
| profile isolated SMP stage, width 8 | 2.49 ms/record |
| **winning plan, measured end to end** | **0.015 ms/record** (65546 rec/s) |

At 32 actors the plan must spend at least `2000 × 18 ms / 32 ≈ 1.1 s` on that
one op, yet the whole 2000-record epoch takes 0.03 s; the process wall clock is
constant (≈29 s) for 8, 2000 and 20000 records, which rules out "the work
happens outside the timed region".

Checks performed: filters do run in these plans (a probe built so that only
`flagged_words` drops records is filtered correctly by both the local and the
fused-Ray plan); threading does not explain it (the op is GIL-bound: 8 threads
give 57 ms/record, worse than 1 thread); the in-flight window does not explain
it (`use_threads=False` → 0.028 s, `max_inflight=1` → 0.088 s).

Conclusion: for these plans the harness's throughput accounting cannot be
reconciled with any faithful operator cost.  Until we count the records and
operator invocations the plan actually performs, the text-workload comparison
cannot be trusted, and fitting the model to it would bake in a measurement
artefact.

## Deep dive 2 — clip, local plan at W=32 (prediction ×2.4–3.3)

PICO, Plumber and Simple-DP all materialize the same plan here: the declared
operator order with every stage INPROCESS at `W=32`; the model scores it at
2017 rec/s while the three runs measure 711 / 828 / 776 rec/s.  The profile
measured each operator with one worker and one process, so the per-worker cost
carries no information about the contention 32 co-located workers create
(memory bandwidth, tokenizer state); the worker-contention points we ship were
measured on SimCLRv2 (1.20 at W=8, 1.81 at W=16) and are applied to every
workload, which is why the error direction differs per workload.

## Deep dive 3 — simclr, where the model does work

PICO 1.26×, Cedar 0.98×, Plumber 0.84×.  These plans place their work on the
pool whose width the profile measured (fused stage at width 6–8), so summing
isolated per-operator costs and dividing by the stage width is the right
composition.  This is the regime the current model was built for.

## Can we predict any plan on any workload?

## Update — the text-workload measurements were a harness artefact

An operator-call counter (`CEDAR_RAY_ACTOR_PYTHONPATH` + a `sitecustomize`
that logs every call) settles the alpaca case.  For a 200-record request:

| plan | `FlaggedWordsFilter` calls | its CPU time | harness epoch |
|---|---|---|---|
| unoptimized local (W=4) | 268 | 5.10 s | 0.23 s |
| Cedar fused-Ray (W=1, 1 actor) | 2245 | 50.94 s | **0.042 s** |

The fused plan does **11× more work than the request** and burns 51 s of CPU
while the harness reports 0.042 s: with a deep in-flight window the sink
receives records from the buffer long before the stage has processed the
queue, and the run is torn down with thousands of records still in flight
(`issued=5945, completed=1500` on one worker).  The harness's "total time"
therefore measures *buffer fill*, not throughput, whenever the source
outruns the stage.

Fix: measure a **drained full pass** (`--num_total_samples 0`, iterate to
exhaustion so the sink blocks until the pipeline is empty).  The same three
alpaca plans under that protocol:

| plan | W | wall (74.7k records) | per-record |
|---|---|---|---|
| Cedar fused-Ray | 32 | 34.4 s | 0.46 ms |
| dj-cedar | 32 | 35.4 s | 0.47 ms |
| PICO (Ray[4,7] + SMP) | 8 | 65.8 s | 0.88 ms |

The 31× gap collapses to 1.9×, and the model's per-worker score for Cedar's
plan (16.9 ms) now matches the measurement (0.46 ms × 32 = 14.7 ms) within
15%.  All headline comparisons must use this protocol.

Yes, under three conditions.

1. **Measure every cost in the form the plan executes it.**  A stage cost must
   be a function of `(operator or operator chain, backend, width, submit
   batch)`, measured through the runtime's own submission path, not the sum of
   single-record, single-actor measurements.  The alpaca and clip cases are the
   two extremes of this failure.
2. **Propagate cardinality on real data.**  All our profiles report
   selectivity 1.0 even for filters that drop records, so the model cannot
   price the reordering/fusion gains on which every text-workload winner
   depends; a plan's per-operator input cardinality must be predicted, not
   assumed constant.
3. **Measure interference per pool per workload.**  Co-locating `W` workers or
   `a` actors changes per-record cost (clip ×2.5, dino ×1.3, simclr ×1.2);
   this factor is a property of the workload's operators, so it must be
   measured inside each profile rather than inherited from another workload.

## What the profile has to measure

| axis | what to measure | why (observed failure) |
|---|---|---|
| operator/stage cost | per-record stage cost for each operator and each *fused chain* the DP may build, at widths {1,2,4,8,32,56} × submit batches {1,4,30,500} | alpaca 0.015 vs 18 ms/record; dino/blip fusion gains |
| cardinality | records in/out per operator on real data (selectivity curve per input size) | filter reordering is the entire gain on alpaca/pubmed/uspto |
| boundary | submit + return cost per `(payload, batch)` for each backend | batching amortizes the per-call part the model currently charges per record |
| transport | per-record bytes ÷ achieved rate per worker, shared link rate, in-flight window budget and its penalty | already calibrated; Ray Data plan now within 1.3× |
| interference | per-workload factor for `W` workers and `a` actors sharing a pool | clip ×2.5 optimistic, simclr ×1.2 |
| source | per-record reader cost and prefetch depth effect | unoptimized plans |
| validation | run 3–5 canonical plans per workload (declared-order local, all-fused Ray, fused SMP, the planner's own plan) and store measured vs predicted | would have caught the alpaca inconsistency before it entered the comparison |

Complexity: all of these are lookups keyed by properties the DP already
carries (operator set, backend, width) plus one cardinality multiplier that the
search already propagates (`_dp_cardinality_prod`), so the transition cost
stays O(1) and the DP state does not grow.
