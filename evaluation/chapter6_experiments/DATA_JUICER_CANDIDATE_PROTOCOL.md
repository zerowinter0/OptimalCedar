# Data-Juicer workload extension protocol

This file freezes the workload-selection rule before optimizer measurements are
collected. It is intended to prevent result-driven workload selection.

## Source

- Recipe repository: `datajuicer/data-juicer-hub`
- Recipe revision: `47fc345`
- Data-Juicer implementation revision used for operator semantics: `bb3d88aac1`
- Migration audit: `DATA_JUICER_RECIPE_AUDIT.md`
- Resource protocol: the Chapter 6 W=8 protocol (`CPU_BUDGET=64`, one
  Ray actor or SMP process per profiled stage, ten seconds per profile stage,
  one shared profile for all optimizers, and a 60-minute plan timeout).
- Historical optimizer set for the batches pre-registered in this file:
  original Cedar, Data-Juicer, DP-Cedar, DP, and Pecan. The later diverse
  workload study in `DATA_JUICER_DIVERSE_WORKLOADS.md` supersedes this
  comparison set for current public claims and requires all six optimizers,
  including DP-two-stage.

Global deduplicators are excluded because Cedar's optimizer operates on
per-sample pipelines. This is the same documented boundary used by the
existing C4 and StackExchange migrations. All per-sample mappers and filters
remain in recipe order before optimization.

## Predeclared selection rule

A candidate must:

1. come from an official Data-Juicer Hub recipe;
2. contain at least eight reorderable per-sample filters;
3. expose heterogeneous costs or selectivities that make joint reorder,
   fusion, and backend selection scientifically relevant;
4. have a reproducible raw-data source and at least 20,000 retained samples;
5. fit the existing at-most-3-GiB input-data policy.

Only the following two candidates are admitted for this extension. They were
chosen before profiling or generating any optimizer plan.

| Workload | Official recipe | Raw data | Per-sample recipe |
|---|---|---|---|
| `pile_europarl` | `refined_recipes/pretrain/pile-europarl-refine.yaml` | The Pile EuroParl mirror `timaeus/pile-europarl`, preserving `meta.pile_set_name=EuroParl` | 5 fixed normalization mappers, 11 reorderable filters |
| `redpajama_code` | `refined_recipes/github_code/redpajama-code-refine.yaml` | `togethercomputer/RedPajama-Data-1T`, `github` configuration | 6 fixed cleaning/normalization mappers, 8 reorderable filters |

The EuroParl mirror is used because the original `the-eye.eu` URL embedded in
the historical Pile loader no longer exists. The downloader records the
resolved dataset revision and SHA-256 digest of the local JSONL file.

## Fixed experiment size

- Each downloaded input is hard-capped at 3 GiB.
- `pile_europarl` uses the largest source-order prefix that does not exceed
  2 GiB, with no duplication or resampling.
- `redpajama_code` uses the first 50,000 valid records in streaming order,
  without duplication or resampling.
- Formal execution consumes 20,000 retained samples for both workloads.
- Every optimizer uses the same local JSONL file, shared profile, sample
  count, cache setting (off), and execution repetitions/order.

Candidate failures and sub-20% outcomes remain in the ledger and denominator;
they are not silently removed.

## Registered feasibility-extension batch

The first formal execution of `pile_europarl` exposed a dataset-feasibility
problem rather than an optimizer result: after reading more than half of the
fixed 2-GiB prefix, the pipeline had retained far fewer than the required
20,000 records. `pile_europarl` remains in the result ledger and denominator.
It is not replaced or discarded.

To keep the workload search reproducible, the following extension batch was
registered on 2026-07-25 before downloading, profiling, generating plans, or
executing any of these four workloads. Selection used only the official recipe
structure and public source metadata:

| Workload | Official recipe | Frozen raw source | Public source metadata |
|---|---|---|---|
| `pile_hackernews` | `refined_recipes/pretrain/pile-hackernews-refine.yaml` | `timaeus/pile-hackernews@a5d6a32ee1039015b8037da6aa776af4cfb89df1` | 100,000 records; 500,518,158 logical bytes |
| `pile_pubmed_abstracts` | `refined_recipes/pretrain/pile-pubmed-abstract-refine.yaml` | `timaeus/pile-pubmed_abstracts@2d733e1624c384e4a97acfd4b93c8e739420b32e` | 100,000 records; 135,484,704 logical bytes |
| `pile_freelaw` | `refined_recipes/pretrain/pile-freelaw-refine.yaml` | `timaeus/pile-freelaw@e5cf633ac70c4659cb4761718bdd93d029df5150` | 100,000 records; 1,588,135,150 logical bytes |
| `pile_uspto_backgrounds` | `refined_recipes/pretrain/pile-uspto-refine.yaml` | `timaeus/pile-uspto_backgrounds@cb4f574c22debca066312ddccd0d048cbd7e148b` | 100,000 records; 428,010,825 logical bytes |

Each recipe has at least nine reorderable per-sample filters, including
heterogeneous tokenization, language-identification, and perplexity costs.
Global SimHash remains outside the Cedar per-sample optimizer boundary. The
complete frozen 100,000-row split fits below 3 GiB in every case, so no
source-order subsampling is needed. A source-only feasibility pass normally
confirms at least 20,000 retained records before optimizer profiling. A
feasibility failure is still reported and retained in the denominator.

The source-only pass is a serial official-order check, not an optimizer
benchmark. While no extension profile or optimizer result existed, FreeLaw
showed that this check itself can run for more than twelve hours on long legal
documents. The preparation rule was therefore bounded at 3,600 seconds. A
completed source-infeasible check remains a failure, but a serial-check timeout
is recorded as `benchmarkable_timeout` and proceeds to the formal W=8,
CPU-budget-64 benchmark. It cannot count as a win: formal execution still has
the unchanged 3,600-second limit and must return 20,000 samples in all three DP
repetitions; otherwise the workload fails. This operational amendment changes
neither input data nor any measured optimizer setting and avoids using an
unmeasured single-process utility as a proxy for the parallel system result.

The extension batch uses exactly the same W=8, CPU-budget-64, shared-profile,
three-repeat round-robin, 3,600-second plan timeout, and 3,600-second execution
timeout protocol as the first batch. Registering this complete batch before
measurement prevents selecting only workloads whose observed optimizer result
is favorable to DP.

### Profile wall-clock bound and failure isolation

The nominal ten-second Cedar stage window starts only after the first output.
Ray profiling also waits for a ten-sample submission batch. On FreeLaw, the
terminal fixed-mapper profile consequently spent more than six hours running
the complete upstream long-document pipeline before its first batch. This was
unbounded warm-up, not a longer measured profiling window.

Before any held-out extension optimizer plan or execution result was produced,
the complete profile command for each workload was bounded at 3,600 wall-clock
seconds. A timeout or invalid profile is recorded as profile_timeout or
profile_failed; the workload remains a failure in the denominator and cannot
produce a speedup claim. Profile generation is isolated by workload, so one
unavailable profile cannot suppress valid measurements for other pre-registered
workloads. A resumed run may reuse a profile only when its original run log
records successful completion and the file passes the frozen resource-signature
and selectivity-schema validation.

### Pre-Code profile compatibility amendment

The first batch generated the shared `redpajama_code` profile before the
selectivity-aware profile schema was implemented. That file contains no
`input_counts`, `output_counts`, or `selectivities`, so the revised DP would
deliberately fall back to its old size-only objective. Before any Code plan or
execution measurement was generated, the serialized continuation was amended
to stop the legacy branch after EuroParl and rerun the complete Code workload
from a newly generated shared profile. This decision is based only on profile
schema compatibility, not on a Code optimizer result.

The legacy run directory remains untouched as provenance. The final audit
uses an explicit EuroParl-only ledger view of that directory, the complete
selectivity-aware Code run, and the complete four-workload extension run. It
requires all six registered candidates to be present, so this split cannot
silently remove a candidate from the denominator.

The final DP objective is the exact selection-aware additive work model. It
jointly evaluates operator work and measured Ray/SMP boundary work while the
DP state enforces the unchanged W=8/CPU-64 stage budget. The other in-scope
optimizers are unchanged; DP-two-stage is excluded as specified above. An
independent exhaustive oracle verifies the objective over all legal orders,
fusion partitions, backend assignments, and a constrained stage budget.

### Post-Code objective correction and held-out freeze

The three completed Code repetitions exposed that assuming ideal overlap
between all physical stages underprices real Cedar stage boundaries. The DP
plan ran in `507.168 +/- 0.742` seconds, while the single-Ray-stage Data-Juicer
plan ran in `459.747 +/- 1.036` seconds. Code is retained as a failed
development workload and cannot be removed from the denominator or converted
into a win.

Before any of the four registered extension workloads was profiled or
executed, the final held-out configuration was therefore frozen to the exact
selection-aware **additive** joint objective. It still jointly searches order,
fusion, backend assignment, and the W=8/CPU-64 resource state; only the
unsupported ideal-overlap assumption is removed. The four extension workloads
remain an unseen validation batch. Their runner metadata records
`dp_objective=additive`, and no further objective choice may be made from their
individual outcomes.

### Formal timeout and censored-runtime rule

This reporting clarification was recorded before any candidate execution
completed and originally used a 300-second plan threshold. The current public
protocol supersedes that threshold with 3,600 seconds: a plan optimization
exceeding 3,600 seconds is reported as
unavailable and provides no execution-time evidence. An execution exceeding
3,600 seconds is right-censored at 3,600 seconds.

A censored competitor may establish a conservative speedup lower bound only
when all three of its formal repetitions time out. For example, if DP
completes in at most 3,000 seconds and the best otherwise available competitor
times out in all three repetitions, then the measured evidence proves a DP
speedup of at least `3600 / 3000 = 1.20x`. One or two timeouts, a failed
process, a missing result, or an optimizer-plan timeout cannot establish this
runtime lower bound. Censored workloads remain in the denominator regardless
of whether the lower bound passes the 20% gate.

### Pre-measurement DP fail-fast rule

Before the fresh Code or extension runs produced any profile or optimizer
result, the serialized runner was configured to stop a workload when DP
cannot possibly supply the three successful repetitions required by this
protocol. This occurs if the DP plan is unavailable or if any DP execution
repetition fails, times out, or exhausts the source.

The workload remains in the registered denominator and is reported as a
failure; missing competitor cells are never imputed. The rule therefore
cannot create a DP win. It only avoids spending additional hours on a
candidate whose required DP mean can no longer be formed. All workloads for
which DP remains potentially valid still execute the complete six-optimizer
(`optimizer`, `dj_optimizer`, `dp_cedar_optimizer`, `dp_optimizer`,
`dp_two_stage_optimizer`, and `pecan_optimizer`),
three-repeat round-robin matrix.

### EuroParl post-timeout resource allocation

EuroParl's first Data-Juicer execution reached the frozen 3,600-second limit.
Its following DP-Cedar execution processed only about 1,600 of 20,000 outputs
after more than half an hour, but non-formal optimizer verification overlapped
part of that run. The DP-Cedar execution is therefore explicitly discarded as
`invalid_interference`; neither its partial throughput nor a projected timeout
is used as formal runtime evidence. The remaining legacy EuroParl repetitions
are terminated and the machine is transferred to the fresh, isolated
selectivity-aware Code and extension runs.

This operational decision cannot produce a favorable EuroParl result:
unexecuted cells are missing rather than imputed, the workload is retained in
the registered denominator, and the analyzer reports it as invalid/failed.
In particular, fewer than three timeout repetitions cannot be used as a
censored runtime lower bound under the rule above. The valid Data-Juicer
timeout, invalid DP-Cedar marker, and all partial logs remain in the run
directory.

### Pre-measurement feasibility amendment

Before any profile or optimizer plan was generated, downloading EuroParl
showed that 17,428 source records already occupied about 1.2 GiB. Therefore a
complete split could not satisfy the predeclared at-most-3-GiB policy. The
selection was amended to the same 2-GiB source-order-prefix rule already used
by the formal RedPajama C4 workload. This amendment was made solely from input
size, before observing any optimizer cost or execution result.
