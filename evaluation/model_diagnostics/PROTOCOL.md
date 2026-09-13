# Controlled family-aggregation / boundary diagnosis (2026-09-06)

These are new synthetic diagnostic workloads, not replacements for Chapter 6.
No optimizer is changed or declared better by construction. Existing results
and failed smoke-test logs are preserved.

## Questions

1. With a fixed CPU reservation, does same-Ray-family stage addition or the
   bottleneck of independent stages better explain end-to-end execution?
2. How much effective width-one processing time is worker compute, and how
   much is source/runtime/transport overhead? Could a separate boundary term
   count overhead already present in an effective coefficient?

## Controlled matrix

- Actual Cedar IterSource / Mapper / Ray variants, controller and optimizer
  bypassed to enforce specified physical plans. No cache, filter, reorder,
  native multithreading or batch-size search.
- Eight pinned local drivers; per-driver Ray widths 6, 3+3, or 2+2+2.
  All three have 48 remote actors and eight local drivers, within CPU_BUDGET=64.
  Remaining CPU budget is unreserved in every plan. Actor placement uses the
  existing remote-only `cedar_remote` resource; no cluster restart or global kill.
- Fixed 32,768 records per run (4,096 per driver), byte payloads 512/65,536/
  1,048,576. This spans latency-, mixed-, and transfer-heavy regimes. Input
  generation uses identical bytes; every record still traverses Cedar's
  ordinary batch-one serialization path (no explicit shared object references).
- Per-record PBKDF2 CPU iterations 0 / 1,200 / 120,000, split evenly across
  stages. Calls use fixed inputs independent of payload and no sleeps.
  Additional function setup from splitting is measured rather than assumed
  absent. The output preserves ID, bytes and a total-work marker; exact ID
  coverage, duplicates and work totals are checked on every run.
- 128-record warm-up epoch per driver on the same actors, excluded from all
  timing summaries. Formal run starts after an eight-driver barrier.
- Three rounds rotate stage order 1-2-3 / 2-3-1 / 3-1-2 within each workload.
  81 runs total. Instrumentation is identical across candidates.

## Frozen calibration

For each payload and stage iteration count 0/400/600/1,200/40,000/60,000/120,000:
measure local callable, Local Cedar pipeline and width-one Ray Cedar pipeline
for at least ten seconds each. Ray has one actor, batch one, and a separate
warm-up; measured streaming does not restart short epochs. Preserve delivered
sample counts, elapsed seconds and backend processing statistics. This single
calibration is shared by all three repetitions and stage configurations.
Width-one profiles and zero-compute runs provide separate overhead controls.
Callable-local timings are host-specific; remote worker timings are the
appropriate compute reference for the Ray plans.

## Artifacts / interpretation

`profile.json`, `profile.log`, `metadata.json`, per-case configurations/logs,
per-worker raw timings and progress, and each case's `result.json` are retained.
Case elapsed time is the maximum driver elapsed time; absolute monotonic start
timestamps allow barrier skew checks. Progress samples support a steady-state
analysis separate from fill/drain. They establish observed throughput, not
direct actor-overlap traces. Backend timers measure processing in each actor;
they do not by themselves separate network, marshaling and queueing.

The profiles can support sum/max and compute-only/effective-plus-boundary
comparisons. A discrepancy motivates a targeted model change; it does not
automatically attribute all error to one component or establish a result for
SMP. Native optimizer coefficient reconstruction and application validation
remain analysis steps after measurements finish.

Each case has a 3,600-second limit, with an earlier driver join deadline and
owned-process-group cleanup. A failed case stops the suite and marks FAILED;
data size or parameters are never reduced to force success. Profiling has a
1,800-second deadline. `STATUS=COMPLETE` means all requested measurements
finished; otherwise inspect STATUS and the named case log. Smoke tests use
small inputs solely to validate mechanics and are kept in separate directories.
