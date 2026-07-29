# Selectivity-aware joint DP design

## Motivation

Cedar's baseline profile records per-pipe latency and average serialized item
size only for samples that reach the output. For a `FilterPipe`, a rejected
sample disappears inside the variant and therefore never reaches
`FeatureProfiler.update_ds`. Consequently,
`baseline.output_sizes[p] / baseline.input_sizes[p]` is an item-size ratio,
not a filter pass ratio.

The current DP uses that item-size ratio as `_dp_r_prod` for both byte-size
scaling and downstream work scaling. It therefore cannot correctly value
putting a cheap selective filter before an expensive language-model or
perplexity operator. This matters especially for the official Data-Juicer
recipes, which contain long chains of filters with heterogeneous costs.

## Profile extension

Selection statistics must be collected during the existing shared profile;
there must be no DP-only data pass.

1. Only when `CEDAR_PROFILE_FILTER_SELECTIVITY=1`, `_profile_feature` wraps
   each `FilterPipe` predicate in a counting callable before materializing
   the baseline or per-operator Ray/SMP profile variant. This keeps the
   instrumentation overhead consistent across the existing profile passes.
   Normal execution variants are unchanged and incur no counter branch.
2. The logical predicates are restored immediately after materialization.
   Before `feature_to_profile.reset()`, `_profile_feature` reads the counters
   retained by those wrappers.
3. Backend mutations can change how far a ten-second pass advances through
   the same source-order prefix. For each filter, the final baseline field
   uses the single pass with the largest observed input count. Counts are
   never summed because the passes revisit overlapping source prefixes.
4. The baseline profile adds:

   ```yaml
   input_counts:
     <pipe_id>: <count>
   output_counts:
     <pipe_id>: <count>
   selectivities:
     <pipe_id>: <output_count / input_count>
   selectivity_observation_sources:
     <pipe_id>: <baseline | RAY:profiled_pipe_id | SMP:profiled_pipe_id>
   ```

5. Non-filter pipes and every operator in an old profile default to
   selectivity `1.0`.
6. Old profiles remain readable and imply selectivity `1.0`; formal
   selectivity-aware results require regenerated shared profiles.

The formal profile fixes one local worker. A filter mutated into a remote
Ray/SMP stage retains identical instrumentation overhead, but its counter
copy is not visible in the driver and therefore reports zero there. Other
local filters in that pass remain valid observations; the maximum-coverage
selection naturally ignores a zero-count remote observation when any local
observation is available.

## DP state and cost

Maintain two independent subset products:

- `B[S]`: per-item byte-size multiplier after operators in subset `S`;
- `N[S]`: surviving-item multiplier after operators in subset `S`.

For an operator or fused block appended after prefix `S`, its work per source
sample is scaled by both the number and size of surviving inputs:

```text
operator_work(S, block) =
    N[S] * B[S] * profiled_per_byte_work(block)
```

For a parallel stage with input prefix `S` and output prefix `T`:

```text
boundary_work(S, T) =
    (N[S] * input_bytes(S) + N[T] * output_bytes(T))
    / measured_boundary_bytes_per_second
```

Transport feasibility still uses per-item size `B`, while aggregate transport
rate uses both `N` and `B`. Cache read work uses the cached prefix's surviving
item count and serialized item size. The final selectivity is invariant under
a semantics-preserving reorder, so minimizing work per source sample also
minimizes work per retained output.

Within a fused block, its exact topological-order recurrence must multiply
each later operator by both preceding products. Treating the entire block as
having one selection ratio is insufficient because filter order is the
optimization decision.

## Additive work objective

The joint DP minimizes total selection-aware operator work plus measured
Ray/SMP stage-boundary work. Its state additionally records parallel-stage CPU
use, so the fixed W=8, CPU-budget-64 constraint is enforced while comparing
otherwise equivalent prefixes. Cache replacement is evaluated in the same
additive objective by replacing the work of the cached prefix in later epochs.

For each legal prefix, the outer search enumerates only its actual previous
subsets instead of scanning every pair of legal masks. This covers the same
transitions while reducing the weak-dependency bound from approximately
`4^n` mask pairs to `3^n` subset transitions.

## Implementation status

The implemented slices are:

- the shared baseline profile emits conditional filter counts and
  selectivities, using the highest-coverage observation from the already
  scheduled baseline/Ray/SMP passes;
- `_dp_r_prod` remains the per-item size product for transport feasibility;
- `_dp_cardinality_prod` and `_dp_volume_prod` represent `N[S]` and
  `N[S] * B[S]`;
- exact within-block work, outer-block work, parallel-boundary work, and cache
  read work use the volume product;
- profiles without `selectivities` reproduce the old size-only objective.
- all workloads use the exact selection-aware additive objective.

A selectivity-aware revision of the heuristic aggregate-rate feasibility
guard remains a separate follow-up change.

## Required validation

- Unit-test filter counters for accepted, rejected, and exception behavior.
- Verify profile YAML count and selectivity fields on a deterministic toy
  pipeline.
- Exhaustively compare the selectivity-aware DP against brute force on small
  dependency graphs.
- Confirm equivalence to the old DP when every selectivity is `1.0`.
- Regenerate one shared formal profile per workload and validate W=8,
  CPU budget 64, and one Ray actor/SMP process per stage.
- Rerun every affected optimizer from the same regenerated profile; report
  optimization time separately and retain all timeout/failure outcomes.

## Validation evidence

As of 2026-07-25:

- the deterministic toy profile test verifies conditional counts of
  `4 -> 2 -> 1` and restoration of the original predicates;
- profile aggregation tests verify maximum-coverage selection, deterministic
  baseline tie-breaking, and count collection in a mutated profile pass;
- the focused optimizer suite has 27 passing tests across optimality, cache,
  fusion, two-stage, profile, recipe-registry, and reporting behavior;
- `verify_dp_optimizer_optimality.py --num-cases 5 --seed 20260725`
  exhaustively enumerated 2,045,952 legal order/fusion/backend plans across
  five random dependency graphs and matched the DP optimum in every case;
- a separate exhaustive case constrained to at most two parallel stages
  matched the resource-indexed DP optimum, covering the same mechanism
  that enforces the formal per-worker stage limit of seven at W=8/CPU-64;
- held-out workloads and cache-enabled workloads use the same additive cost
  semantics, avoiding an experiment-only objective switch.
