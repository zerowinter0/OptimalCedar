# cm_optimizer

Selector: 16 (OptimizerOptions.use_my_optimizer=16 or eval_cedar
--use_my_optimizer 16). compare_optimizer_perf accepts cm_optimizer.
No default optimizer list was changed.

CmOptimizer inherits Cedar Optimizer's reordering enumeration, staged offload,
fusion, caching, prefetch and worker allocation. It does not use DP search.

## Profile

Selector 16, or CEDAR_PROFILE_CM=1 for a shared profile, runs the enhanced
profiling path used by dp_optimizer, with one actor/process per stage.
Default stage duration remains 10 seconds. Worker-side offload measurements,
wall latencies, filter counts and existing physical metadata are preserved.

An additional local-only pass captures real intermediate inputs and their natural
statistics, then sweeps supported CPU image sizes. Input construction and sizing
are outside the callable timer. See ../client/AFFINE_PROFILE.md for sampling,
validation, budgets, supported types and limitations. The ordinary baseline is
not changed. Local/native pools use one thread; rejected records remain counted.

cm_model.operators[pipe_id] stores:
- k in ms/byte and b in ms/record;
- sample count, measured size range, mean input/output bytes;
- nonnegative least-squares fit, R-squared and up to 12 linear size-bin means;
- filter input/output counts and selectivity;
- method, tag and fallback reason.

At most 4096 natural timing observations per callable are retained. Supported
CPU images additionally use immutable snapshots and controlled size scans.
Insufficient natural size variation falls back to a constant only when no
complete supported sweep is available. Such fallback does not establish
size-independence. New sweep models include held-out validation and noise flags.
Missing observations retain an explicitly unavailable model and Cedar fallback.
Old profiles without cm_model are rejected by cm_optimizer.

Input x is the whole-record size using Cedar's size measurement, not a
modality-specific field size. Native readers therefore retain Cedar's path
representation sizing. This implementation is the requested scalar affine
model, not a field-aware multimodal size propagation system.

## Cost

For a candidate prefix:
- S is estimated bytes per remaining record;
- q is the product of predecessor filter selectivities;
- current operator contribution is q * (k*S+b), ms per source record;
- S is updated using measured per-record output/input size ratios;
- q is updated separately using filter selectivity.

Filter costs include rejected records. q=0 makes downstream cost zero,
including intercepts. The baseline ratios and selectivities are treated as
order-independent estimates; correlated filters and fixed-output transforms
can violate this approximation. New controlled-sweep models clamp predictions
to their measured input range; older models retain their historical behavior.
Fit diagnostics remain in the profile.

For remote variants, the local curve's relative change is shared across
backends, anchored to the enhanced profile's worker compute mean. If absent,
a valid Cedar Amdahl estimate is used; an invalid estimate falls back to local
cost. Source costs and Cedar's other physical costing heuristics remain intact.
There is no additional SMP/Ray size sweep.

Cedar's old k=1,b=0 description refers to normalized cost ratio versus input
size ratio. The new stored coefficients instead have explicit ms/byte and
ms/record units.

## Usage

Generate a new profile using the workload's normal profiling entrypoint and
selector 16. Pass that YAML to the same workload with selector 16 to optimize.
Use CEDAR_PROFILE_CM=1 to add the model while preparing a common DP/Cedar
comparison profile. The extra profile fields are ignored by the old optimizer.
The original workload's data, dependencies and transform parameters are unchanged.
