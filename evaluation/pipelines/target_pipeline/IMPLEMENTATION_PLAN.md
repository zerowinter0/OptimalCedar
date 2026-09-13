# Target pipeline migration plan

Goal: five executable, source-attributed Cedar preprocessing workloads:
SimCLR two-view (existing Cedar transform chain), DINO multi-crop,
SwAV multi-crop, Transformers CLIP image/text, BLIP pretraining image/text.

Scope: additive files under this directory; preserve existing pipelines and
optimizer settings. Hub 47fc345 was audited and supplies no admitted recipe.
Independent view/field chains may interleave; preserve all within-chain
dependencies. This is not a claim of within-view transform commutativity.

- [ ] Add tests for legal-order equivalence, output shapes, nontrivial size
  changes, source-reference output, and actual Cedar DAG constraints.
- [ ] Implement named stages, deterministic per-record random streams,
  metadata and a shared Cedar dataset entry point.
- [ ] Migrate the five recipes at their source crop counts/resolutions.
- [ ] Pin public source snapshots and licenses in provenance metadata.
- [ ] Run equivalence checks and Cedar execution smoke checks in Docker.
- [ ] Document inputs, operators, limitations, and the verification results.

Formal experiments are not launched. W=8, CPU_BUDGET=64, one actor/process
per profiling stage, ten-second profiling, and three round-robin repetitions
remain requirements of the existing experiment harness. Smoke fixtures are
validation-only, never formal data.
