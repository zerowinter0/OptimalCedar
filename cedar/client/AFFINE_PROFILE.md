# Controlled local affine profiling

Selectors 2 (DP) and 16 (CM) share this profiling path. The ordinary enhanced
backend profile is unchanged; an additional natural-input pass records actual
cardinality, per-record input/output sizes, timings and CPU image snapshots.

For image mapper/filter inputs (PIL or CPU CHW tensors, optionally an explicitly
identified field), the additional pass:
- Holds up to six immutable snapshots per operator, with a 64 MiB shared cap.
- Preserves dtype, channels, aspect ratio, other fields and original snapshots.
- Constructs ten linearly spaced pixel-area targets, small-size points and
  natural-size points. Upper support covers the per-operator natural 95th
  percentiles and four times the target operator's median, capped at 2048².
- Holds out three intermediate sizes for validation.
- Randomizes size order and rotates input content. Each size is warmed up.
- Uses at least three rounds, four calls per round, and 20 ms accumulated
  callable time where possible, capped at 256 calls. No reset of random seeds
  per call forces stochastic branches into a particular outcome.
- Adds rounds (up to six) when timing CV exceeds 15%. A soft wall-time budget
  per operator defaults to the profile duration (10 seconds), configurable via
  CEDAR_CM_SWEEP_TIME_SEC. A running callable cannot be preempted safely.
- Excludes input construction, copies and size measurement from timing.
- Restores Python, NumPy and Torch CPU RNG states after each sweep.

Fits use equal weight per target size, retain small-size points without bin
merging, and constrain k,b >= 0. The profile also saves the unconstrained fit,
raw timing batches, validation errors, timing CV, actual measured byte range,
and the untouched natural fit. Maximum held-out relative error above 30% or
timing CV above 15% marks the fit low_confidence. This flag is diagnostic; the
curve remains usable within its measured range. New controlled-sweep curves
are clamped to measured support by both DP and CM rather than extrapolated.

Only timing coefficients change. Natural input/output mean sizes, selectivity
and counts are retained for propagation. The size axis remains whole-record
bytes, even when one identified field is resized. Joint changes of multiple
fields and semantic validity after an arbitrary reorder are not established.

Batcher uses the configured batch size and the native batching iterator.
Batch size 1 is a passthrough and retains its natural timing. The batching
microbenchmark includes iterator setup, unlike an isolated stack-only kernel.

ImageReader no longer fits decode time against pathname-object bytes. It
currently retains a constant natural estimate with an explicit reason; fitting
file-byte/decoded-pixel/format axes requires a separate optimizer size axis.
Unsupported text/token/GPU inputs retain their natural fit with a reason;
arbitrary text or token repetition is not introduced.

Incomplete sweeps (fewer than three rounds for any target) retain the natural
fit and store partial measurements for diagnosis. Old profiles keep their
existing semantics; regenerate cm_model to activate this procedure.
