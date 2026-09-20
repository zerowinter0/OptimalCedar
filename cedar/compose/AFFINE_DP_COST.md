# Joint DP with affine operator costs

Every operator is priced by one measured equation, ``cost(record) =
k * input_bytes + b``. The joint DP and the layered Simple-DP/PICO variants
read the same fitted layer, ``physical_model.operator_affine``, which the
default layered profiling protocol produces. Worker profiling, width
measurements, the joint subset search, resource allocation, cache and
transport models are retained.

For each candidate prefix, x is bytes per surviving record and q is the
surviving record fraction. Compute work is q * (k*x + b). Local fitted
coefficients use milliseconds per byte and milliseconds per record. Remote
backends keep the isolated worker measurement as the anchor, rescaled to the
candidate input size by the same curve. Width-dependent worker measurements use
the same curve. Both slope and intercept are weighted by q; only x changes with
content size.

There is no per-data/per-record classification any more: an operator whose cost
is flat in the payload is simply the k = 0 member of the same family, and
profiling fits it instead of abstaining. Operators with no measured
coefficients are listed in ``physical_model.operator_affine.unfitted_operators``;
an optimizer that is asked to price one raises instead of falling back to
byte-proportional compute. Cost controls that must reproduce Cedar's historical
model (``old_dp_optimizer``, ``old_dp_boundary``, ``SimpleDpOptimizer``,
``DpTwoStageOptimizer``, ``ExpOptimizer``) declare
``uses_affine_operator_cost = False`` and are the only paths that still price
compute by serialized byte volume.

Historical profiles may still carry ``cm_model`` (selector 2) and
``operator_compute_scaling`` entries; both are ignored by the DP. Regenerate the
layered profile to obtain the kx+b layer, using controlled local image sweeps,
validation and clamped extrapolation (see ../client/AFFINE_PROFILE.md). Sizes
cover the whole record, not individual modality fields. Multiplicative size
ratios and independent selectivities remain approximations. No additional
SMP/Ray size sweep is performed.
