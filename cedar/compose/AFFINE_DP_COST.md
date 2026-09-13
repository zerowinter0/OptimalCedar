# Joint DP with affine operator costs

Selector 2 (dp_optimizer) now collects the same local-only cm_model profile
section as selector 16. Worker profiling, width measurements, the joint subset
search, resource allocation, cache and transport models are retained.

For each candidate prefix, x is bytes per surviving record and q is the
surviving record fraction. Compute work is q * (k*x + b). Local fitted
coefficients use milliseconds per byte and milliseconds per record. Remote
backends retain DP's conservative measured/inferred baseline anchor, scaled by
the local curve ratio. Width-dependent worker measurements use the same curve.
Both slope and intercept are weighted by q; only x changes with content size.

PERDATA/PERRECORD annotations are ignored when cm_model is present. Missing or
unsupported operators in that section use an explicit constant estimate;
insufficient size variation in profiling produces k=0 rather than an invented
slope. For compatibility, profiles entirely missing cm_model emit a warning and
retain the historical model. Regenerate them with selector 2 to upgrade.

The existing outputs/cm_vs_cedar_simclr_20260911/profile.yaml contains cm_model
and can be reused for DP. Regenerate cm_model to use controlled local image
sweeps, validation and clamped extrapolation (see ../client/AFFINE_PROFILE.md).
Sizes cover the whole record, not individual modality fields. Multiplicative
size ratios and independent selectivities remain approximations. No additional
SMP/Ray size sweep is performed.
