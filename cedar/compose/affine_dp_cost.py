"""Affine ``kx+b`` compute pricing shared by the joint DP optimizers.

Every priced operator carries measured affine coefficients in
``physical_model.operator_affine``; the DP prices it as
``surviving_records * (k * input_bytes + b)``. Boundary, cache and transport
costs retain the DP's byte-volume model.
"""


class AffineDpCostMixin:
    """Keep only the parts of the affine objective the base optimizer lacks.

    ``MyOptimizer`` owns the affine accessors, the work-product recurrence and
    the per-operator cost hook. This mixin adds the profile contract check and
    the measured co-run residual the joint DP applies to local operators.
    """

    def _init_stats(self):
        super()._init_stats()
        if not getattr(self, "uses_affine_operator_cost", False):
            # Historical cost-model control: byte-volume compute pricing does
            # not read the fitted operator layer.
            self._dp_co_run_factors = {}
            return
        affine = self.profiled_stats.get("physical_model", {}).get(
            "operator_affine"
        )
        if not isinstance(affine, dict) or affine.get("schema_version") != 1:
            raise ValueError(
                "The joint DP prices every operator with fitted kx+b; "
                "physical_model.operator_affine schema_version=1 is required"
            )
        if not affine.get("operators"):
            raise ValueError(
                "physical_model.operator_affine contains no fitted operators"
            )
        calibration = self.profiled_stats.get("calibration", {})
        self._dp_co_run_factors = (
            calibration.get("co_run_factors", {})
            if isinstance(calibration, dict)
            else {}
        )
