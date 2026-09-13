"""Affine per-record compute model for the joint DP optimizer.

All compute multipliers include cardinality separately from per-record bytes.
Boundary, cache and transport costs retain the DP's byte-volume model.
"""
import logging
import math
import os

from .optimizer import Optimizer, PipeVariantType
from .utils import topological_sort
from .affine_cost_utils import affine_value


def _entry(mapping, pid, default=None):
    return mapping.get(pid, mapping.get(str(pid), default))


class AffineDpCostMixin:
    _dp_affine_enabled = False


    def _dp_co_run_factor(self, p_id: int) -> float:
        """Return the measured co-run residual for one local operator.

        Local operators are profiled in isolation but execute next to the other
        workers and stages of the materialized plan. Replaying an executed plan
        against its own per-stage trace (calibration.co_run_factors) measures
        that residual as the median in-plan service divided by the modeled
        service. Values are close to one for compute-bound operators and above
        one for contention-sensitive ones such as the image reader.
        """
        factors = getattr(self, "_dp_co_run_factors", None)
        if not factors:
            return 1.0
        raw = factors.get(p_id, factors.get(str(p_id)))
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return 1.0
        if not math.isfinite(value) or value <= 0.0:
            return 1.0
        return value

    def _init_stats(self):
        super()._init_stats()
        section = self.profiled_stats.get("cm_model", {})
        # ``CEDAR_DP_DISABLE_AFFINE=1`` prices the DP with the same per-pipe
        # trace latencies every other planner reads, instead of the isolated
        # affine fits, so a run can compare all systems on one cost basis.  The
        # calibrated layers (scaling, boundary, transport) are unaffected.
        if os.environ.get("CEDAR_DP_DISABLE_AFFINE", "").strip() in (
            "1",
            "true",
            "True",
            "yes",
        ):
            section = {}
        if section and section.get("schema_version") != 1:
            raise ValueError("Unsupported cm_model schema; regenerate profile with selector 2")
        self._dp_affine_enabled = bool(section)
        self._dp_affine_models = section.get("operators", {})
        calibration = self.profiled_stats.get("calibration", {})
        self._dp_co_run_factors = (
            calibration.get("co_run_factors", {})
            if isinstance(calibration, dict)
            else {}
        )
        if not section:
            logging.getLogger(__name__).warning(
                "Profile has no cm_model: retaining the legacy cost model for compatibility; "
                "regenerate with selector 2 to enable affine costs.")
            return
        reach = 1.0
        self._dp_affine_baseline_reach = {}
        selectivities = self._dp_observed_selectivities()
        for pid in topological_sort(self.logical_graph):
            self._dp_affine_baseline_reach[pid] = reach
            reach *= float(_entry(selectivities, pid, 1.0))
            model = _entry(self._dp_affine_models, pid, {})
            if model.get("pipe_name") not in (None, self.logical_pipes[pid].get_logical_name()):
                raise ValueError(f"CM profile does not match pipe {pid}")
            if ("k" in model) != ("b" in model):
                raise ValueError(f"Incomplete affine coefficients for pipe {pid}")
            if "k" in model and any(
                not math.isfinite(float(model[key])) or float(model[key]) < 0
                for key in ("k", "b")
            ):
                raise ValueError(f"Invalid affine coefficients for pipe {pid}")
            x, y = model.get("input_mean_bytes", 0), model.get("output_mean_bytes")
            if x > 0 and y is not None:
                self._data_size_ratio_map[pid] = float(y) / float(x)

    def _dp_affine_value(self, pid, size):
        model = _entry(self._dp_affine_models, pid, {})
        if "k" in model:
            return affine_value(model, size)
        # Missing/unsupported observations do not imply proportional byte work.
        reach = self._dp_affine_baseline_reach.get(pid, 1.0)
        return max(0.0, self._base_cost_map[pid]) / max(reach, 1e-12)

    def _dp_affine_worker_cost(self, pid, mean, input_size):
        reference = self.profiled_stats["baseline"]["input_sizes"][pid]
        y0 = self._dp_affine_value(pid, reference)
        shape = self._dp_affine_value(pid, input_size) / y0 if y0 > 0 else 1.0
        return mean * (self._dp_affine_baseline_reach.get(pid, 1.0) or 1.0) * shape

    def _calculate_pipe_cost(self, p_id, input_size, desc):
        if not self._dp_affine_enabled:
            return super()._calculate_pipe_cost(p_id, input_size, desc)
        reach = self._dp_affine_baseline_reach.get(p_id, 1.0) or 1.0
        local = (
            reach
            * self._dp_affine_value(p_id, input_size)
            * self._dp_co_run_factor(p_id)
        )
        if desc is None or desc.variant_type in (None, PipeVariantType.INPROCESS):
            width = self._dp_profiled_width_compute_cost(
                p_id, PipeVariantType.SMP, input_size)
            return max(local, width) if width is not None else local
        vt = desc.variant_type
        baseline_input = self.profiled_stats["baseline"]["input_sizes"][p_id]
        backend = _entry(self.profiled_stats.get("offloads", {}).get(vt.name, {}), p_id, {})
        direct = backend.get("backend_compute", {})
        candidates = []
        mean = direct.get("mean_ms_per_sample")
        if (direct.get("count", 0) > 0 and mean is not None
                and math.isfinite(float(mean)) and float(mean) >= 0):
            candidates.append(self._dp_affine_worker_cost(p_id, float(mean), input_size))
        # Preserve DP's conservative backend anchor; replace only size scaling.
        inferred = Optimizer._calculate_pipe_cost(self, p_id, baseline_input, desc)
        if math.isfinite(inferred) and inferred > 0:
            reference = self._dp_affine_value(p_id, baseline_input)
            shape = self._dp_affine_value(p_id, input_size) / reference if reference > 0 else 1.0
            candidates.append(inferred * shape)
        width = self._dp_profiled_width_compute_cost(p_id, vt, input_size)
        if width is not None:
            candidates.append(width)
        return max(candidates) if candidates else local

    def _dp_compute_work_prod(self, mask, operator_idx=None):
        if not self._dp_affine_enabled:
            return super()._dp_compute_work_prod(mask, operator_idx)
        if operator_idx is None:
            return self._dp_work_prod(mask)
        pid = self._dp_inner_ops[operator_idx]
        source_size = self.profiled_stats["baseline"]["output_sizes"][self._get_source_p_id()]
        size = source_size * self._dp_r_prod[mask]
        return self._dp_cardinality_prod[mask] * self._dp_affine_value(pid, size)

    def _dp_compute_cost_denominator(self, operator_idx, baseline_input_size, source_size):
        if not self._dp_affine_enabled:
            return super()._dp_compute_cost_denominator(operator_idx, baseline_input_size, source_size)
        pid = self._dp_inner_ops[operator_idx]
        denominator = (source_size * (self._dp_affine_baseline_reach.get(pid, 1.0) or 1.0)
                       * self._dp_affine_value(pid, baseline_input_size))
        # Zero fitted cost and zero original reach must not make placement infeasible.
        return denominator if denominator > 0 else source_size
