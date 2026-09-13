"""Cedar's staged optimizer with a locally profiled affine cost model."""
from contextlib import contextmanager
import math

from cedar.compose.optimizer import Optimizer
from cedar.compose.affine_cost_utils import affine_value
from cedar.compose.utils import topological_sort
from cedar.pipes import FilterPipe
from cedar.pipes.context import PipeVariantType


def entry(mapping, pid, default=None):
    return mapping.get(pid, mapping.get(str(pid), default))


class CmOptimizer(Optimizer):
    """Inherit all Cedar search, offload, fusion, cache and allocation passes."""

    def _init_stats(self):
        super()._init_stats()
        section = self.profiled_stats.get("cm_model", {})
        if section.get("schema_version") != 1:
            raise ValueError("cm_optimizer requires a cm_model profile; profile with selector 16")
        self._cm_models = section["operators"]
        self._cm_selectivities = {}
        baseline = self.profiled_stats["baseline"]
        for pid, pipe in self.logical_pipes.items():
            model = entry(self._cm_models, pid, {})
            if model.get("pipe_name") not in (None, pipe.get_logical_name()):
                raise ValueError(f"CM profile does not match pipe {pid}")
            if "k" in model:
                if any(not math.isfinite(float(model[k])) or float(model[k]) < 0
                       for k in ("k", "b")):
                    raise ValueError(f"Invalid affine coefficients for pipe {pid}")
            selectivity = entry(baseline.get("selectivities", {}), pid)
            if selectivity is None:
                selectivity = model.get("selectivity", 1.0)
            selectivity = float(selectivity) if isinstance(pipe, FilterPipe) else 1.0
            if not math.isfinite(selectivity) or not 0 <= selectivity <= 1:
                raise ValueError(f"Invalid selectivity for pipe {pid}")
            self._cm_selectivities[pid] = selectivity
            # Per-record sizes and cardinality are separate quantities.
            x, y = model.get("input_mean_bytes", 0), model.get("output_mean_bytes")
            if x > 0 and y is not None:
                self._data_size_ratio_map[pid] = float(y) / float(x)
        self._cm_baseline_reach = {}
        reach = 1.0
        for pid in topological_sort(self.logical_graph):
            self._cm_baseline_reach[pid] = reach
            reach *= self._cm_selectivities[pid]
        self._cm_reach = {}

    @contextmanager
    def _cost_context(self, graph, physical_specs=None, plan=None):
        previous = self._cm_reach
        self._cm_reach = {}
        reach = 1.0
        descriptors = plan.pipe_descs if plan is not None else self.physical_plan.pipe_descs
        try:
            for pid in topological_sort(graph):
                desc = (physical_specs or {}).get(pid, descriptors.get(pid))
                members = ([pid] if pid in self.logical_pipes else
                           list(getattr(desc, "fused_pipes", None) or []))
                for member in members:
                    self._cm_reach[member] = reach
                    reach *= self._cm_selectivities.get(member, 1.0)
            yield
        finally:
            self._cm_reach = previous

    def calculate_cost(self, graph, physical_specs=None, fused_pipes=None,
                       caching_on=False, plan=None):
        with self._cost_context(graph, physical_specs, plan):
            return super().calculate_cost(graph, physical_specs, fused_pipes,
                                          caching_on, plan)

    def calculate_final_plan_cost_breakdown(self, plan=None):
        active = plan if plan is not None else self.physical_plan
        with self._cost_context(active.graph, active.pipe_descs, active):
            return super().calculate_final_plan_cost_breakdown(plan)

    def _calculate_pipe_cost(self, p_id, input_size, desc):
        model = entry(self._cm_models, p_id, {})
        size0 = self.profiled_stats["baseline"]["input_sizes"][p_id]
        original_reach = self._cm_baseline_reach.get(p_id, 1.0)
        reach = self._cm_reach.get(p_id, 1.0)
        if reach == 0:
            return 0.0
        if "k" in model and "b" in model:
            local = affine_value(model, input_size)
            reference = max(1e-12, affine_value(model, size0))
        else:
            # Unsupported/unobserved operators retain an explicit Cedar fallback.
            reference = self._base_cost_map[p_id] / max(original_reach, 1e-12)
            local = reference * input_size / size0 if size0 > 0 else reference
        if desc is None or desc.variant_type in (None, PipeVariantType.INPROCESS):
            return reach * local
        measurement = entry(self.profiled_stats.get("offloads", {}).get(
            desc.variant_type.name, {}), p_id, {})
        direct = measurement.get("backend_compute", {})
        mean = direct.get("mean_ms_per_sample")
        if (direct.get("count", 0) > 0 and mean is not None
                and math.isfinite(mean) and mean >= 0):
            # DP's worker measurement anchors backend cost; local shape is shared.
            return reach * mean * local / max(reference, 1e-12)
        old = super()._calculate_pipe_cost(p_id, size0, desc)
        if math.isfinite(old) and old > 0:
            return reach * old / max(original_reach, 1e-12) * local / max(reference, 1e-12)
        return reach * local


CMOptimizer = CmOptimizer
