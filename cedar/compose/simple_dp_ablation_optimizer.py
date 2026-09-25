"""Simple-DP variants and retained historical implementation helpers.

The current boundary, workers-boundary and workers-width-boundary variants
use isolated layered-profile compute data. OldDpBoundaryOptimizer preserves
the otherwise identical Cedar whole-pipeline/Amdahl cost path as a control.
"""
from dataclasses import dataclass
import logging
import os
import math
import time
from typing import Any, Dict, List, Optional, Tuple

from cedar.pipes import PipeVariantType, PipeVariantContextFactory
from .my_optimizer import MyOptimizer
from .optimizer import Optimizer
from .dp_optimizer import DpOptimizer, DpObjectiveCost, DpResourceUsage
from .simple_dp_optimizer import SimpleDpOptimizer


logger = logging.getLogger(__name__)


class SimpleDpWorkersOptimizer(SimpleDpOptimizer):
    def _physical_opt(self):
        super()._physical_opt()
        if not self.options.enable_local_parallelism:
            return
        plan = self.physical_plan
        ray = sum(plan.pipe_descs[p].variant_type in
                  (PipeVariantType.RAY, PipeVariantType.TF_RAY) for p in plan.graph)
        smp = sum(plan.pipe_descs[p].variant_type == PipeVariantType.SMP
                  for p in plan.graph)
        candidates = []
        for workers in range(1, self._cap_local_workers(
                self.options.available_local_cpus) + 1):
            limits = self._dp_limits_for_workers(workers)
            if limits is None or limits.ray_cpus < ray or limits.smp_cpus < smp:
                continue
            candidates.append((self._last_dp_state_cost / workers, workers))
        if not candidates:
            raise RuntimeError('No feasible W for the Cedar-cost plan')
        _, workers = min(candidates)
        self._worker_search_evidence = candidates
        plan.set_local_workers(workers)
        self._dp_selected_workers = workers
        self._allocate_final_remote_stage_resources()
        self._tune_final_ray_stage_contexts()


class OldDpBoundaryOptimizer(SimpleDpOptimizer):
    """Boundary-aware Simple-DP evaluated with Cedar's legacy profile."""

    cedar_objective = False

    def _fusion_cost_ratio(self, order):
        return 1.0

    def _dp_regular_transition_cost(self, prev_mask, block):
        # Only the requested fixed + bytes/throughput boundary. Do not call
        # PICO's newer object-boundary, cardinality, or inflight models.
        source_size = self.profiled_stats['baseline']['output_sizes'][
            self._get_source_p_id()]
        input_size = source_size * self._dp_r_prod[prev_mask]
        output_size = source_size * self._dp_r_prod[prev_mask | block.mask]
        return block.cost + self._dp_boundary_cost_ms(
            block.variant, input_size, output_size)


class _LayeredProfileCostMixin:
    """Price operators only from the isolated layered profile.

    The dual profile retains Cedar's whole-pipeline ``throughput`` values so
    legacy baselines can share one profiling run.  These optimizers must not
    read those values: local compute comes from the affine/wall-clock layer,
    and backend compute comes from isolated actor/process measurements.
    """

    uses_affine_operator_cost = True

    @staticmethod
    def _profile_entry(mapping, p_id):
        if not isinstance(mapping, dict):
            return None
        return mapping.get(p_id, mapping.get(str(p_id)))

    @staticmethod
    def _valid_backend_compute(
        entry, allow_unconverged: bool = False, label: str = ""
    ):
        """Measured backend compute of one operator, or None when unusable.

        The search only accepts measurements that reached their convergence
        target.  Pricing a materialized plan (``allow_unconverged``) also
        accepts a measurement whose adaptive run stopped at the time cap: that
        plan exists and has to be ranked, and dropping the measurement would
        silently replace it with the local cost instead.
        """
        if not isinstance(entry, dict):
            return None
        direct = entry.get("backend_compute")
        if not isinstance(direct, dict):
            return None
        try:
            mean = float(direct["mean_ms_per_sample"])
            count = int(direct.get("count", 0))
        except (KeyError, TypeError, ValueError):
            return None
        adaptive = direct.get("adaptive_profile", {})
        unconverged = bool(
            isinstance(adaptive, dict)
            and adaptive
            and adaptive.get("converged") is not True
        )
        if count <= 0 or not math.isfinite(mean) or mean < 0.0:
            return None
        if unconverged:
            if not allow_unconverged:
                return None
            logger.warning(
                "[LayeredSimpleDp] pricing %s with an unconverged backend "
                "measurement: mean=%.4f ms/sample count=%d rse=%s stop=%s",
                label or "a pipe",
                mean,
                count,
                adaptive.get("observed_rse"),
                adaptive.get("stop_reason"),
            )
        return mean

    def _init_stats(self):
        Optimizer._init_stats(self)
        affine = self.profiled_stats.get("physical_model", {}).get(
            "operator_affine"
        )
        if not isinstance(affine, dict) or affine.get("schema_version") != 1:
            raise ValueError(
                "Layered Simple-DP requires physical_model.operator_affine "
                "schema_version=1; regenerate the profile in layered mode."
            )
        if not self.profiled_stats.get("baseline", {}).get("input_sizes"):
            raise ValueError(
                "Layered Simple-DP requires baseline input sizes to price "
                "each operator's fitted kx+b"
            )
        for p_id in list(self._base_cost_map):
            pipe = self.logical_pipes.get(p_id)
            if pipe is not None and pipe.is_source():
                # The source stage is never priced by the DP and has no
                # measured callable, so its baseline attribution stays as
                # profiled.
                continue
            # Kept up to date for callers that read the baseline map; the DP
            # itself prices every operator through the affine cost hook.
            reference = self.profiled_stats["baseline"]["input_sizes"][p_id]
            self._base_cost_map[p_id] = self._dp_affine_value(
                p_id, reference
            )

    def _dp_compute_work_prod(self, mask, operator_idx=None):
        # The layered variants price compute with the fitted kx+b recurrence
        # instead of inheriting SimpleDpOptimizer's byte-volume product.
        return MyOptimizer._dp_compute_work_prod(self, mask, operator_idx)

    def _dp_compute_cost_denominator(
        self, operator_idx, baseline_input_size, source_size
    ):
        return MyOptimizer._dp_compute_cost_denominator(
            self, operator_idx, baseline_input_size, source_size
        )

    def _calculate_pipe_cost(self, p_id, input_size, desc):
        """Price one operator from isolated measurements with its fitted kx+b.

        The layered variants deliberately ignore Cedar's whole-pipeline
        throughput entries: local compute comes from the fitted affine layer
        and backend compute from isolated actor/process measurements, both
        rescaled to candidate input sizes by the same affine shape.
        """
        local = (
            self._dp_affine_value(p_id, input_size)
            * self._dp_co_run_factor(p_id)
        )
        if desc is None or desc.variant_type in (
            None,
            PipeVariantType.INPROCESS,
        ):
            return local
        entries = self.profiled_stats.get("offloads", {}).get(
            desc.variant_type.name, {}
        )
        entry = self._profile_entry(entries, p_id)
        mean = self._valid_backend_compute(
            entry,
            allow_unconverged=self._dp_lenient_replay(),
            label=f"pipe {p_id} on {desc.variant_type.name}",
        )
        if mean is None:
            raise RuntimeError(
                f"Pipe {p_id} has no valid layered "
                f"{desc.variant_type.name} backend_compute measurement"
            )
        # The isolated backend measurement is authoritative for this
        # placement; the fitted kx+b only rescales it to the candidate input
        # size. Width candidates are priced by the DP provider from the
        # measured scaling curves.
        return self._dp_affine_worker_cost(p_id, mean, input_size)

    def _layered_boundary_cost_ms(self, prev_mask, block):
        throughput = self._dp_boundary_throughput(block.variant)
        if throughput is None:
            return 0.0
        source_size = self.profiled_stats["baseline"]["output_sizes"][
            self._get_source_p_id()
        ]
        next_mask = prev_mask | block.mask
        input_item = source_size * self._dp_r_prod[prev_mask]
        output_item = source_size * self._dp_r_prod[next_mask]
        input_reach = self._dp_cardinality_prod[prev_mask]
        output_reach = self._dp_cardinality_prod[next_mask]
        batch = (
            self._dp_ray_submit_batch_size(input_item, output_item)
            if block.variant in (PipeVariantType.RAY, PipeVariantType.TF_RAY)
            else 1
        )
        fixed = input_reach * self._dp_boundary_fixed_latency_ms(
            block.variant
        ) / batch
        bytes_per_source = (
            input_item * input_reach + output_item * output_reach
        )
        return fixed + bytes_per_source / throughput * 1000.0


class SimpleDpBoundaryOptimizer(
    _LayeredProfileCostMixin, OldDpBoundaryOptimizer
):
    """Boundary-aware Simple-DP using only the new layered profile."""

    cedar_objective = False

    def _dp_regular_transition_cost(self, prev_mask, block):
        return block.cost + self._layered_boundary_cost_ms(prev_mask, block)


class LayeredSimpleDpOptimizer(SimpleDpBoundaryOptimizer):
    """New-profile scalar DP without any stage-boundary charge."""

    def _dp_regular_transition_cost(self, prev_mask, block):
        return block.cost


class _RepresentationComputeMixin:
    """Price operator compute by elements inside a representation class.

    ``physical_model.compute_model`` fits ``k_(operator, class) * elements + b``
    on payloads the profiled pipeline really produced, including the
    representations a *reordered* plan can hand an operator (a uint8 payload
    when ``to_float`` moves behind the image transforms).  The DP therefore
    propagates two features beside the byte volume it keeps for boundaries:

      * ``element_prod[mask]`` -- the product of the per-operator element
        ratios, i.e. how many elements reach a position;
      * ``class_state[mask]`` -- the representation class after applying the
        operators in the mask.

    Both are functions of the *set* of preceding operators for the workloads
    this model was fitted on (every transition is monotone: ``to_float`` sets
    the element type, ``grayscale`` reduces channels).  The consistency check
    in ``repr_state_is_order_independent`` fails loudly rather than silently
    merging two different states if a future recipe breaks that property.
    """

    uses_representation_compute_model = True
    # Same two parameters per curve as the byte model, but indexed by the
    # representation class the plan will hand the operator.
    repr_model_variant = "affine"

    def _repr_model(self) -> dict:
        section = (self.profiled_stats or {}).get("physical_model", {})
        model = section.get("compute_model") if isinstance(section, dict) else None
        if not isinstance(model, dict) or model.get("schema_version") != 1:
            raise RuntimeError(
                "The representation-aware compute model requires "
                "physical_model.compute_model schema_version=1 in the shared "
                "profile; regenerate the profile instead of falling back to "
                "the byte model"
            )
        return model

    def _repr_source_record(self) -> Tuple[str, float]:
        model = self._repr_model()
        klass = model.get("source_class")
        elements = model.get("source_elements")
        if klass is None or not elements:
            raise RuntimeError(
                "physical_model.compute_model has no source record features"
            )
        return str(klass), float(elements)

    def _repr_tables(self) -> Tuple[List[float], List[str]]:
        """Element multiplier and representation class per operator mask."""
        cached = getattr(self, "_repr_tables_cache", None)
        if cached is not None:
            return cached
        model = self._repr_model()
        ratios = model.get("element_ratio") or {}
        transitions = model.get("class_transition") or {}
        inner = list(self._dp_inner_ops)
        source_class, _ = self._repr_source_record()
        n = len(inner)
        full = 1 << n
        element_prod = [1.0] * full
        class_state = [source_class] * full
        for mask in range(1, full):
            # Apply the operators of a mask in the feature's declared order
            # (increasing profile index), so the state is built along a valid
            # chain: the reader must precede the image transforms, otherwise a
            # transition table is queried with a class it never saw and the
            # state silently stops updating.  Removing the highest index and
            # applying it last keeps the subset recurrence linear in 2**n.
            index = mask.bit_length() - 1
            prev = mask ^ (1 << index)
            p_id = inner[index]
            ratio = ratios.get(str(p_id))
            try:
                ratio_value = float(ratio) if ratio is not None else 1.0
            except (TypeError, ValueError):
                ratio_value = 1.0
            if not math.isfinite(ratio_value) or ratio_value <= 0.0:
                ratio_value = 1.0
            element_prod[mask] = element_prod[prev] * ratio_value
            table = transitions.get(str(p_id)) or {}
            previous = class_state[prev]
            class_state[mask] = str(table.get(previous, previous))
        self._repr_tables_cache = (element_prod, class_state)
        return element_prod, class_state

    def repr_state_is_order_independent(self) -> bool:
        """Every order of the same operator set must agree on the class state."""
        model = self._repr_model()
        transitions = model.get("class_transition") or {}
        source_class, _ = self._repr_source_record()
        inner = [int(p_id) for p_id in self._dp_inner_ops]
        states = {(): source_class}
        for size in range(1, len(inner) + 1):
            for prefix in [p for p in states if len(p) == size - 1]:
                klass = states[prefix]
                for p_id in inner:
                    if p_id in prefix:
                        continue
                    table = transitions.get(str(p_id)) or {}
                    nxt = str(table.get(klass, klass))
                    key = tuple(sorted(prefix + (p_id,)))
                    existing = states.get(key)
                    if existing is not None and existing != nxt:
                        return False
                    states[key] = nxt
        return True

    def _repr_curves(self, p_id: int) -> dict:
        model = self._repr_model()
        operators = model.get("operators") or {}
        entry = operators.get(str(p_id), operators.get(p_id))
        if not isinstance(entry, dict):
            raise RuntimeError(
                f"compute_model has no curves for operator {p_id}; the profile "
                "must be regenerated (a legal plan is not silently dropped)"
            )
        return entry.get("by_class") or {}

    def _repr_compute_cost(self, p_id: int, elements: float, klass: str) -> float:
        curves = self._repr_curves(p_id)
        curve = curves.get(klass)
        if curve is None:
            # The DP prices masks that violate a dependency while it builds its
            # tables, so a missing class cannot abort the search.  It becomes a
            # prohibitive price *and* is recorded: a final plan that still
            # contains an unmeasured class is rejected by
            # ``assert_plan_covered`` instead of being silently mispriced by
            # the byte model.
            missing = getattr(self, "_repr_missing_classes", None)
            if missing is None:
                missing = set()
                self._repr_missing_classes = missing
            if (int(p_id), str(klass)) not in missing:
                missing.add((int(p_id), str(klass)))
                logger.warning(
                    "representation-aware cost model has no curve for "
                    "operator %s in class %s (measured %s); that candidate "
                    "position is priced prohibitively and reported at the end",
                    p_id,
                    klass,
                    sorted(curves),
                )
            return self._REPR_UNPRICED_PENALTY_MS
        k = float(curve["k_ms_per_element"])
        b = (
            0.0
            if self.repr_model_variant == "proportional"
            else float(curve["b_ms"])
        )
        value = k * max(0.0, float(elements)) + b
        return value if math.isfinite(value) else 0.0

    def _repr_declared_features(self, p_id: int) -> Tuple[float, str]:
        entry = (self._repr_model().get("operators") or {}).get(str(p_id)) or {}
        elements = entry.get("own_elements")
        klass = entry.get("own_class")
        if elements is None or klass is None:
            raise RuntimeError(
                f"compute_model has no profiled input features for operator "
                f"{p_id}; regenerate the profile"
            )
        return float(elements), str(klass)

    def _dp_compute_work_prod(
        self, mask: int, operator_idx: Optional[int] = None
    ) -> float:
        if operator_idx is None:
            return self._dp_work_prod(mask)
        element_prod, class_state = self._repr_tables()
        p_id = self._dp_inner_ops[operator_idx]
        _, source_elements = self._repr_source_record()
        elements = source_elements * element_prod[mask]
        per_record = self._repr_compute_cost(
            p_id, elements, class_state[mask]
        )
        return self._dp_cardinality_prod[mask] * per_record

    def _dp_compute_cost_denominator(
        self,
        operator_idx: int,
        baseline_input_size: float,
        source_size: float,
    ) -> float:
        p_id = self._dp_inner_ops[operator_idx]
        elements, klass = self._repr_declared_features(p_id)
        per_record = self._repr_compute_cost(p_id, elements, klass)
        if per_record <= 0.0:
            raise RuntimeError(
                f"Operator {p_id} has no positive representation-aware cost at "
                "its profiled position"
            )
        return source_size * per_record

    def _calculate_pipe_cost(self, p_id, input_size, desc):
        """Price one operator at its *profiled* representation.

        Position-dependent pricing happens in ``_dp_compute_work_prod``; this
        hook only anchors the per-operator constant the DP normalizes against,
        so the byte ``input_size`` no longer drives compute.
        """
        elements, klass = self._repr_declared_features(p_id)
        local = self._repr_compute_cost(p_id, elements, klass)
        if desc is None or desc.variant_type in (
            None,
            PipeVariantType.INPROCESS,
        ):
            return max(local, 1e-12)
        backend_entry = self._profile_entry(
            self.profiled_stats.get("offloads", {}).get(
                desc.variant_type.name, {}
            ),
            p_id,
        )
        direct = (
            backend_entry.get("backend_compute")
            if isinstance(backend_entry, dict)
            else None
        )
        if isinstance(direct, dict):
            try:
                mean = float(direct["mean_ms_per_sample"])
            except (KeyError, TypeError, ValueError):
                mean = float("nan")
            if math.isfinite(mean) and mean >= 0.0:
                return max(min(mean, local), 1e-12)
        return max(local, 1e-12)


    _REPR_UNPRICED_PENALTY_MS = 1.0e6

    def repr_unpriced_positions(self):
        return sorted(getattr(self, "_repr_missing_classes", set()))

    def assert_plan_covered(self, plan) -> None:
        """Reject a returned plan that contains an unmeasured representation."""
        element_prod, class_state = self._repr_tables()
        index_of = {
            int(p_id): index for index, p_id in enumerate(self._dp_inner_ops)
        }
        missing = []
        mask = 0
        graph = {int(k): v for k, v in plan.graph.items()}
        children = {
            int(child)
            for value in graph.values()
            for child in ([int(x) for x in value.split(",")] if value else [])
        }
        node = next(p for p in graph if p not in children)
        while True:
            if node in index_of:
                index = index_of[node]
                if index not in [
                    i for i in range(len(self._dp_inner_ops)) if mask >> i & 1
                ]:
                    klass = class_state[mask]
                    if klass not in self._repr_curves(node):
                        missing.append((node, klass))
                    mask |= 1 << index
            value = graph.get(node)
            nxt = [int(x) for x in value.split(",")] if value else []
            if not nxt:
                break
            node = nxt[0]
        if missing:
            raise RuntimeError(
                "the selected plan prices operators in representations the "
                f"profile never measured: {missing}; regenerate the profile "
                "with those classes measured instead of falling back to bytes"
            )

class SimpleDpBoundaryAffineElementsOptimizer(
    _RepresentationComputeMixin, SimpleDpBoundaryOptimizer
):
    """M3: affine in elements, one curve per operator, no representation split."""

    repr_model_variant = "affine"

    def _repr_compute_cost(self, p_id: int, elements: float, klass: str) -> float:
        """M3 ignores the position class and always uses the operator's own."""
        _, declared_class = self._repr_declared_features(p_id)
        return super()._repr_compute_cost(p_id, elements, declared_class)


class SimpleDpBoundaryAffineReprProportionalOptimizer(
    _RepresentationComputeMixin, SimpleDpBoundaryOptimizer
):
    """M4: representation-aware, through the origin (``b = 0``)."""

    repr_model_variant = "proportional"


class SimpleDpBoundaryAffineReprOptimizer(
    _RepresentationComputeMixin, SimpleDpBoundaryOptimizer
):
    """M5: representation-aware affine ``k_(i,z) * elements + b_(i,z)``."""

    repr_model_variant = "affine"


class SimpleDpAffineReprOptimizer(SimpleDpBoundaryAffineReprOptimizer):
    """M5 without a stage-boundary charge (the brute-force test's objective).

    With no boundary terms the objective is exactly the sum of the per-operator
    representation-aware costs, which is what the acceptance test enumerates.
    """

    def _dp_regular_transition_cost(self, prev_mask, block):
        return block.cost


@dataclass(frozen=True)
class _SharedCommunicationObjective(DpObjectiveCost):
    """Replica compute plus fixed overhead L, and W-scaled byte service R.

    score/W = (compute + fixed submission overhead)/W + byte service.
    This deliberately excludes PICO's lane exposure and any max objective.
    """

    @property
    def score(self):
        return self.local_serial + self.ray_serial


class SimpleDpWorkersBoundaryOptimizer(SimpleDpBoundaryOptimizer):
    """Choose W jointly with a W-conditioned, boundary-aware Simple-DP plan.

    Each candidate W defines independent per-worker Ray and SMP CPU limits.
    The DP is rerun under that slice, so changing W may change ordering,
    variants, fusion, and cache placement. Every parallel stage remains at
    width one; stage-width allocation is outside this optimizer's search.
    """

    preserve_optimizer_widths = True

    def _dp_candidate_parallelisms(self, variant, execution_resource):
        return (1,)

    def _allocate_final_remote_stage_resources(self):
        # Materialize the fixed width represented in every DP candidate.
        # Cedar's post-pass would otherwise expand Ray stages after the
        # W-conditioned feasibility decision.
        DpOptimizer._allocate_final_remote_stage_resources(self)

    @staticmethod
    def _result_resource_usage(result):
        ray = 0
        smp = 0
        for block in result.blocks:
            variant = result.variants_by_idx.get(
                block[0], PipeVariantType.INPROCESS
            )
            if variant in (PipeVariantType.RAY, PipeVariantType.TF_RAY):
                ray += 1
            elif variant == PipeVariantType.SMP:
                smp += 1
        return DpResourceUsage(ray_cpus=ray, smp_cpus=smp)

    def _worker_resource_groups(self):
        budget = self._dp_worker_budget()
        if budget is None:
            return []
        local_budget, ray_budget, local_reserve, ray_reserve = budget
        # Charge one CPU per local worker. Ray stages
        # are optional, so a zero-sized Ray slice is still a legal candidate.
        # Respect the workload's own replica cap as well. GPU-backed workloads
        # such as LLaVA deliberately set available_local_cpus=1 because every
        # complete Feature replica owns model state on the accelerator.
        max_workers = min(
            self._cap_local_workers(self.options.available_local_cpus),
            local_budget // (1 + local_reserve),
            ray_budget,
        )
        # Costs now depend on W, including measured width curves. Do not
        # collapse workers with identical CPU slices or reuse another W's plan.
        groups = [
            (workers, DpResourceUsage(
                ray_cpus=max(0, ray_budget // workers - ray_reserve),
                smp_cpus=max(0, local_budget // workers - 1 - local_reserve),
            ))
            for workers in range(max_workers, 0, -1)
        ]
        # ``CEDAR_WORKER_SEARCH_SET`` restricts the candidate set.  The W
        # scaling experiment uses it to hold the worker count at one value so
        # every point is the optimizer's own best plan at that W; unset, the
        # full ladder is kept and behaviour is unchanged.
        raw = os.environ.get("CEDAR_WORKER_SEARCH_SET")
        if raw:
            allowed = set()
            for token in str(raw).split(","):
                token = token.strip()
                if token.isdigit():
                    allowed.add(int(token))
            if allowed:
                groups = [group for group in groups if group[0] in allowed]
        return groups

    def _dp_initial_objective_cost(self):
        return _SharedCommunicationObjective(local_serial=float(
            self._base_cost_map[self._get_source_p_id()]))

    def calculate_dp_objective_cost(
        self, plan=None, search_result=None, inner_ops=None, lenient_replay=None
    ):
        # Replaying a materialized plan must use its W, not the W from the
        # most recent candidate search or another plan.
        kwargs = dict(
            plan=plan,
            search_result=search_result,
            inner_ops=inner_ops,
            lenient_replay=lenient_replay,
        )
        if plan is None:
            return super().calculate_dp_objective_cost(**kwargs)
        previous = getattr(self, '_dp_selected_workers', None)
        self._dp_selected_workers = max(1, int(plan.n_local_workers))
        try:
            return super().calculate_dp_objective_cost(**kwargs)
        finally:
            self._dp_selected_workers = previous

    def _unscaled_boundary_transfer_ms(self, prev_mask, block):
        throughput = self._dp_boundary_throughput(block.variant)
        if throughput is None:
            return 0.0
        source_size = self.profiled_stats['baseline']['output_sizes'][
            self._get_source_p_id()]
        next_mask = prev_mask | block.mask
        input_bytes = (
            source_size * self._dp_r_prod[prev_mask]
            * self._dp_cardinality_prod[prev_mask]
        )
        output_bytes = (
            source_size * self._dp_r_prod[next_mask]
            * self._dp_cardinality_prod[next_mask]
        )
        # Sum actual surviving bytes in both directions; fixed request costs
        # remain in the local part of the objective.
        return (input_bytes + output_bytes) / throughput * 1000.0

    def _dp_accumulate_objective_cost(self, previous, extra_cost, block, prev_mask):
        workers = max(1, int(getattr(self, '_dp_selected_workers', None) or 1))
        byte_service = self._unscaled_boundary_transfer_ms(prev_mask, block)
        # Replace the single-pair term; stage width adds no queue pairs.
        aggregate_service = byte_service
        if block.variant == PipeVariantType.SMP and byte_service:
            from cedar.client.boundary_profiler import smp_aggregate_throughput
            model = self._dp_boundary_profile(PipeVariantType.SMP) or {}
            curve = model.get("aggregate_transport")
            if curve is None:
                raise ValueError("W-boundary planning requires a measured SMP aggregate_transport curve")
            context = PipeVariantContextFactory.create_context(variant_type=PipeVariantType.SMP)
            bandwidth = smp_aggregate_throughput(curve, workers, context.max_inflight)
            aggregate_service = (byte_service *
                self._dp_boundary_throughput(PipeVariantType.SMP) / bandwidth)
        return _SharedCommunicationObjective(
            local_serial=previous.local_serial + max(0.0, extra_cost - byte_service),
            ray_serial=previous.ray_serial + workers * aggregate_service,
        )


    def _dp_reorder_offload_cache_fusion(self, inner_ops):
        if not inner_ops:
            return [], None
        groups = self._worker_resource_groups()
        if not groups:
            logger.info(
                "[SimpleDpWorkersBoundaryOptimizer] Resource matching is "
                "disabled; falling back to one boundary-aware DP search."
            )
            return super()._dp_reorder_offload_cache_fusion(inner_ops)

        started = time.monotonic()
        best = None
        evidence = []
        for workers, limit in groups:
            self._dp_selected_workers = workers
            self.physical_plan.set_local_workers(workers)
            try:
                result = self._run_cedar_dp_search(inner_ops, resource_limit=limit)
            except RuntimeError as exc:
                if ("no feasible final state" not in str(exc)
                        and "required parallel CPU total" not in str(exc)):
                    raise
                evidence.append(dict(workers=workers, limits=limit.as_dict(),
                                     status='infeasible_resource_slice'))
                continue
            score = result.cost / workers
            evidence.append({
                "workers": workers,
                "limits": limit.as_dict(),
                "plan_resource_usage": self._result_resource_usage(result).as_dict(),
                "plan_cost": result.cost,
                "compute_and_fixed_ms_per_replica": result.objective.local_serial,
                "unscaled_boundary_ms_per_sample": result.objective.ray_serial / workers,
                "score": score,
                "source": "conditioned_search",
                "status": "evaluated",
            })
            key = (score, result.cost, workers)
            if best is None or key < best[:3]:
                best = (score, result.cost, workers, result, limit)

        if best is None:
            raise RuntimeError("No feasible W-conditioned Simple-DP plan")
        score, _, workers, result, limit = best
        self._worker_search_evidence = evidence
        self._dp_selected_workers = workers
        self.physical_plan.set_local_workers(workers)
        logger.info("[%s] worker search evidence=%s", type(self).__name__, evidence)
        logger.info(
            "[%s] selected W=%s, limits=%s, widths=%s, compute_and_fixed_ms_per_replica=%s, "
            "unscaled_boundary_ms_per_sample=%s, score_ms_per_sample=%s, "
            "searches=%s, elapsed=%.3fs",
            type(self).__name__, workers, limit.as_dict(), result.parallelism_by_idx,
            result.objective.local_serial, result.objective.ray_serial / workers,
            score, len(groups), time.monotonic() - started,
        )
        return self._apply_cedar_dp_search_result(result, inner_ops)


class SimpleDpMaxWorkersBoundaryOptimizer(SimpleDpWorkersBoundaryOptimizer):
    """Use the largest feasible W, then optimize the boundary-aware plan.

    Unlike ``SimpleDpWorkersBoundaryOptimizer``, this policy does not trade
    worker count against modeled plan cost. It fixes the largest W admitted
    by the workload and the two CPU budgets, and only falls back to a smaller
    resource group when the DP has no legal plan at that W.
    """

    def _dp_reorder_offload_cache_fusion(self, inner_ops):
        if not inner_ops:
            return [], None
        groups = self._worker_resource_groups()
        if not groups:
            logger.info(
                "[SimpleDpMaxWorkersBoundaryOptimizer] Resource matching is "
                "disabled; falling back to one boundary-aware DP search."
            )
            return SimpleDpBoundaryOptimizer._dp_reorder_offload_cache_fusion(
                self, inner_ops
            )

        started = time.monotonic()
        evidence = []
        for workers, limit in groups:
            try:
                result = self._run_cedar_dp_search(
                    inner_ops, resource_limit=limit
                )
            except RuntimeError as exc:
                if (
                    "no feasible final state" not in str(exc)
                    and "required parallel CPU total" not in str(exc)
                ):
                    raise
                evidence.append({
                    "workers": workers,
                    "limits": limit.as_dict(),
                    "status": "infeasible_resource_slice",
                })
                continue

            evidence.append({
                "workers": workers,
                "limits": limit.as_dict(),
                "plan_resource_usage": self._result_resource_usage(
                    result
                ).as_dict(),
                "plan_cost": result.cost,
                "status": "selected_largest_feasible_workers",
            })
            self._worker_search_evidence = evidence
            self._dp_selected_workers = workers
            self.physical_plan.set_local_workers(workers)
            logger.info(
                "[SimpleDpMaxWorkersBoundaryOptimizer] selected largest "
                "feasible W=%s, limits=%s, plan_cost=%s, elapsed=%.3fs, "
                "evidence=%s",
                workers,
                limit.as_dict(),
                result.cost,
                time.monotonic() - started,
                evidence,
            )
            return self._apply_cedar_dp_search_result(result, inner_ops)

        raise RuntimeError(
            "No feasible plan for any W in max-worker boundary search"
        )


class SimpleDpWorkersWidthBoundaryOptimizer(SimpleDpWorkersBoundaryOptimizer):
    """Jointly choose W, stage widths and a boundary-aware Simple-DP plan.

    Every W defines an independent per-worker Ray/SMP resource slice.  Unlike
    ``SimpleDpWorkersBoundaryOptimizer``, a parallel stage may consume more
    than one CPU from that slice, and the DP accounts for the sum of the
    selected actor/process widths.  This combines only the three named
    ablations: W-conditioned planning, measured layered-profile width curves,
    and the fixed-plus-byte boundary model.
    """

    joint_actor_allocation = True

    def _dp_parallel_stage_cpu_limit(self):
        # SimpleDpOptimizer deliberately disables resource-conditioned widths.
        # This joint W/width variant must use the selected W's separate pools.
        return DpOptimizer._dp_parallel_stage_cpu_limit(self)

    def _dp_candidate_parallelisms(self, variant, execution_resource):
        # Search the measured width ladder plus the slice limit instead of
        # every integer width. Nothing is measured between the ladder's
        # points, and enumerating all of them made the joint W search
        # quadratic in the CPU budget (W=8 alone took 14 s on a nine-operator
        # pipeline). The slice limit stays a candidate so a plan may still use
        # the whole slice; it is priced by the fitted width curve.
        return DpOptimizer._dp_candidate_parallelisms(
            self, variant, execution_resource
        )

    def _dp_pipe_cost_at_parallelism(
        self, p_id, variant_type, parallelism, width_one_cost
    ):
        # Reuse only the scaling-curve lookup.  The width-one anchor still
        # comes from this class's isolated backend_compute measurement.
        return DpOptimizer._dp_pipe_cost_at_parallelism(
            self, p_id, variant_type, parallelism, width_one_cost
        )

    def _dp_regular_transition_cost(self, prev_mask, block):
        return (
            block.cost / max(1, block.parallelism)
            + self._layered_boundary_cost_ms(prev_mask, block)
        )

    @staticmethod
    def _result_resource_usage(result):
        ray = 0
        smp = 0
        for block in result.blocks:
            variant = result.variants_by_idx.get(
                block[0], PipeVariantType.INPROCESS
            )
            width = max(
                1, int(result.parallelism_by_idx.get(block[0], 1))
            )
            if variant in (PipeVariantType.RAY, PipeVariantType.TF_RAY):
                ray += width
            elif variant == PipeVariantType.SMP:
                smp += width
        return DpResourceUsage(ray_cpus=ray, smp_cpus=smp)

    def _dp_reorder_offload_cache_fusion(self, inner_ops):
        return super()._dp_reorder_offload_cache_fusion(inner_ops)


@dataclass(frozen=True)
class _VariantObjective(DpObjectiveCost):
    @property
    def score(self):
        # Exactly max, independent of process-wide PICO exposure settings.
        return max(self.local_serial, self.ray_serial, self.smp_serial,
                   self.gpu_serial)


class SimpleDpVariantOptimizer(SimpleDpOptimizer):
    cedar_objective = False

    def _dp_initial_objective_cost(self):
        return _VariantObjective(local_serial=float(
            self._base_cost_map[self._get_source_p_id()]))

    def _dp_accumulate_objective_cost(self, previous, extra_cost, block, prev_mask):
        field = {PipeVariantType.RAY: 'ray_serial',
                 PipeVariantType.TF_RAY: 'ray_serial',
                 PipeVariantType.SMP: 'smp_serial'}.get(block.variant, 'local_serial')
        values = {name: getattr(previous, name) for name in
                  ('local_serial', 'ray_serial', 'smp_serial', 'gpu_serial')}
        values[field] += extra_cost
        return _VariantObjective(**values)


class SimpleDpWidthOptimizer(SimpleDpOptimizer):
    """Add resource-feasible widths while retaining scalar Cedar block costs.

    W is frozen from the simple-DP baseline. Cedar has only width-one costs;
    the width-only extension uses service = Cedar block cost / width. PICO's
    fitted fanout exponent and measured-width caps are separate hypotheses
    and are deliberately excluded from this one-factor ablation.
    """
    cedar_objective = False
    joint_actor_allocation = True
    preserve_optimizer_widths = True

    def _physical_opt(self):
        # Freeze the W that the actual simple-DP baseline would execute, so
        # this ablation changes stage widths only. Charge this preliminary
        # planning to the same optimization timer.
        import copy
        from .feature import apply_profile_matched_resources
        baseline = SimpleDpOptimizer()
        baseline.init(self.logical_pipes, self.logical_graph)
        plan = baseline.run(self.profiled_stats, copy.copy(self.options))
        budget = self._dp_worker_budget()
        if budget:
            apply_profile_matched_resources(
                plan, self.profiled_stats, budget[0],
                num_samples=self.options.num_samples)
        self._dp_selected_workers = plan.n_local_workers
        self.physical_plan.set_local_workers(plan.n_local_workers)
        super()._physical_opt()

    def _dp_parallel_stage_cpu_limit(self):
        return DpOptimizer._dp_parallel_stage_cpu_limit(self)

    def _dp_candidate_parallelisms(self, variant, execution_resource):
        # Search the measured width ladder plus the slice limit instead of
        # every integer width. Nothing is measured between the ladder's
        # points, and enumerating all of them made the joint W search
        # quadratic in the CPU budget (W=8 alone took 14 s on a nine-operator
        # pipeline). The slice limit stays a candidate so a plan may still use
        # the whole slice; it is priced by the fitted width curve.
        return DpOptimizer._dp_candidate_parallelisms(
            self, variant, execution_resource
        )

    def _dp_regular_transition_cost(self, prev_mask, block):
        return block.cost / max(1, block.parallelism)

    def _allocate_final_remote_stage_resources(self):
        DpOptimizer._allocate_final_remote_stage_resources(self)


class UnoptimizedOptimizer(Optimizer):
    """Declared graph, one driver, no optimizer passes (Cedar runtime)."""
    preserve_optimizer_widths = True

    def _logical_opt(self):
        pass

    def _physical_opt(self):
        self.physical_plan.set_local_workers(1)
        for desc in self.physical_plan.pipe_descs.values():
            desc.variant_type = PipeVariantType.INPROCESS
            desc.variant_ctx = PipeVariantContextFactory.create_context(
                variant_type=PipeVariantType.INPROCESS)

class SimpleDpWorkersWidthBoundaryAffineReprOptimizer(
    _RepresentationComputeMixin, SimpleDpWorkersWidthBoundaryOptimizer
):
    """PICO with the representation-aware compute model (M5)."""

    repr_model_variant = "affine"


class _ProportionalByteComputeMixin:
    """Price operator compute by byte volume with Cedar's proportional rule.

    Same search, boundary model, backend support and W ladder as the final
    W-only PICO; only the compute term is the historical
    ``cost x bytes / profiled bytes`` rule anchored on the profile's per
    operator baseline cost.  This is the byte-proportional arm of the model
    ablation (``C1``), not a re-implementation of Cedar's staged search.
    """

    uses_affine_operator_cost = False

    def _prop_byte_work_prod(
        self, mask: int, operator_idx: Optional[int] = None
    ) -> float:
        if operator_idx is None:
            return self._dp_work_prod(mask)
        p_id = self._dp_inner_ops[operator_idx]
        source_size = float(
            self.profiled_stats["baseline"]["output_sizes"][
                self._get_source_p_id()
            ]
        )
        item_bytes = source_size * self._dp_r_prod[mask]
        reference = float(self.profiled_stats["baseline"]["input_sizes"][p_id])
        base = float(self._base_cost_map[p_id])
        if reference <= 0.0:
            return self._dp_cardinality_prod[mask] * base
        return self._dp_cardinality_prod[mask] * base * (item_bytes / reference)

    def _dp_compute_work_prod(self, mask: int, operator_idx: Optional[int] = None):
        return self._prop_byte_work_prod(mask, operator_idx)

    def _dp_compute_cost_denominator(
        self, operator_idx: int, baseline_input_size: float, source_size: float
    ) -> float:
        p_id = self._dp_inner_ops[operator_idx]
        return source_size * float(self._base_cost_map[p_id])


class SimpleDpWorkersByteProportionalOptimizer(
    _ProportionalByteComputeMixin, SimpleDpWorkersBoundaryOptimizer
):
    """C1 arm: byte-proportional compute, identical W-only search."""


class SimpleDpWorkersBoundaryAffineReprOptimizer(
    _RepresentationComputeMixin, SimpleDpWorkersBoundaryOptimizer
):
    """Final PICO: W-only search, representation-aware affine compute, boundary
    and fusion/backend/cache search unchanged (per-stage width fixed at 1)."""

    repr_model_variant = "affine"


class SimpleDpWorkersAffineReprOptimizer(SimpleDpWorkersBoundaryAffineReprOptimizer):
    """C1 arm: same W-only search and model, explicit boundary term removed."""

    def _dp_regular_transition_cost(self, prev_mask, block):
        return block.cost
