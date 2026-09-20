"""Simple-DP variants and retained historical implementation helpers.

The current boundary, workers-boundary and workers-width-boundary variants
use isolated layered-profile compute data. OldDpBoundaryOptimizer preserves
the otherwise identical Cedar whole-pipeline/Amdahl cost path as a control.
"""
from dataclasses import dataclass
import logging
import math
import time

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
    def _valid_backend_compute(entry):
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
        if (
            count <= 0
            or not math.isfinite(mean)
            or mean < 0.0
            or (
                isinstance(adaptive, dict)
                and adaptive
                and adaptive.get("converged") is not True
            )
        ):
            return None
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
        mean = self._valid_backend_compute(entry)
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
        return [
            (workers, DpResourceUsage(
                ray_cpus=max(0, ray_budget // workers - ray_reserve),
                smp_cpus=max(0, local_budget // workers - 1 - local_reserve),
            ))
            for workers in range(max_workers, 0, -1)
        ]

    def _dp_initial_objective_cost(self):
        return _SharedCommunicationObjective(local_serial=float(
            self._base_cost_map[self._get_source_p_id()]))

    def calculate_dp_objective_cost(self, plan=None, search_result=None, inner_ops=None):
        # Replaying a materialized plan must use its W, not the W from the
        # most recent candidate search or another plan.
        if plan is None:
            return super().calculate_dp_objective_cost(
                plan=plan, search_result=search_result, inner_ops=inner_ops)
        previous = getattr(self, '_dp_selected_workers', None)
        self._dp_selected_workers = max(1, int(plan.n_local_workers))
        try:
            return super().calculate_dp_objective_cost(
                plan=plan, search_result=search_result, inner_ops=inner_ops)
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
