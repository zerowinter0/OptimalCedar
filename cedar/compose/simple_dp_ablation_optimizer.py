"""One-factor additions to the Cedar-cost subset DP.

No class enables PICO's affine, selectivity, concurrency, lane-exposure, or
cross-host fitted models. The legacy W-only ablation retains a W-independent
plan; the combined W+boundary optimizer explicitly conditions DP on W.
"""
from dataclasses import dataclass
import logging
import time

from cedar.pipes import PipeVariantType, PipeVariantContextFactory
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


class SimpleDpBoundaryOptimizer(SimpleDpOptimizer):
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
        # One local runtime worker and its reserve must both fit. Ray stages
        # are optional, so a zero-sized Ray slice is still a legal candidate.
        max_workers = min(
            self._cap_local_workers(local_budget),
            local_budget // (1 + local_reserve),
            ray_budget,
        )
        grouped = {}
        for workers in range(1, max_workers + 1):
            limit = DpResourceUsage(
                ray_cpus=max(0, ray_budget // workers - ray_reserve),
                smp_cpus=max(
                    0, local_budget // workers - 1 - local_reserve
                ),
            )
            # For identical constraints, the largest W always has the lower
            # cost/W score and is the only member that can win.
            grouped[(limit.ray_cpus, limit.smp_cpus)] = (workers, limit)
        return sorted(grouped.values(), key=lambda item: item[0], reverse=True)

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
        unconstrained = self._run_cedar_dp_search(inner_ops)
        unconstrained_usage = self._result_resource_usage(unconstrained)
        best = None
        evidence = []
        searched_limits = {}

        for workers, limit in groups:
            lower_bound = unconstrained.cost / workers
            if best is not None and lower_bound >= best[0]:
                evidence.append({
                    "workers": workers,
                    "limits": limit.as_dict(),
                    "lower_bound": lower_bound,
                    "status": "pruned_by_unconstrained_lower_bound",
                })
                continue

            if unconstrained_usage <= limit:
                result = unconstrained
                source = "unconstrained_reuse"
            else:
                key = (limit.ray_cpus, limit.smp_cpus)
                if key not in searched_limits:
                    try:
                        searched_limits[key] = self._run_cedar_dp_search(
                            inner_ops, resource_limit=limit
                        )
                    except RuntimeError as exc:
                        if (
                            "no feasible final state" not in str(exc)
                            and "required parallel CPU total" not in str(exc)
                        ):
                            raise
                        searched_limits[key] = None
                if searched_limits[key] is None:
                    evidence.append({
                        "workers": workers,
                        "limits": limit.as_dict(),
                        "lower_bound": lower_bound,
                        "status": "infeasible_resource_slice",
                    })
                    continue
                result = searched_limits[key]
                source = "conditioned_search"

            score = result.cost / workers
            evidence.append({
                "workers": workers,
                "limits": limit.as_dict(),
                "plan_resource_usage": self._result_resource_usage(
                    result
                ).as_dict(),
                "plan_cost": result.cost,
                "score": score,
                "source": source,
                "status": "evaluated",
            })
            key = (score, result.cost, workers)
            if best is None or key < best[:3]:
                best = (score, result.cost, workers, result, limit)

        if best is None:
            raise RuntimeError("No feasible W-conditioned Simple-DP plan")

        _, _, workers, result, limit = best
        self._worker_search_evidence = evidence
        self._dp_selected_workers = workers
        self.physical_plan.set_local_workers(workers)
        logger.info(
            "[SimpleDpWorkersBoundaryOptimizer] worker search evidence=%s",
            evidence,
        )
        logger.info(
            "[SimpleDpWorkersBoundaryOptimizer] selected W=%s, limits=%s, "
            "plan_cost=%s, cost/W=%s, searches=%s, elapsed=%.3fs",
            workers,
            limit.as_dict(),
            result.cost,
            result.cost / workers,
            len(searched_limits) + 1,
            time.monotonic() - started,
        )
        return self._apply_cedar_dp_search_result(result, inner_ops)


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
        return tuple(range(1, self._dp_max_candidate_parallelism(
            variant, execution_resource) + 1))

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
