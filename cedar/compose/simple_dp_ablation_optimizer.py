"""One-factor additions to the Cedar-cost subset DP.

No class enables PICO's affine, selectivity, concurrency, lane-exposure, or
cross-host fitted models. W search retains the same W-independent Cedar DP
solution and evaluates its feasible replicas analytically (no redundant DP).
"""
from dataclasses import dataclass

from cedar.pipes import PipeVariantType, PipeVariantContextFactory
from .optimizer import Optimizer
from .dp_optimizer import DpOptimizer, DpObjectiveCost
from .simple_dp_optimizer import SimpleDpOptimizer


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
