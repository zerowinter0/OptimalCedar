"""Cedar's staged search re-priced by PICO's layered cost models.

Search structure is Cedar's own ``Optimizer``: the same logical pass
(reordering, cache placement), the same physical passes (worker heuristic,
offload/fusion candidate enumeration, TF fusion), the same post-hoc stage
widths, and the same acceptance rule ("take it when it lowers the cost").
Only the plan evaluator is replaced: every ``calculate_cost`` call is answered
by the matching Simple-DP ablation class, so a staged plan and a DP plan are
priced by bit-identical code and the comparison isolates the search.

Three tiers mirror the DP ablation ladder:

  ``StagedBoundaryOptimizer``                 boundary + Cedar compute
  ``StagedBoundaryAffineOptimizer``           + per-operator affine kx+b
  ``StagedWorkersBoundaryAffineOptimizer``    + the W (replica) decision

The staged search has no joint W dimension: the first two tiers keep Cedar's
own worker heuristic, and the third prices the finished plan at every legal W
and keeps the cheapest ``score / W`` (the W-conditioned model's own unit).
Stage widths stay at the width the staged method materializes; during the
search every not-yet-materialized stage is priced at width one, which is the
width the matching DP ablations search over.
"""
from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from cedar.pipes import (
    PipeExecutionResource,
    PipeVariantType,
    PipeVariantContextFactory,
)

from .optimizer import (
    FUSED_PIPE_NAME,
    Optimizer,
    OptimizerOptions,
    PhysicalPlan,
    PipeDesc,
)
from .utils import flip_adj_list
from .simple_dp_ablation_optimizer import (
    OldDpBoundaryOptimizer,
    SimpleDpBoundaryOptimizer,
    SimpleDpWorkersBoundaryAffineReprOptimizer,
    SimpleDpWorkersBoundaryOptimizer,
)

logger = logging.getLogger(__name__)


class _StagedLayeredCostMixin:
    """Answer every staged cost query with one PICO cost model."""

    dp_cost_optimizer_class: Optional[type] = None
    # The objective is the tier's PICO model, not Cedar's Amdahl/byte model:
    # the harness must report this optimizer's plan cost from the same code.
    cedar_objective = False

    def _init_stats(self):
        Optimizer._init_stats(self)
        self._layered_cost_optimizer = None
        self._layered_cost_inner_ops: Optional[Tuple[int, ...]] = None
        self._layered_cost_evidence: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # pricing
    # ------------------------------------------------------------------
    def _layered_cost_pricer(self):
        if self.dp_cost_optimizer_class is None:
            raise RuntimeError(
                f"{type(self).__name__} must declare dp_cost_optimizer_class"
            )
        if self._layered_cost_optimizer is None:
            pricer = self.dp_cost_optimizer_class()
            pricer.init(self.logical_pipes, self.logical_graph)
            pricer.profiled_stats = self.profiled_stats
            pricer.options = self.options
            pricer._validate_stats()
            pricer._init_stats()
            self._layered_cost_optimizer = pricer
        return self._layered_cost_optimizer

    @staticmethod
    def _linear_path(plan: PhysicalPlan) -> List[int]:
        predecessors = {
            child for children in plan.graph.values() for child in children
        }
        sources = [p_id for p_id in plan.graph if p_id not in predecessors]
        if len(sources) != 1:
            raise RuntimeError(
                f"Staged layering requires one linear source, got {sources}"
            )
        path: List[int] = []
        current = sources[0]
        while True:
            path.append(current)
            successors = plan.graph.get(current, set())
            if not successors:
                break
            if len(successors) != 1:
                raise RuntimeError(
                    "Staged layering requires a linear plan, "
                    f"pipe {current} fans out to {successors}"
                )
            current = next(iter(successors))
        if len(path) != len(plan.graph):
            raise RuntimeError("Staged layering found nodes off the main path.")
        return path

    def _staged_inner_ops(self, plan: PhysicalPlan) -> List[int]:
        """Operator ids of a candidate plan, in the plan's own order.

        Cache/prefetch nodes and materialized fused blocks are not operators;
        the DP prices the latter from their member list.
        """
        operators = set(self._layered_cost_pricer()._base_cost_map)
        return [
            p_id for p_id in self._linear_path(plan)[1:] if p_id in operators
        ]

    def _staged_price(
        self, plan: PhysicalPlan, workers: Optional[int] = None
    ) -> float:
        """Price one candidate with the tier's own cost model.

        ``workers=None`` uses the worker count the staged method chose for the
        candidate, which is how the W-conditioned tier sees the boundary
        term's aggregate transport at the W it is actually planning for.
        """
        pricer = self._layered_cost_pricer()
        inner_ops = self._staged_inner_ops(plan)
        if tuple(inner_ops) != self._layered_cost_inner_ops:
            # The DP quantifies over its inner-op order, so a reordered plan
            # has to be re-indexed before it can be priced.
            pricer._prepare_dp_metadata(list(inner_ops))
            self._layered_cost_inner_ops = tuple(inner_ops)
        if workers is None:
            workers = int(plan.n_local_workers or 1)
        pricing_plan = plan
        if int(plan.n_local_workers or 1) != int(workers):
            pricing_plan = PhysicalPlan(
                plan.graph, plan.pipe_descs, int(workers)
            )
        pricer.physical_plan = pricing_plan
        return pricer.calculate_dp_objective_cost(plan=pricing_plan)

    # ------------------------------------------------------------------
    # harness adapters
    # ------------------------------------------------------------------
    def _get_linear_inner_ops(self) -> List[int]:
        return self._staged_inner_ops(self.physical_plan)

    def _prepare_dp_metadata(self, inner_ops: List[int]) -> None:
        pricer = self._layered_cost_pricer()
        pricer._prepare_dp_metadata(list(inner_ops))
        self._layered_cost_inner_ops = tuple(inner_ops)

    def calculate_dp_objective_cost(
        self,
        plan: Optional[PhysicalPlan] = None,
        search_result: Any = None,
        inner_ops: Optional[List[int]] = None,
        lenient_replay: Optional[bool] = None,
    ) -> float:
        """Price a materialized plan with this tier's model (harness entry)."""
        if plan is None:
            raise ValueError(
                "The staged ablation optimizers only price materialized plans."
            )
        return self._staged_price(plan)

    # ------------------------------------------------------------------
    # candidate materialization
    # ------------------------------------------------------------------
    @staticmethod
    def _normalized_groups(
        fused_pipes: Optional[Union[Sequence[int], Sequence[Sequence[int]]]]
    ) -> List[List[int]]:
        if not fused_pipes:
            return []
        raw = list(fused_pipes)
        if raw and isinstance(raw[0], (list, tuple, set)):
            return [list(group) for group in raw]
        return [list(raw)]

    def _candidate_desc(
        self, base: PhysicalPlan, p_id: int, desc: PipeDesc
    ) -> PipeDesc:
        merged = copy.copy(desc)
        template = base.pipe_descs.get(p_id)
        if merged.name is None and template is not None:
            merged.name = template.name
        if merged.execution_resource is None and template is not None:
            merged.execution_resource = template.execution_resource
        if merged.variant_type is None:
            merged.variant_type = PipeVariantType.INPROCESS
        if merged.variant_ctx is None:
            merged.variant_ctx = PipeVariantContextFactory.create_context(
                variant_type=merged.variant_type
            )
        return merged

    def _fuse_block(
        self, plan: PhysicalPlan, group: List[int], desc: PipeDesc
    ) -> int:
        order = self._linear_path(plan)
        position = {p_id: index for index, p_id in enumerate(order)}
        missing = [p_id for p_id in group if p_id not in position]
        if missing:
            raise RuntimeError(
                f"Cannot fuse pipes {missing}: not on the candidate path"
            )
        members = sorted(group, key=lambda p_id: position[p_id])
        new_p_id = max(plan.pipe_descs) + 1
        execution_resource = (
            PipeExecutionResource.CUDA
            if any(
                plan.pipe_descs[p_id].execution_resource
                == PipeExecutionResource.CUDA
                for p_id in members
            )
            else PipeExecutionResource.CPU
        )
        plan.pipe_descs[new_p_id] = PipeDesc(
            name=FUSED_PIPE_NAME,
            variant_type=desc.variant_type,
            variant_ctx=desc.variant_ctx,
            fused_pipes=list(members),
            execution_resource=execution_resource,
        )
        input_graph = flip_adj_list(plan.graph)
        predecessors = input_graph[members[0]]
        if len(predecessors) != 1:
            raise RuntimeError(
                f"Cannot fuse {members}: the head has {len(predecessors)} inputs"
            )
        predecessor = next(iter(predecessors))
        if len(plan.graph[predecessor]) != 1:
            raise RuntimeError(
                f"Cannot fuse {members}: the head's input fans out"
            )
        successors = plan.graph[members[-1]]
        if len(successors) > 1:
            raise RuntimeError(
                f"Cannot fuse {members}: the tail fans out to {successors}"
            )
        plan.graph[predecessor] = {new_p_id}
        plan.graph[new_p_id] = set(successors)
        for p_id in members:
            del plan.graph[p_id]
        return new_p_id

    def _candidate_plan(
        self,
        graph: Optional[Dict[int, Set[int]]],
        physical_specs: Optional[Dict[int, PipeDesc]],
        fused_pipes: Optional[Union[Sequence[int], Sequence[Sequence[int]]]],
        plan: Optional[PhysicalPlan],
    ) -> PhysicalPlan:
        base = plan if plan is not None else self.physical_plan
        candidate = PhysicalPlan(
            graph=copy.deepcopy(graph if graph is not None else base.graph),
            pipe_descs={
                p_id: self._candidate_desc(base, p_id, copy.copy(desc))
                for p_id, desc in base.pipe_descs.items()
            },
            n_local_workers=int(base.n_local_workers or 1),
        )
        specs = physical_specs or {}
        for p_id, desc in specs.items():
            candidate.pipe_descs[p_id] = self._candidate_desc(base, p_id, desc)
        for group in self._normalized_groups(fused_pipes):
            if len(group) < 2:
                continue
            if not all(p_id in candidate.graph for p_id in group):
                # Already materialized by an earlier pass (the plan's own
                # fused node replaced these members).
                continue
            head_desc = specs.get(group[0])
            if head_desc is None:
                head_desc = candidate.pipe_descs[group[0]]
            self._fuse_block(candidate, group, head_desc)
        return candidate

    def calculate_cost(
        self,
        graph: Optional[Dict[int, Set[int]]] = None,
        physical_specs: Optional[Dict[int, PipeDesc]] = None,
        fused_pipes: Optional[Union[Sequence[int], Sequence[Sequence[int]]]] = None,
        caching_on: Optional[bool] = False,
        plan: Optional[PhysicalPlan] = None,
    ) -> float:
        candidate = self._candidate_plan(
            graph, physical_specs, fused_pipes, plan
        )
        try:
            return self._staged_price(candidate)
        except (ValueError, RuntimeError) as exc:
            # The tier's model refused the candidate: for the W-conditioned
            # tier this is the per-worker CPU slice rejecting a stage that
            # does not fit, i.e. the same verdict the DP search would give.
            logger.warning(
                "[%s] candidate priced as infeasible: %s: %s",
                type(self).__name__,
                type(exc).__name__,
                exc,
            )
            return float("inf")


class StagedBoundaryOptimizer(_StagedLayeredCostMixin, Optimizer):
    """Cedar's staged search priced with boundary + Cedar compute."""

    dp_cost_optimizer_class = OldDpBoundaryOptimizer


class StagedBoundaryAffineOptimizer(
    StagedBoundaryOptimizer
):
    """Cedar's staged search priced with boundary + per-operator kx+b."""

    dp_cost_optimizer_class = SimpleDpBoundaryOptimizer


class StagedWorkersBoundaryAffineOptimizer(
    StagedBoundaryAffineOptimizer
):
    """The affine/boundary model plus the worker (W) decision.

    The staged passes produce one plan shape; this variant then prices that
    plan at every W the resource envelope admits and keeps the cheapest
    ``score / W``, exactly the unit the DP's worker search reports.
    """

    dp_cost_optimizer_class = SimpleDpWorkersBoundaryOptimizer

    def run(
        self, profiled_data: Union[str, Dict[str, Any]], options: OptimizerOptions
    ) -> PhysicalPlan:
        plan = Optimizer.run(self, profiled_data, options)
        if options.enable_local_parallelism:
            self._select_workers_by_cost(plan)
        return plan

    def _select_workers_by_cost(self, plan: PhysicalPlan) -> int:
        pricer = self._layered_cost_pricer()
        groups = pricer._worker_resource_groups()
        if not groups:
            raise RuntimeError(
                "Worker-cost selection requires the resource-matched budget "
                "(CEDAR_MATCH_PROFILE_RESOURCES=1 with a CPU budget)"
            )
        evidence: List[Dict[str, Any]] = []
        best: Optional[Tuple[float, int]] = None
        for workers, limits in groups:
            try:
                score = self._staged_price(plan, workers=workers)
            except (ValueError, RuntimeError) as exc:
                evidence.append(
                    {
                        "workers": workers,
                        "limits": limits.as_dict(),
                        "status": f"infeasible: {type(exc).__name__}: {exc}",
                    }
                )
                continue
            cost = score / workers
            evidence.append(
                {
                    "workers": workers,
                    "limits": limits.as_dict(),
                    "score": score,
                    "cost_ms_per_source_record": cost,
                    "status": "evaluated",
                }
            )
            key = (cost, workers)
            if best is None or key < best:
                best = key
        if best is None:
            raise RuntimeError("No feasible W for the staged layered plan")
        cost, workers = best
        plan.set_local_workers(workers)
        self._layered_cost_evidence = evidence
        logger.info(
            "[%s] worker search evidence=%s", type(self).__name__, evidence
        )
        logger.info(
            "[%s] selected W=%s, cost=%s ms/source-record",
            type(self).__name__,
            workers,
            cost,
        )
        return workers


class StagedWorkersBoundaryAffineReprOptimizer(
    StagedWorkersBoundaryAffineOptimizer
):
    """C2: staged search priced by the *final* W-only PICO model.

    Only the search organisation differs from ``pico_final`` (staged keeps its
    intermediate decisions instead of searching them jointly); the compute
    model, boundary model, backend/fusion support, W ladder and width rule are
    the same oracle class the final optimizer uses.
    """

    dp_cost_optimizer_class = SimpleDpWorkersBoundaryAffineReprOptimizer
