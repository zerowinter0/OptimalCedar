"""Cost models of existing ML data pipeline systems, applied to a Cedar plan.

Every model answers exactly one question: *given a materialized Cedar physical
plan and the profiler statistics that system would have collected, what
per-record service time does its cost model predict?*  These are cost models
only - no plan search and no online control - so a plan produced by any
optimizer can be scored by every system's model.  That is how we show where
each model's blind spots are: for the same three plans, which models rank them
in the measured order and which do not.

References and modelled scope
-----------------------------
cedar      Cedar (VLDB'24): per-pipe trace latency scaled by input size along
           the critical path; an offloaded pipe is discounted by the measured
           offload throughput (Amdahl).  We call the real implementation so the
           number is exactly what Cedar's planner optimises.
plumber    Plumber (MLSys'22): per-stage per-core rate from measured latencies,
           stages capped at one core unless widened, pipeline paced by the
           slowest stage.  Single host: no fusion, no reorder, no transport.
pecan      Pecan (ATC'24): per-transformation measured cost with volume
           reduction ordering and a hybrid local/remote worker placement.  Its
           controller minimises the measured batch processing time ``bpt`` and
           then the batch processing cost ``bpc = bpt * (c_a * n_a + c_w * n_w)``;
           applied to a fixed plan the bpt estimate is the slowest stage served
           from the worker pool of that plan.
tfdata     tf.data (SIGMOD'19): the graph is pipelined and paced by the slowest
           transformation after its parallelism is applied; prefetching hides
           latency, so no stage-boundary or transport term is charged.
raydata    Ray Data: streaming blocks.  Stage parallelism is the CPU budget
           divided evenly across stages, blocks add a size-dependent overhead,
           and the pipeline is paced by its slowest stage.
fastflow   FastFlow: bottleneck offload.  Stage services are taken at one core
           (plus the width the plan gives them) and the pipeline is paced by the
           slowest stage, so offloading the bottleneck is the only lever.
aero       Aero (Adaptive Query Processing of ML Queries): service rate of one
           worker executing a predicate of cost ``c`` is ``1/c``; the pipeline
           throughput is the minimum rate over workers, i.e. ``min_i(w_i/c_i)``.

All models report milliseconds per record per worker, the unit the DP
objective and the measured numbers use.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .optimizer import PhysicalPlan, PipeVariantType


@dataclass(frozen=True)
class Stage:
    """One materialized stage of a plan."""

    members: Tuple[int, ...]
    variant: str
    width: int


@dataclass(frozen=True)
class SystemCostEstimate:
    """Result of one system's cost model on one plan."""

    system: str
    ms_per_record: float
    note: str = ""


def _entry(mapping: Any, key: int, default: Any = None) -> Any:
    if not isinstance(mapping, dict):
        return default
    if key in mapping:
        return mapping[key]
    return mapping.get(str(key), default)


def plan_stages(plan: PhysicalPlan) -> List[Stage]:
    """Return the plan's stages in execution order (source to sink).

    A fused pipe keeps its members in ``fused_pipes``, which is what lets each
    model charge the fusion the plan actually materialized.
    """
    graph = {int(k): set(v) for k, v in plan.graph.items()}
    children = {child for parents in graph.values() for child in parents}
    node = next((pid for pid in graph if pid not in children), None)
    order: List[int] = []
    while node is not None:
        order.append(node)
        nexts = sorted(graph.get(node, set()))
        node = nexts[0] if nexts else None
    stages: List[Stage] = []
    for p_id in order:
        desc = plan.pipe_descs[p_id]
        name = desc.name or ""
        if "Prefetcher" in name:
            continue
        members = tuple(desc.fused_pipes) if desc.fused_pipes else (p_id,)
        variant = (
            desc.variant_type.name if desc.variant_type is not None else "INPROCESS"
        )
        width = 1
        ctx = desc.variant_ctx
        if variant in ("RAY", "TF_RAY"):
            width = int(getattr(ctx, "n_actors", 1) or 1)
        elif variant == "SMP":
            width = int(getattr(ctx, "n_procs", 1) or 1)
        stages.append(Stage(members=members, variant=variant, width=max(1, width)))
    return stages


def _multiset_latency_ms(
    profile: Dict[str, Any],
    members: Sequence[int],
    *,
    source: str = "baseline",
    affine_costs: Optional[Dict[int, float]] = None,
) -> float:
    """Sum of the measured per-record costs of the stage's members."""
    if source == "affine":
        if affine_costs:
            total = 0.0
            for member in members:
                try:
                    total += float(affine_costs[int(member)])
                except (KeyError, TypeError, ValueError):
                    continue
            if total > 0.0:
                return total
        operators = profile.get("cm_model", {}).get("operators", {})
        total = 0.0
        for member in members:
            entry = _entry(operators, int(member), {}) or {}
            try:
                total += float(entry.get("mean_ms", 0.0))
            except (TypeError, ValueError):
                continue
        if total > 0.0:
            return total
    latencies = profile.get("baseline", {}).get("latencies", {})
    total = 0.0
    for member in members:
        try:
            total += float(_entry(latencies, int(member), 0.0)) / 1e6
        except (TypeError, ValueError):
            continue
    return total


class SystemCostModel:
    """Common interface: score one plan, no plan search."""

    name: str = "system"
    reference: str = ""
    cost_source: str = "baseline"

    def calculate_cost(
        self,
        plan: PhysicalPlan,
        profile: Dict[str, Any],
        workers: int = 1,
        affine_costs: Optional[Dict[int, float]] = None,
    ) -> SystemCostEstimate:
        raise NotImplementedError


class PlumberCostModel(SystemCostModel):
    """Plumber (MLSys'22): per-stage rates on one host, paced by the slowest."""

    name = "plumber"
    reference = "Plumber (MLSys'22)"
    cost_source = "baseline"

    def calculate_cost(
        self, plan, profile, workers=1, affine_costs=None
    ):
        stages = plan_stages(plan)
        slowest = 0.0
        for stage in stages:
            service = _multiset_latency_ms(
                profile,
                stage.members,
                source=self.cost_source,
                affine_costs=affine_costs,
            )
            slowest = max(slowest, service / stage.width)
        return SystemCostEstimate(
            self.name,
            slowest,
            "paced by the slowest stage; assumes every stage owns its own core "
            "(un-parallelized stages are not summed); no transport/width model",
        )


class PecanCostModel(SystemCostModel):
    """Pecan (ATC'24): bpt estimate over its local/remote worker placement."""

    name = "pecan"
    reference = "Pecan (ATC'24)"
    cost_source = "baseline"

    def calculate_cost(
        self, plan, profile, workers=1, affine_costs=None
    ):
        stages = plan_stages(plan)
        slowest = 0.0
        remote = 0
        for stage in stages:
            service = _multiset_latency_ms(
                profile,
                stage.members,
                source=self.cost_source,
                affine_costs=affine_costs,
            )
            # Pecan parallelises over both local and remote data workers.
            width = (
                stage.width
                if stage.variant in ("RAY", "TF_RAY", "SMP")
                else 1
            )
            if stage.variant in ("RAY", "TF_RAY"):
                remote += 1
            slowest = max(slowest, service / width)
        return SystemCostEstimate(
            self.name,
            slowest,
            "batch processing time of the plan's local/remote split "
            f"({remote} remote stage(s)); placement is chosen by its controller; "
            "money term bpc = bpt*(c_a*n_a + c_w*n_w) not priced here",
        )


class TfDataCostModel(SystemCostModel):
    """tf.data (SIGMOD'19): pipelined graph, parallelism per transformation."""

    name = "tfdata"
    reference = "tf.data (SIGMOD'19)"
    cost_source = "baseline"

    def calculate_cost(
        self, plan, profile, workers=1, affine_costs=None
    ):
        stages = plan_stages(plan)
        slowest = 0.0
        for stage in stages:
            service = _multiset_latency_ms(
                profile,
                stage.members,
                source=self.cost_source,
                affine_costs=affine_costs,
            )
            slowest = max(slowest, service / stage.width)
        return SystemCostEstimate(
            self.name,
            slowest,
            "prefetch hides stage latency: no boundary, transport or host term; "
            "each transformation is assumed to have its own core",
        )


class RayDataCostModel(SystemCostModel):
    """Ray Data: even CPU split across stages plus block overhead."""

    name = "raydata"
    reference = "Ray Data (streaming blocks)"
    cost_source = "baseline"

    def calculate_cost(
        self, plan, profile, workers=1, affine_costs=None
    ):
        stages = [s for s in plan_stages(plan) if s.members]
        if not stages:
            return SystemCostEstimate(self.name, 0.0, "empty plan")
        cpu_budget = int(
            profile.get("resource_config", {}).get("cpu_budget", 0) or 0
        )
        if cpu_budget <= 0:
            cpu_budget = max(1, workers) * 8
        share = max(1, cpu_budget // max(1, workers) // len(stages))
        slowest = 0.0
        for stage in stages:
            service = _multiset_latency_ms(
                profile,
                stage.members,
                source=self.cost_source,
                affine_costs=affine_costs,
            )
            slowest = max(slowest, service / share)
        return SystemCostEstimate(
            self.name,
            slowest,
            f"stage parallelism={share} from an even CPU split; "
            "block overhead and backpressure not modelled",
        )


class FastFlowCostModel(SystemCostModel):
    """FastFlow: offload the bottleneck stage, single host otherwise."""

    name = "fastflow"
    reference = "FastFlow (bottleneck offload)"
    cost_source = "baseline"

    def calculate_cost(
        self, plan, profile, workers=1, affine_costs=None
    ):
        stages = plan_stages(plan)
        services = [
            _multiset_latency_ms(
                profile,
                stage.members,
                source=self.cost_source,
                affine_costs=affine_costs,
            )
            / stage.width
            for stage in stages
        ]
        slowest = max(services) if services else 0.0
        return SystemCostEstimate(
            self.name,
            slowest,
            "bottleneck stage paced after offload; other stages stay at one core",
        )


class AeroCostModel(SystemCostModel):
    """Aero (adaptive ML query processing): queueing bottleneck over workers."""

    name = "aero"
    reference = "Aero (adaptive ML query processing)"
    cost_source = "baseline"

    def calculate_cost(
        self, plan, profile, workers=1, affine_costs=None
    ):
        stages = plan_stages(plan)
        rate = None
        for stage in stages:
            service = _multiset_latency_ms(
                profile,
                stage.members,
                source=self.cost_source,
                affine_costs=affine_costs,
            )
            if service <= 0.0:
                continue
            stage_rate = stage.width / service
            rate = stage_rate if rate is None else min(rate, stage_rate)
        if not rate:
            return SystemCostEstimate(self.name, 0.0, "no measurable stage")
        return SystemCostEstimate(
            self.name,
            1.0 / rate,
            "throughput = min_i(w_i / c_i): service rate per worker is 1/cost",
        )


class CedarCostModel(SystemCostModel):
    """Cedar (VLDB'24): its own size-scaled critical-path cost."""

    name = "cedar"
    reference = "Cedar (VLDB'24)"
    cost_source = "baseline"

    def __init__(self, optimizer: Optional[Any] = None):
        self._optimizer = optimizer

    def calculate_cost(
        self, plan, profile, workers=1, affine_costs=None
    ):
        if self._optimizer is None:
            raise RuntimeError(
                "Cedar's cost function needs a prepared Optimizer instance; "
                "pass one to CedarCostModel(optimizer)."
            )
        value = self._optimizer.calculate_cost(plan.graph, plan=plan)
        return SystemCostEstimate(
            self.name,
            float(value),
            "Cedar's own units (size-scaled pipe latency along the critical "
            "path, offloaded pipes discounted by measured offload throughput)",
        )


def default_models(
    cedar_optimizer: Optional[Any] = None,
) -> Dict[str, SystemCostModel]:
    """Registry of every model we can score a plan with."""
    models: List[SystemCostModel] = [
        CedarCostModel(cedar_optimizer),
        PlumberCostModel(),
        PecanCostModel(),
        TfDataCostModel(),
        RayDataCostModel(),
        FastFlowCostModel(),
        AeroCostModel(),
    ]
    return {model.name: model for model in models}


def calculate_all(
    plan: PhysicalPlan,
    profile: Dict[str, Any],
    cedar_optimizer: Optional[Any] = None,
    workers: int = 1,
) -> Dict[str, SystemCostEstimate]:
    """Score one plan with every registered system cost model."""
    registry = default_models(cedar_optimizer)
    estimates: Dict[str, SystemCostEstimate] = {}
    for name, model in registry.items():
        try:
            estimates[name] = model.calculate_cost(plan, profile, workers)
        except Exception as exc:  # noqa: BLE001 - report, do not hide
            estimates[name] = SystemCostEstimate(name, float("nan"), f"error: {exc}")
    return estimates
