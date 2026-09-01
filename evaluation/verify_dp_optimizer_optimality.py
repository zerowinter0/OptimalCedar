"""Randomized exhaustive optimality verifier for :class:`DpOptimizer`.

Each generated case has exactly six reorderable operators.  The verifier
enumerates every legal topological order, every fusion partition, every
assignment of the three physical backends used by the case, and (when a pool
limit is supplied) every feasible integer stage-width assignment. It then
compares that independent oracle with the plan produced by DpOptimizer.

All optimizer passes except caching are enabled. Prefetch and local parallelism
do not alter this cost model, but enabling them exercises the requested setup.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from cedar.compose import Feature
from cedar.compose import constants
from cedar.compose.dp_optimizer import DpOptimizer
from cedar.compose.optimizer import OptimizerOptions, PhysicalPlan
from cedar.pipes import MapperPipe, Pipe, PipeVariantType
from cedar.sources import IterSource


def _identity(value):
    return value


NUM_OPERATORS = 6
BACKENDS: Tuple[str, ...] = ("INPROCESS", "SMP", "RAY")


def unconstrained_plan_count() -> int:
    """Return the exact six-op plan count when there are no dependencies."""
    partition_assignments = sum(
        math.comb(NUM_OPERATORS - 1, blocks - 1) * len(BACKENDS) ** blocks
        for blocks in range(1, NUM_OPERATORS + 1)
    )
    return math.factorial(NUM_OPERATORS) * partition_assignments


@dataclass(frozen=True)
class OperatorSpec:
    tag: str
    size_ratio: float
    selectivity: float
    costs: Dict[str, float]


@dataclass(frozen=True)
class GeneratedCase:
    seed: int
    operators: Tuple[OperatorSpec, ...]
    # (predecessor index, successor index)
    dependencies: Tuple[Tuple[int, int], ...]


@dataclass(frozen=True)
class ExhaustiveResult:
    cost: float
    order: Tuple[int, ...]
    backends: Tuple[str, ...]
    legal_orders: int
    enumerated_plans: int


@dataclass(frozen=True)
class VerificationResult:
    oracle: ExhaustiveResult
    dp_cost: float
    dp_order: Tuple[int, ...]
    dp_backends: Tuple[str, ...]


class GeneratedFeature(Feature):
    def __init__(self, case: GeneratedCase) -> None:
        super().__init__()
        self.case = case

    def _compose(self, source_pipes: List[Pipe]) -> Pipe:
        pipe = source_pipes[0]
        for index, spec in enumerate(self.case.operators):
            predecessors = [
                self.case.operators[pred].tag
                for pred, succ in self.case.dependencies
                if succ == index
            ]
            pipe = MapperPipe(pipe, _identity, tag=spec.tag)
            if predecessors:
                pipe = pipe.depends_on(predecessors)
        return pipe


def generate_case(seed: int, max_dependencies: int = 4) -> GeneratedCase:
    """Generate one deterministic, sparse six-operator problem."""
    if max_dependencies < 0:
        raise ValueError("max_dependencies must be non-negative")

    rng = random.Random(seed)
    operators = tuple(
        OperatorSpec(
            tag=f"op_{index}",
            size_ratio=round(rng.uniform(0.2, 1.5), 8),
            selectivity=round(rng.uniform(0.2, 1.0), 8),
            costs={
                backend: round(rng.uniform(0.05, 20.0), 8)
                for backend in BACKENDS
            },
        )
        for index in range(NUM_OPERATORS)
    )

    possible_edges = [
        (pred, succ)
        for pred in range(NUM_OPERATORS)
        for succ in range(pred + 1, NUM_OPERATORS)
    ]
    edge_limit = min(max_dependencies, len(possible_edges))
    edge_count = rng.randint(0 if edge_limit == 0 else 1, edge_limit)
    dependencies = tuple(sorted(rng.sample(possible_edges, edge_count)))
    return GeneratedCase(seed, operators, dependencies)


def _is_legal_order(
    order: Sequence[int], dependencies: Iterable[Tuple[int, int]]
) -> bool:
    positions = {operator: position for position, operator in enumerate(order)}
    return all(positions[pred] < positions[succ] for pred, succ in dependencies)


def _effective_dependencies(
    case: GeneratedCase,
) -> Tuple[Tuple[int, int], ...]:
    """Add the Cedar Feature sink invariant to generated dependencies."""
    output = len(case.operators) - 1
    return tuple(
        sorted(set(case.dependencies) | {(i, output) for i in range(output)})
    )


def _partition_order(
    order: Tuple[int, ...], boundary_mask: int
) -> Tuple[Tuple[int, ...], ...]:
    blocks: List[Tuple[int, ...]] = []
    start = 0
    for position in range(NUM_OPERATORS - 1):
        if boundary_mask & (1 << position):
            blocks.append(order[start : position + 1])
            start = position + 1
    blocks.append(order[start:])
    return tuple(blocks)


def _plan_cost(
    case: GeneratedCase,
    blocks: Sequence[Sequence[int]],
    block_backends: Sequence[str],
    block_widths: Sequence[int],
) -> float:
    """Independent width-aware resource-family bottleneck objective."""
    item_size = 1.0
    cardinality = 1.0
    volume = 1.0
    local_serial = 0.0
    ray_serial = 0.0
    smp_serial = 0.0
    for block, backend, width in zip(blocks, block_backends, block_widths):
        block_input = volume
        block_compute = 0.0
        if backend == "SMP":
            boundary_throughput = constants.LOCAL_PARALLELISM_THRESHOLD
        elif backend == "RAY":
            boundary_throughput = constants.RAY_STAGE_BOUNDARY_THROUGHPUT
        else:
            boundary_throughput = None
        for operator_index in block:
            operator = case.operators[operator_index]
            operator_cost = operator.costs[backend]
            if backend != "INPROCESS":
                operator_cost = max(
                    operator_cost,
                    operator.costs["INPROCESS"]
                    / constants.MAX_UNIDENTIFIABLE_OPERATOR_SPEEDUP,
                )
            block_compute += volume * operator_cost
            item_size *= operator.size_ratio
            cardinality *= operator.selectivity
            volume = item_size * cardinality
        boundary_work = 0.0
        if boundary_throughput is not None:
            boundary_work = (
                (block_input + volume)
                / boundary_throughput
                * 1000.0
            )
        if backend == "INPROCESS":
            local_serial += block_compute
        elif backend == "RAY":
            ray_serial = max(
                ray_serial, block_compute / width + boundary_work
            )
        else:
            smp_serial = max(
                smp_serial, block_compute / width + boundary_work
            )
    return max(local_serial, ray_serial, smp_serial)


def exhaustive_oracle(
    case: GeneratedCase,
    parallel_stage_limit: int | None = None,
) -> ExhaustiveResult:
    """Enumerate every legal order, partition, backend, and integer width."""
    best_cost = math.inf
    best_order: Tuple[int, ...] = ()
    best_backends: Tuple[str, ...] = ()
    legal_orders = 0
    enumerated_plans = 0
    for order in itertools.permutations(range(NUM_OPERATORS)):
        if not _is_legal_order(order, _effective_dependencies(case)):
            continue
        legal_orders += 1
        for boundary_mask in range(1 << (NUM_OPERATORS - 1)):
            blocks = _partition_order(order, boundary_mask)
            for block_backends in itertools.product(BACKENDS, repeat=len(blocks)):
                max_width = parallel_stage_limit or 1
                width_choices = [
                    (1,)
                    if backend == "INPROCESS"
                    else tuple(range(1, max_width + 1))
                    for backend in block_backends
                ]
                for block_widths in itertools.product(*width_choices):
                    if parallel_stage_limit is not None:
                        ray_width = sum(
                            width
                            for backend, width in zip(
                                block_backends, block_widths
                            )
                            if backend == "RAY"
                        )
                        smp_width = sum(
                            width
                            for backend, width in zip(
                                block_backends, block_widths
                            )
                            if backend == "SMP"
                        )
                        if (
                            ray_width > parallel_stage_limit
                            or smp_width > parallel_stage_limit
                        ):
                            continue
                    enumerated_plans += 1
                    cost = _plan_cost(
                        case,
                        blocks,
                        block_backends,
                        block_widths,
                    )
                    if cost < best_cost:
                        best_cost = cost
                        best_order = order
                        best_backends = tuple(
                            backend
                            for block, backend in zip(
                                blocks, block_backends
                            )
                            for _ in block
                        )
    if not best_order:
        raise RuntimeError("Generated dependency graph has no legal order")
    return ExhaustiveResult(
        best_cost, best_order, best_backends, legal_orders, enumerated_plans
    )


def _offload_throughput(
    baseline_throughput: float,
    total_inprocess_cost: float,
    inprocess_cost: float,
    target_cost: float,
) -> float:
    """Invert Cedar's Amdahl calculation so it yields ``target_cost``."""
    denominator = 1.0 + (target_cost - inprocess_cost) / total_inprocess_cost
    if denominator <= 0:
        raise ValueError("Generated backend cost cannot be represented")
    return baseline_throughput / denominator


def build_profile(
    case: GeneratedCase, feature: GeneratedFeature
) -> Tuple[Dict, Dict[str, int]]:
    """Encode direct per-backend costs in Cedar's profiling representation."""
    p_id_by_tag = {
        pipe.tag: p_id
        for p_id, pipe in feature.logical_pipes.items()
        if pipe.tag is not None
    }
    source_p_id = next(
        p_id for p_id, pipe in feature.logical_pipes.items() if pipe.is_source()
    )

    total_inprocess_cost = sum(
        operator.costs["INPROCESS"] for operator in case.operators
    )
    baseline_throughput = 1000.0 / total_inprocess_cost
    input_sizes = {source_p_id: 1.0}
    output_sizes = {source_p_id: 1.0}
    latencies = {source_p_id: 0.0}
    selectivities = {}

    for operator in case.operators:
        p_id = p_id_by_tag[operator.tag]
        input_sizes[p_id] = 1.0
        output_sizes[p_id] = operator.size_ratio
        latencies[p_id] = operator.costs["INPROCESS"]
        selectivities[p_id] = operator.selectivity

    offloads: Dict[str, Dict[int, Dict[str, float]]] = {
        "SMP": {},
        "RAY": {},
    }
    for backend in ("SMP", "RAY"):
        for operator in case.operators:
            p_id = p_id_by_tag[operator.tag]
            offloads[backend][p_id] = {
                "throughput": _offload_throughput(
                    baseline_throughput,
                    total_inprocess_cost,
                    operator.costs["INPROCESS"],
                    operator.costs[backend],
                )
            }

    profile = {
        "baseline": {
            "throughput": baseline_throughput,
            "latencies": latencies,
            "input_sizes": input_sizes,
            "output_sizes": output_sizes,
            "selectivities": selectivities,
        },
        "disk_info": {"read_latency": 0.0, "write_latency": 0.0},
        "offloads": offloads,
    }
    return profile, p_id_by_tag


def _linear_order(plan: PhysicalPlan, source_p_id: int) -> List[int]:
    order: List[int] = []
    current = source_p_id
    visited = {source_p_id}
    while plan.graph[current]:
        if len(plan.graph[current]) != 1:
            raise RuntimeError("DpOptimizer returned a non-linear plan")
        current = next(iter(plan.graph[current]))
        if current in visited:
            raise RuntimeError("DpOptimizer returned a cyclic plan")
        visited.add(current)
        order.append(current)
    if len(visited) != len(plan.graph):
        raise RuntimeError("DpOptimizer returned a disconnected plan")
    return order


def run_dp_optimizer(
    case: GeneratedCase,
    parallel_stage_limit: int | None = None,
) -> Tuple[float, Tuple[int, ...], Tuple[str, ...]]:
    feature = GeneratedFeature(case)
    feature.apply(IterSource([0]))
    profile, p_id_by_tag = build_profile(case, feature)
    optimizer = DpOptimizer()
    optimizer.init(feature.logical_pipes, feature.logical_adj_list)
    resource_env_names = (
        "CEDAR_MATCH_PROFILE_RESOURCES",
        "CEDAR_PROFILE_MATCH_FIXED_LOCAL_WORKERS",
        "CEDAR_PROFILE_MATCH_CPU_BUDGET",
        "CEDAR_DP_RUNTIME_CPU_RESERVE_PER_WORKER",
    )
    old_resource_env = {
        name: os.environ.get(name) for name in resource_env_names
    }
    try:
        if parallel_stage_limit is None:
            for name in resource_env_names:
                os.environ.pop(name, None)
        else:
            if parallel_stage_limit < 0:
                raise ValueError("parallel_stage_limit must be non-negative")
            os.environ["CEDAR_MATCH_PROFILE_RESOURCES"] = "1"
            os.environ["CEDAR_PROFILE_MATCH_FIXED_LOCAL_WORKERS"] = "1"
            os.environ["CEDAR_PROFILE_MATCH_CPU_BUDGET"] = str(
                parallel_stage_limit + 1
            )
            os.environ["CEDAR_DP_RUNTIME_CPU_RESERVE_PER_WORKER"] = "0"
        plan = optimizer.run(
            profile,
            OptimizerOptions(
                enable_prefetch=True,
                enable_offload=True,
                enable_reorder=True,
                enable_local_parallelism=True,
                available_local_cpus=16,
                enable_fusion=True,
                enable_caching=False,
            ),
        )
    finally:
        for name, old_value in old_resource_env.items():
            if old_value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old_value
    source_p_id = next(
        p_id for p_id, pipe in feature.logical_pipes.items() if pipe.is_source()
    )
    ordered_p_ids = _linear_order(plan, source_p_id)
    index_by_p_id = {
        p_id_by_tag[operator.tag]: index
        for index, operator in enumerate(case.operators)
    }
    order: List[int] = []
    backends: List[str] = []
    for p_id in ordered_p_ids:
        desc = plan.pipe_descs[p_id]
        if p_id in index_by_p_id:
            block = [index_by_p_id[p_id]]
        elif desc.fused_pipes:
            block = [index_by_p_id[x] for x in desc.fused_pipes]
        else:
            continue
        order.extend(block)
        backends.extend([desc.variant_type.name] * len(block))
    cost = optimizer._last_dp_state_cost
    return cost, tuple(order), tuple(backends)


def verify_case(
    case: GeneratedCase,
    rel_tol: float = 1e-10,
    abs_tol: float = 1e-10,
    parallel_stage_limit: int | None = None,
) -> VerificationResult:
    oracle = exhaustive_oracle(
        case,
        parallel_stage_limit=parallel_stage_limit,
    )
    dp_cost, dp_order, dp_backends = run_dp_optimizer(
        case,
        parallel_stage_limit=parallel_stage_limit,
    )

    effective_dependencies = _effective_dependencies(case)
    if not _is_legal_order(dp_order, effective_dependencies):
        raise AssertionError(
            f"DpOptimizer returned an illegal order: seed={case.seed}, "
            f"order={dp_order}, dependencies={effective_dependencies}"
        )
    if not math.isclose(dp_cost, oracle.cost, rel_tol=rel_tol, abs_tol=abs_tol):
        raise AssertionError(
            "DpOptimizer is not globally optimal for generated case: "
            f"seed={case.seed}, "
            f"oracle={oracle.cost:.15g}, "
            f"dp={dp_cost:.15g}, oracle_order={oracle.order}, "
            f"dp_order={dp_order}, oracle_backends={oracle.backends}, "
            f"dp_backends={dp_backends}"
        )
    return VerificationResult(oracle, dp_cost, dp_order, dp_backends)


def _write_failed_case(path: Path, case: GeneratedCase) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(case), indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate sparse six-operator pipelines and compare DpOptimizer "
            "against an independent exhaustive oracle."
        )
    )
    parser.add_argument("--num-cases", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--max-dependencies", type=int, default=4)
    parser.add_argument("--rel-tol", type=float, default=1e-10)
    parser.add_argument("--abs-tol", type=float, default=1e-10)
    parser.add_argument(
        "--failed-case",
        type=Path,
        default=Path("evaluation/failed_dp_optimality_case.json"),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional machine-readable aggregate result.",
    )
    args = parser.parse_args()

    if args.num_cases <= 0:
        parser.error("--num-cases must be positive")

    print(f"unconstrained_plan_count={unconstrained_plan_count()}")
    seed_rng = random.Random(args.seed)
    total_legal_orders = 0
    total_enumerated_plans = 0
    for case_index in range(args.num_cases):
        case_seed = seed_rng.randrange(0, 2**63)
        case = generate_case(case_seed, args.max_dependencies)
        try:
            result = verify_case(
                case,
                args.rel_tol,
                args.abs_tol,
            )
        except Exception:
            _write_failed_case(args.failed_case, case)
            print(f"Failed case saved to {args.failed_case}")
            raise

        print(
            f"[case {case_index:04d}] PASS seed={case_seed} "
            f"dependencies={len(case.dependencies)} "
            f"legal_orders={result.oracle.legal_orders} "
            f"enumerated_plans={result.oracle.enumerated_plans} "
            f"cost={result.dp_cost:.12f}"
        )
        total_legal_orders += result.oracle.legal_orders
        total_enumerated_plans += result.oracle.enumerated_plans

    print(
        f"PASS: DpOptimizer matched the exhaustive optimum in "
        f"all {args.num_cases} generated cases."
    )
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps(
                {
                    "status": "pass",
                    "num_operators": NUM_OPERATORS,
                    "num_cases": args.num_cases,
                    "seed": args.seed,
                    "max_dependencies": args.max_dependencies,
                    "backends": list(BACKENDS),
                    "passes": ["reorder", "fusion", "offload"],
                    "cache_enabled": False,
                    "total_legal_orders": total_legal_orders,
                    "total_enumerated_plans": total_enumerated_plans,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

if __name__ == "__main__":
    main()
