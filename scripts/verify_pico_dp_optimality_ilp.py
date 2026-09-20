"""Verify the PICO DP with an independent integer program.

PICO = ``SimpleDpWorkersWidthBoundaryOptimizer`` (selector 27).  Its DP
minimises a four-coordinate objective (local / ray / smp / gpu serial lanes)
that is *additively accumulated* per transition, with the scalar score being
``max(lane)`` (lane exposure is zero by default).  That makes the search a
min-max shortest path over the DP's own transition graph.

This script

  1. runs the optimizer with the general Pareto DP at a fixed worker count,
     recording **every** transition it evaluates (upper-bound pruning off), with
     the per-edge lane increments taken from the optimizer's own accumulation,
  2. rebuilds that transition graph as a MILP: binary arc variables, flow
     conservation, lane coordinates as linear functions of the arcs, and
     ``min T s.t. T >= lane`` for each lane,
  3. solves it with HiGHS (``scipy.optimize.milp``) and compares the optimum
     with the cost the DP reported.

Equal values mean the DP's dominance pruning / state merging / shortest-path
selection returned the exact optimum of its cost model over the candidate
space it enumerates.  A MILP optimum below the DP cost would be a counterexample.

Usage (inside the container):
  python -u scripts/verify_pico_dp_optimality_ilp.py \
      --workload commonvoice \
      --profile outputs/.../commonvoice/profiles/shared.yaml \
      --workers 64
"""

import argparse
import logging
import os
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from scipy.optimize import Bounds, LinearConstraint, milp  # noqa: E402
from scipy.sparse import coo_matrix  # noqa: E402

from cedar.compose import OptimizerOptions  # noqa: E402
from cedar.compose.dp_optimizer import (  # noqa: E402
    BackPointer,
    DpOptimizer,
    DpResourceUsage,
    ExtensibleDpSearch,
)
from cedar.compose.simple_dp_ablation_optimizer import (  # noqa: E402
    SimpleDpWorkersWidthBoundaryOptimizer,
)
from cedar.sources import LocalFSSource  # noqa: E402

DATASETS = {
    "commonvoice": (
        "evaluation/pipelines/commonvoice/cedar_dataset.py",
        "datasets/commonvoice/cv15_en_train_300000",
        1,
        "CommonvoiceFeature",
    ),
    "simclrv2": (
        "evaluation/pipelines/simclrv2/cedar_dataset.py",
        "evaluation/datasets/imagenette2/imagenette2/train",
        4,
        "SimCLRV2Feature",
    ),
}


@dataclass
class Edge:
    prev_mask: int
    prev_state: str
    next_mask: int
    next_state: str
    inc: Tuple[float, float, float, float]
    block: str
    order: Tuple[int, ...]
    variant: Any
    parallelism: int
    next_cpu: Dict[str, float]


def build_feature(workload: str):
    from evaluation.eval_cedar import import_module_from_path

    module_path, data_dir, batch_size, cls_name = DATASETS[workload]
    module = import_module_from_path(str((ROOT / module_path).resolve()))
    feature = getattr(module, cls_name)(batch_size=batch_size)
    feature.apply(LocalFSSource(str(ROOT / data_dir), recursive=True, max_samples=64))
    return feature


def state_key(state) -> str:
    return repr(state)


def patch_search(runs: List[Dict[str, Any]], captured: Dict[str, Any]) -> None:
    """Record every evaluated transition per search run, plus its result."""
    state = {"current": None}

    def traced_try_extend(
        self,
        dp,
        back,
        prev_mask,
        next_mask,
        block_mask,
        incumbent_score=float("inf"),
    ):
        run = state["current"]
        for block in self.block_provider.candidates_for_prefix(
            prev_mask, block_mask
        ):
            if not self._block_can_follow(prev_mask, block):
                continue
            regular = self.optimizer._dp_regular_transition_cost(prev_mask, block)
            for cur_state, objectives in dp[prev_mask].items():
                resource = DpResourceUsage()
                if self.parallel_stage_cpu_limit is not None:
                    resource = (
                        cur_state.parallel_stage_cpus
                        + self.optimizer._dp_parallel_stage_cpu_cost(block)
                    )
                    if resource > self.parallel_stage_cpu_limit:
                        continue
                for choice in self.cache_policy.transitions(
                    prev_mask, next_mask, cur_state, regular, block, resource
                ):
                    for previous in objectives:
                        candidate = self._accumulate_objective(
                            previous,
                            choice.extra_cost,
                            block,
                            choice.replaces_prefix_cost,
                            prev_mask,
                        )
                        if run is not None:
                            run["edges"].append(
                                Edge(
                                    prev_mask=prev_mask,
                                    prev_state=state_key(cur_state),
                                    next_mask=next_mask,
                                    next_state=state_key(choice.state),
                                    inc=(
                                        candidate.local_serial
                                        - previous.local_serial,
                                        candidate.ray_serial
                                        - previous.ray_serial,
                                        candidate.smp_serial
                                        - previous.smp_serial,
                                        candidate.gpu_serial
                                        - previous.gpu_serial,
                                    ),
                                    block=str(block.order)
                                    + "/"
                                    + block.variant.name
                                    + f"/w{int(block.parallelism or 1)}",
                                    order=tuple(block.order),
                                    variant=block.variant,
                                    parallelism=int(block.parallelism or 1),
                                    next_cpu=dict(
                                        choice.state.parallel_stage_cpus.as_dict()
                                    ),
                                )
                            )
                        frontier = dp[next_mask].setdefault(choice.state, [])
                        if any(self._dominates(old, candidate) for old in frontier):
                            continue
                        removed = [
                            old
                            for old in frontier
                            if self._dominates(candidate, old)
                        ]
                        for old in removed:
                            frontier.remove(old)
                            back[next_mask].pop((choice.state, old), None)
                        frontier.append(candidate)
                        back[next_mask][(choice.state, candidate)] = BackPointer(
                            prev_mask=prev_mask,
                            prev_state=cur_state,
                            prev_objective=previous,
                            block=block,
                            cache_after_idx=choice.cache_after_idx,
                        )

    def all_search_classes(cls=ExtensibleDpSearch):
        yield cls
        for sub in cls.__subclasses__():
            yield from all_search_classes(sub)

    def make_traced_run(original_run):
        def _traced_run(self):
            run = {
                "id": len(runs),
                "class": type(self).__name__,
                "workers": getattr(self.optimizer, "_dp_selected_workers", None),
                "cpu_limit": (
                    None
                    if self.parallel_stage_cpu_limit is None
                    else self.parallel_stage_cpu_limit.as_dict()
                ),
                "required_final_cpu": (
                    None
                    if self.required_final_parallel_stage_cpus is None
                    else self.required_final_parallel_stage_cpus.as_dict()
                ),
                "start_state": state_key(self.cache_policy.initial_state()),
                "initial_objective": self._initial_objective(),
                "full_mask": (1 << len(self.inner_ops)) - 1,
                "edges": [],
            }
            runs.append(run)
            state["current"] = run
            result = original_run(self)
            run["result"] = result
            run["cost"] = float(result.cost)
            run["blocks"] = [list(b) for b in result.blocks]
            return result

        return _traced_run

    patched = []
    for cls in all_search_classes():
        if "_try_extend" in cls.__dict__ or cls is ExtensibleDpSearch:
            cls._try_extend = traced_try_extend
            patched.append(cls.__name__)
        if "run" in cls.__dict__ or cls is ExtensibleDpSearch:
            cls.run = make_traced_run(cls.run)
    captured["patched_classes"] = patched



def solve_milp(
    edges: List[Edge], captured: Dict[str, Any], goal_filter: bool = True
) -> Dict[str, Any]:
    """Min-max path MILP over the recorded transition graph."""
    nodes: Dict[str, int] = {}
    start_key = (0, captured["start_state"])
    nodes[str(start_key)] = 0
    for edge in edges:
        nodes.setdefault(str((edge.prev_mask, edge.prev_state)), len(nodes))
        nodes.setdefault(str((edge.next_mask, edge.next_state)), len(nodes))
    n_nodes, n_edges = len(nodes), len(edges)
    full_mask = captured["full_mask"]
    required_cpu = captured.get("required_final_cpu")
    def norm_cpu(payload):
        return {
            k: float(v)
            for k, v in (payload or {}).items()
            if float(v) != 0.0
        }

    def goal_ok(edge: Edge) -> bool:
        if edge.next_mask != full_mask:
            return False
        if (not goal_filter) or required_cpu is None:
            return True
        return norm_cpu(edge.next_cpu) == norm_cpu(required_cpu)

    goal_nodes = [
        e for e in edges
        if goal_ok(e)
    ]
    goals = [
        nodes[str((e.next_mask, e.next_state))] for e in goal_nodes
    ]
    goal_set = set(goals)

    # flow conservation: one row per node (start injects one unit, all other
    # nodes balance), then four lane rows and four T >= lane rows.
    flow_rows, flow_cols, flow_data = [], [], []
    flow_lb, flow_ub = [], []
    for e_idx, edge in enumerate(edges):
        u = nodes[str((edge.prev_mask, edge.prev_state))]
        v = nodes[str((edge.next_mask, edge.next_state))]
        flow_rows += [u, v]
        flow_cols += [e_idx, e_idx]
        flow_data += [-1.0, 1.0]
    for node in range(n_nodes):
        if node == nodes[str(start_key)]:
            # row value is (in - out); the source injects one unit of flow
            flow_lb.append(-1.0)
            flow_ub.append(-1.0)
        elif node in goal_set:
            # The path terminates here; conservation is intentionally not
            # imposed on goals, every other node balances, so the unit of flow
            # can only stop at a full-mask state.
            flow_lb.append(0.0)
            flow_ub.append(1.0)
        else:
            flow_lb.append(0.0)
            flow_ub.append(0.0)

    n_vars = n_edges + 4 + 1
    T_idx = n_vars - 1
    lane_rows, lane_cols, lane_data = [], [], []
    lane_lb, lane_ub = [], []
    base = captured["initial_objective"]
    bases = [
        base.local_serial,
        base.ray_serial,
        base.smp_serial,
        base.gpu_serial,
    ]
    for lane in range(4):
        # Z_lane - sum(inc_e) * x_e = base_lane  (lane coordinates are additive)
        row = len(lane_lb)
        for e_idx, edge in enumerate(edges):
            lane_rows.append(row)
            lane_cols.append(e_idx)
            lane_data.append(edge.inc[lane])
        lane_rows.append(row)
        lane_cols.append(n_edges + lane)
        lane_data.append(-1.0)
        lane_lb.append(-bases[lane])
        lane_ub.append(-bases[lane])
        # T - Z_lane >= 0
        row2 = len(lane_lb)
        lane_rows.append(row2)
        lane_cols.append(n_edges + lane)
        lane_data.append(-1.0)
        lane_rows.append(row2)
        lane_cols.append(T_idx)
        lane_data.append(1.0)
        lane_lb.append(0.0)
        lane_ub.append(np.inf)

    A_flow = coo_matrix(
        (flow_data, (flow_rows, flow_cols)), shape=(n_nodes, n_vars)
    ).tocsr()
    A_lane = coo_matrix(
        (lane_data, (lane_rows, lane_cols)), shape=(len(lane_lb), n_vars)
    ).tocsr()
    from scipy.sparse import vstack

    A = vstack([A_flow, A_lane]).tocsr()
    lb = np.array(flow_lb + lane_lb)
    ub = np.array(flow_ub + lane_ub)

    c = np.zeros(n_vars)
    c[T_idx] = 1.0
    integrality = np.ones(n_vars)
    integrality[n_edges:] = 0
    bounds = Bounds(
        lb=np.zeros(n_vars),
        ub=np.concatenate([np.ones(n_edges), np.full(5, np.inf)]),
    )
    result = milp(
        c,
        constraints=LinearConstraint(A, lb, ub),
        integrality=integrality,
        bounds=bounds,
        options={"disp": False, "time_limit": 3600},
    )
    chosen = [i for i, x in enumerate(result.x[:n_edges]) if x > 0.5] if result.x is not None else []
    # order the chosen arcs along the path so the solution can be replayed
    ordered_edges: List[Edge] = []
    if chosen:
        by_prev: Dict[Tuple[int, str], List[int]] = {}
        for e_idx in chosen:
            by_prev.setdefault(
                (edges[e_idx].prev_mask, edges[e_idx].prev_state), []
            ).append(e_idx)
        cursor = (0, captured["start_state"])
        while cursor in by_prev:
            e_idx = by_prev[cursor][0]
            edge = edges[e_idx]
            ordered_edges.append(edge)
            cursor = (edge.next_mask, edge.next_state)
    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "n_goals": len(goals),
        "status": int(result.status),
        "message": result.message,
        "milp_optimum": float(result.fun) if result.fun is not None else None,
        "chosen_edges": [edges[i].block for i in chosen],
        "ordered_blocks": [
            (e.order, e.variant, e.parallelism) for e in ordered_edges
        ],
        "lanes": (
            None
            if result.x is None
            else [float(result.x[n_edges + k]) for k in range(4)]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", choices=sorted(DATASETS), required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--cpu-budget", type=int, default=64)
    parser.add_argument("--ray-cpu-budget", type=int, default=64)
    parser.add_argument("--no-goal-filter", action="store_true",
                        help="accept any full-mask final state")
    args = parser.parse_args()

    logging.disable(logging.INFO)
    os.environ["CEDAR_MATCH_PROFILE_RESOURCES"] = "1"
    os.environ["CEDAR_PROFILE_MATCH_CPU_BUDGET"] = str(args.cpu_budget)
    os.environ["CEDAR_PROFILE_MATCH_RAY_CPU_BUDGET"] = str(args.ray_cpu_budget)
    os.environ["CEDAR_PROFILE_MATCH_FIXED_LOCAL_WORKERS"] = str(args.workers)
    os.environ["CEDAR_DP_SEARCH_MODE"] = "general"
    os.environ["CEDAR_DP_WORKER_SEARCH"] = "0"

    profile = yaml.safe_load(args.profile.read_text())
    feature = build_feature(args.workload)
    optimizer = SimpleDpWorkersWidthBoundaryOptimizer()
    feature.set_optimizer(optimizer)
    options = OptimizerOptions(
        enable_prefetch=True,
        est_throughput=None,
        available_local_cpus=args.cpu_budget,
        enable_offload=True,
        enable_reorder=True,
        enable_local_parallelism=True,
        enable_fusion=True,
        num_samples=0,
        use_my_optimizer=27,
        reorder_timeout_sec=7200.0,
    )

    runs: List[Dict[str, Any]] = []
    captured: Dict[str, Any] = {}
    patch_search(runs, captured)
    plan = optimizer.run(profile, options)
    print(f"workload        : {args.workload} (fixed W={args.workers})")
    print(f"search classes  : {captured.get('patched_classes')}")
    print(f"final plan cost : {optimizer.calculate_dp_objective_cost(plan=plan):.6f}")
    print()
    header = (
        f"{'run':>4}{'search':>22}{'W':>5}{'DP cost':>14}{'nodes':>8}"
        f"{'edges':>8}{'MILP opt':>16}{'replay':>14}{'gap':>12}"
    )
    print(header)
    for run in runs:
        edges = run["edges"]
        if not edges:
            continue
        milp_result = solve_milp(edges, run, goal_filter=not args.no_goal_filter)
        specs = [
            (tuple(order), variant, False, parallelism)
            for order, variant, parallelism in milp_result["ordered_blocks"]
        ]
        replay = float("nan")
        replay_note = ""
        if specs:
            try:
                replay = optimizer._replay_dp_objective(
                    specs, optimizer._dp_inner_ops
                ).score
            except Exception as exc:  # noqa: BLE001
                replay_note = f"{type(exc).__name__}: {exc}"
        dp_cost = run["cost"]
        optimum = milp_result["milp_optimum"]
        gap = dp_cost - optimum if optimum is not None else float("nan")
        print(
            f"{run['id']:>4}{run['class']:>22}{str(run['workers']):>5}"
            f"{dp_cost:>14.6f}{milp_result['n_nodes']:>8}"
            f"{milp_result['n_edges']:>8}"
            f"{(optimum if optimum is not None else float('nan')):>16.6f}"
            f"{replay:>14.6f}"
            f"{gap:>12.2e}"
        )
        print(f"     DP blocks  : {run['blocks']}")
        print(f"     MILP blocks: {milp_result['chosen_edges']}")
        if replay_note:
            print(f"     replay     : INFEASIBLE -> {replay_note}")
        verdict = (
            "DP IS OPTIMAL"
            if abs(gap) <= 1e-6 * max(1.0, dp_cost)
            else (
                "DP SUBOPTIMAL (but MILP plan not replayable)"
                if replay_note
                else "DP SUBOPTIMAL"
            )
        )
        print(f"     verdict    : {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
