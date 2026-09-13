"""Score the three SimCLRv2 plans under each cost model.

For every plan produced by the three systems (Cedar, Plumber-style, PICO) we
report the per-record cost each cost model assigns to it:

  * Cedar's model      : Optimizer.calculate_cost (size-scaled pipe latency).
  * Plumber-style model: per-stage rate = 1 / sum(member latencies), stage
                         capacity = its configured width, one machine.
  * PICO model         : DpOptimizer.calculate_dp_objective_cost (stage terms).

Measured per-record times come from the same W=8 benchmark runs.

Usage: python -u tmp_analysis/score_three_plans.py [profile.yaml]
"""

import json
import math
import os
import sys
from pathlib import Path

ROOT = Path("/workspace/OptimalCedar")
sys.path.insert(0, str(ROOT))

os.environ.setdefault("CEDAR_MATCH_PROFILE_RESOURCES", "1")
os.environ.setdefault("CEDAR_PROFILE_MATCH_FIXED_LOCAL_WORKERS", "8")
os.environ.setdefault("CEDAR_PROFILE_MATCH_CPU_BUDGET", "64")
os.environ.setdefault("CEDAR_PROFILE_MATCH_RAY_CPU_BUDGET", "64")

import yaml  # noqa: E402

from cedar.compose import OptimizerOptions  # noqa: E402
from cedar.compose.dp_optimizer import (  # noqa: E402
    BlockCandidateProvider,
    DpOptimizer,
)
from cedar.compose.optimizer import PhysicalPlan, PipeVariantType  # noqa: E402
from cedar.sources import LocalFSSource  # noqa: E402
from evaluation.pipelines.simclrv2 import cedar_dataset as original_simclrv2  # noqa: E402
from evaluation.pipelines.target_pipeline.simclr.cedar_dataset import (  # noqa: E402
    DATASET_LOC,
    SimCLRV2Feature,
)

PROFILE = Path(
    sys.argv[1]
    if len(sys.argv) > 1
    else ROOT / "outputs/plumber_bench_20260912/simclr_profile.yaml"
)
PLAN_DIR = ROOT / "tmp_analysis/plans/simclr_bench"
PLANS = {
    "cedar": PLAN_DIR / "optimizer.yaml",
    "plumber": PLAN_DIR / "plumber_optimizer.yaml",
    "pico": PLAN_DIR / "dp_optimizer.yaml",
}
MEASURED_SEC = {"cedar": 2.053, "plumber": 1.078, "pico": 0.517}
WORKERS = 8
MEMBERS = {
    10: [3, 6, 2, 4, 5, 7, 1],
    11: [2, 5, 4, 7, 1],
}


def build_feature():
    data_dir = (
        Path(original_simclrv2.__file__).resolve().parents[2].joinpath(DATASET_LOC)
    )
    feature = SimCLRV2Feature(batch_size=4)
    feature.apply(LocalFSSource(str(data_dir / "imagenette2" / "train"), recursive=True))
    return feature


def options():
    return OptimizerOptions(
        enable_prefetch=True,
        est_throughput=None,
        available_local_cpus=64,
        enable_offload=True,
        enable_reorder=True,
        enable_local_parallelism=True,
        enable_fusion=True,
        num_samples=9469,
        use_my_optimizer=2,
        reorder_timeout_sec=3600.0,
    )


def load_plan(path):
    data = yaml.safe_load(Path(path).read_text())
    payload = data.get("physical_plan", data)
    payload = payload.get("feature_r0", payload)
    payload["graph"] = {int(k): v for k, v in payload["graph"].items()}
    payload["pipes"] = {int(k): v for k, v in payload["pipes"].items()}
    for pipe in payload["pipes"].values():
        pipe["variant"] = pipe.get("variant") or "INPROCESS"
        pipe.setdefault("variant_ctx", {"variant_type": pipe["variant"]})
    return payload


def plumber_prediction(payload, profile):
    """Plumber's rate model applied to an arbitrary materialized plan.

    Stages are whatever the plan declares: a fused pipe is one stage whose
    service is the sum of its members' measured latencies (the rate model has
    no notion of fusion), and a stage's capacity is its configured width on the
    single machine the model assumes.
    """
    latencies = profile["baseline"]["latencies"]
    graph = payload["graph"]
    child_of = {}
    for parent, children in graph.items():
        for child in children if isinstance(children, list) else [children]:
            if child is None or child == "":
                continue
            child_of[int(child)] = parent
    stages = {}
    for p_id, desc in payload["pipes"].items():
        if desc.get("name") == "PrefetcherPipe":
            continue
        members = MEMBERS.get(p_id, [p_id])
        service = sum(float(latencies[m]) / 1e6 for m in members if m in latencies)
        if service <= 0.0:
            continue
        ctx = desc.get("variant_ctx") or {}
        width = int(ctx.get("n_procs") or ctx.get("n_actors") or 1)
        variant = desc.get("variant") or "INPROCESS"
        if variant == "INPROCESS":
            width = 1
        stages[p_id] = (service, width, variant)
    # Per-worker per-record service time = slowest stage on the record's path.
    slowest = max(service / max(1, width) for service, width, _ in stages.values())
    return slowest, stages


def concurrency_aware_score(detail):
    """Aggregate stage terms by the concurrency the backend really provides.

    The current DP objective keeps one coordinate per *backend family* and
    takes the maximum, which implicitly assumes every family overlaps with
    every other one.  The runtime does not behave that way:

      * INPROCESS operators are threads inside one worker process and their
        per-record CPU work is conserved, so their service times add up on the
        worker's core.
      * An SMP stage owns its processes, so it pipelines next to the worker's
        chain: the slower of the two paces the pipeline (max).
      * A Ray stage is submitted by the worker and its payload crosses the
        network for every record; in our measurements that path is on the
        record's critical path (driver serialization + round trip + remote
        service), so it adds to the worker's chain instead of overlapping.
    """
    worker = 0.0
    lanes = []
    for stage in detail:
        variant = stage["variant"]
        lane = (
            stage["per_record_lane"]
            + stage["boundary_local"]
            + stage["boundary_external"]
        )
        if variant == "INPROCESS":
            worker += stage["compute"]
        elif variant in ("RAY", "TF_RAY"):
            worker += lane
        else:
            lanes.append(lane)
    return max([worker] + lanes), worker, lanes


def main():
    feature = build_feature()
    optimizer = DpOptimizer()
    feature.set_optimizer(optimizer)
    optimizer.run(str(PROFILE), options())
    profile = yaml.safe_load(PROFILE.read_text())
    inner_ops = optimizer._dp_inner_ops
    provider = BlockCandidateProvider(optimizer, inner_ops)
    provider.prepare()

    rows = []
    for name, path in PLANS.items():
        payload = load_plan(path)
        plan = PhysicalPlan.from_dict(payload)
        cedar_cost = optimizer.calculate_cost(plan.graph, plan=plan)
        pico_cost = optimizer.calculate_dp_objective_cost(plan=plan)
        plumber_cost, stages = plumber_prediction(payload, profile)
        # Detailed decomposition of our own model for this plan.
        blocks = optimizer._dp_blocks_from_physical_plan(plan, inner_ops)
        block_specs = []
        detail = []
        prev_mask = 0
        for order, variant, cache_after, parallelism in blocks:
            block = provider.candidate_for_order(
                order,
                variant,
                prefix_mask=prev_mask,
                parallelism=parallelism,
            )
            boundary_local, boundary_external = (
                optimizer._dp_stage_boundary_components(prev_mask, block)
            )
            detail.append(
                {
                    "ops": [inner_ops[i] for i in order],
                    "variant": variant.name,
                    "width": parallelism,
                    "compute": round(block.cost, 4),
                    "per_record_lane": round(
                        block.cost / max(1, parallelism)
                        if variant.name in ("RAY", "TF_RAY", "SMP")
                        else block.cost,
                        4,
                    ),
                    "boundary_local": round(boundary_local, 4),
                    "boundary_external": round(boundary_external, 4),
                }
            )
            prev_mask |= block.mask
            block_specs.append((tuple(order), variant, cache_after, parallelism))
        objective = optimizer._replay_dp_objective(block_specs, inner_ops)
        concurrency_score, worker_lane, parallel_lanes = concurrency_aware_score(
            detail
        )
        rows.append(
            {
                "plan": name,
                "measured_ms_per_record": MEASURED_SEC[name] * WORKERS,
                "cedar_model": cedar_cost,
                "plumber_model": plumber_cost,
                "pico_model": pico_cost,
                "pico_families": {
                    "local_serial": round(objective.local_serial, 4),
                    "ray_serial": round(objective.ray_serial, 4),
                    "smp_serial": round(objective.smp_serial, 4),
                    "gpu_serial": round(objective.gpu_serial, 4),
                    "score": round(objective.score, 4),
                },
                "pico_detail": detail,
                "concurrency_aware": {
                    "worker_core_serial": round(worker_lane, 4),
                    "parallel_lanes": [round(lane, 4) for lane in parallel_lanes],
                    "score": round(concurrency_score, 4),
                },
                "plumber_stages": {
                    str(pid): {
                        "service_ms": round(service, 3),
                        "width": width,
                        "variant": variant,
                    }
                    for pid, (service, width, variant) in stages.items()
                },
            }
        )

    print(
        f"{'plan':<8} {'measured':>9} {'cedar model':>12} "
        f"{'plumber model':>14} {'pico model':>11}"
    )
    for row in rows:
        print(
            f"{row['plan']:<8} {row['measured_ms_per_record']:9.2f} "
            f"{row['cedar_model']:12.2f} {row['plumber_model']:14.2f} "
            f"{row['pico_model']:11.2f}"
        )
    out = ROOT / "outputs/plumber_bench_20260912/model_estimates.json"
    out.write_text(json.dumps(rows, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
