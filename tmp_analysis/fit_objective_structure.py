"""Compare two offload cost structures against measured plans.

    python tmp_analysis/fit_objective_structure.py <results.json> [more.json]

Every measured cell gives a plan, its worker count and its throughput, i.e. a
per-worker per-record cycle time ``W / throughput``.  Two candidate
structures predict that cycle from the same per-stage terms:

  additive (current): stage service and marshalling both sit on the worker
      lane, so ``cycle = local_work + ray_service + marshalling``
  lanes: the stage owns its own lane and the worker only pays the marshalling
      and round trip, so
      ``cycle = max(local_work + marshalling, ray_service, smp_service)``

The script reports, per structure, the correlation with the measured cycle
and the median predicted/measured ratio.
"""

import importlib
import json
import math
import os
import statistics
import sys
from pathlib import Path

ROOT = Path("/workspace/OptimalCedar")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tmp_analysis"))

os.environ.setdefault("CEDAR_MATCH_PROFILE_RESOURCES", "1")
os.environ.setdefault("CEDAR_PROFILE_MATCH_CPU_BUDGET", "64")
os.environ.setdefault("CEDAR_PROFILE_MATCH_RAY_CPU_BUDGET", "64")

from cedar.compose import OptimizerOptions  # noqa: E402
from cedar.compose.dp_optimizer import DpOptimizer  # noqa: E402
from cedar.compose.optimizer import PhysicalPlan  # noqa: E402
from cedar.pipes import PipeVariantType  # noqa: E402

WORKLOAD_DATASETS = {
    "simclr": ("evaluation/pipelines/target_pipeline/simclr/cedar_dataset.py",
               {"workload": "simclr"}),
    "blip": ("evaluation/pipelines/target_pipeline/blip/cedar_dataset.py",
             {"workload": "blip",
              "dataset_path": str(ROOT / "datasets/target_pipeline_bench/blip.jsonl")}),
    "clip": ("evaluation/pipelines/target_pipeline/clip/cedar_dataset.py",
             {"workload": "clip",
              "dataset_path": str(ROOT / "datasets/target_pipeline_bench/clip.jsonl")}),
    "dino": ("evaluation/pipelines/target_pipeline/dino/cedar_dataset.py",
             {"workload": "dino",
              "dataset_path": str(ROOT / "datasets/target_pipeline_bench/dino.jsonl"),
              "views": "2"}),
    "alpaca_cot": ("evaluation/pipelines/alpaca_cot/cedar_dataset.py", {}),
    "pile_hackernews": ("evaluation/pipelines/pile_hackernews/cedar_dataset.py", {}),
    "pile_pubmed_abstracts": (
        "evaluation/pipelines/target_pipeline/hub/pile_pubmed_abstracts/cedar_dataset.py",
        {}),
    "pile_uspto_backgrounds": (
        "evaluation/pipelines/target_pipeline/hub/pile_uspto_backgrounds/cedar_dataset.py",
        {}),
    "bloom_oscar": ("evaluation/pipelines/bloom_oscar/cedar_dataset.py", {}),
}
PROFILE_DIRS = [
    ROOT / "outputs/pico_drained_20260914/profiles",
    ROOT / "outputs/plumber_bench_20260912",
]


def profile_for(workload: str) -> Path:
    for base in PROFILE_DIRS:
        candidate = base / f"{workload}_profile.yaml"
        if candidate.is_file():
            return candidate
    raise SystemExit(f"no profile for {workload}")


def build_optimizer(workload: str, samples: int = 2000):
    from evaluation.cedar_utils import CedarEvalSpec
    from evaluation.eval_cedar import import_module_from_path

    dataset_file, kwargs = WORKLOAD_DATASETS[workload]
    module = import_module_from_path(str((ROOT / dataset_file).resolve()))
    getter = getattr(module, "get_target_dataset", None) or module.get_dataset
    profile = profile_for(workload)
    spec = CedarEvalSpec(
        batch_size=4,
        num_total_samples=samples,
        num_epochs=0,
        config=None,
        kwargs=dict(kwargs),
        use_ray=True,
        ray_ip="172.23.166.105:6379",
        profiled_stats=str(profile),
        disable_optimizer=True,
        disable_controller=True,
        disable_caching=True,
    )
    dataset = getter(spec)
    feature = next(iter(dataset.features.values()))
    optimizer = DpOptimizer()
    feature.set_optimizer(optimizer)
    optimizer.run(
        str(profile),
        OptimizerOptions(
            enable_prefetch=True,
            est_throughput=None,
            available_local_cpus=64,
            enable_offload=True,
            enable_reorder=True,
            enable_local_parallelism=True,
            enable_fusion=True,
            num_samples=samples,
            use_my_optimizer=2,
            reorder_timeout_sec=300.0,
        ),
    )
    return optimizer


def stage_terms(optimizer, blocks):
    """Per-stage (local, ray_service, marshalling, smp) terms."""
    provider_type = None
    from cedar.compose.dp_optimizer import BlockCandidateProvider

    provider = BlockCandidateProvider(optimizer, optimizer._dp_inner_ops)
    provider.prepare()
    prev_mask = 0
    local = ray_service = marshalling = smp = 0.0
    for order, variant, _cache, parallelism in blocks:
        block = provider.candidate_for_order(
            order, variant, prefix_mask=prev_mask, parallelism=parallelism
        )
        boundary_local, boundary_parallel = (
            optimizer._dp_stage_boundary_components(prev_mask, block)
        )
        service_width = max(
            1.0, optimizer._dp_service_parallelism(block)
        ) * optimizer._dp_stage_concurrency(variant)
        compute = block.cost * optimizer._dp_stage_factor(variant) / service_width
        if variant in (PipeVariantType.RAY, PipeVariantType.TF_RAY):
            round_trip = optimizer._dp_cross_host_round_trip_ms(
                optimizer._dp_stage_transport_bytes(prev_mask, block)
            )
            ray_service += compute
            marshalling += boundary_local + boundary_parallel + round_trip
        elif variant == PipeVariantType.SMP:
            if optimizer._dp_effective_parallelism(block) <= 1:
                local += compute + boundary_local + boundary_parallel
            else:
                smp = max(smp, compute + boundary_local + boundary_parallel)
        else:
            local += block.cost
        prev_mask |= block.mask
    return local, ray_service, marshalling, smp


def spearman(a, b):
    n = len(a)
    if n < 3:
        return float("nan")

    def ranks(values):
        order = sorted(range(n), key=lambda i: values[i])
        out = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and values[order[j + 1]] == values[order[i]]:
                j += 1
            average = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = average
            i = j + 1
        return out

    ra, rb = ranks(a), ranks(b)
    mean = (n + 1) / 2.0
    num = sum((ra[i] - mean) * (rb[i] - mean) for i in range(n))
    den = sum((v - mean) ** 2 for v in ra)
    return num / den if den else float("nan")


def main() -> int:
    rows = []
    for path in sys.argv[1:]:
        payload = json.loads(Path(path).read_text())
        workload = Path(path).stem.split("__")[0]
        if workload not in WORKLOAD_DATASETS:
            continue
        optimizer = build_optimizer(workload)
        for run in payload.get("runs", []):
            perf = run.get("perf_time_sec")
            raw = run.get("raw_workload_results") or {}
            batch = raw.get("batch_size") or 4
            samples = run.get("num_samples") or 0
            plans = run.get("physical_plans_by_feature") or {}
            if not plans or not perf or perf != perf or perf == float("inf"):
                continue
            plan_dict = next(iter(plans.values()))
            plan_dict = json.loads(json.dumps(plan_dict))
            for pipe in (plan_dict.get("pipes") or {}).values():
                pipe["variant"] = pipe.get("variant") or "INPROCESS"
                pipe.setdefault("variant_ctx", {"variant_type": pipe["variant"]})
                pipe["variant_ctx"].setdefault(
                    "variant_type", pipe["variant"]
                )
            plan_dict["graph"] = {
                int(key): value
                for key, value in (plan_dict.get("graph") or {}).items()
            }
            plan_dict["pipes"] = {
                int(key): value
                for key, value in (plan_dict.get("pipes") or {}).items()
            }
            try:
                plan = PhysicalPlan.from_dict(plan_dict)
            except Exception as exc:  # noqa: BLE001
                print(f"  skip {workload}/{run.get('optimizer')}: {exc}")
                continue
            workers = max(1, int(plan.n_local_workers or 1))
            records = samples / batch if samples else None
            if not records:
                continue
            measured_cycle = workers * perf / records * 1000.0
            try:
                blocks = optimizer._dp_blocks_from_physical_plan(
                    plan, optimizer._dp_inner_ops
                )
                local, ray_service, marshalling, smp = stage_terms(
                    optimizer, blocks
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  skip {workload}/{run.get('optimizer')}: {exc}")
                continue
            rows.append(
                {
                    "workload": workload,
                    "optimizer": run.get("optimizer"),
                    "measured": measured_cycle,
                    "local": local,
                    "ray_service": ray_service,
                    "marshalling": marshalling,
                    "smp": smp,
                    "additive": max(
                        local + ray_service + marshalling, smp, 0.0
                    ),
                    "lanes": max(local + marshalling, ray_service, smp, 0.0),
                }
            )

    if not rows:
        print("no measured plans")
        return 1
    measured = [row["measured"] for row in rows]
    print(f"\n{len(rows)} measured plans")
    print(
        f"{'workload':<22}{'planner':<22}{'meas':>8}{'local':>8}"
        f"{'ray':>8}{'marsh':>8}{'smp':>7}{'add':>8}{'lane':>8}"
    )
    for row in sorted(rows, key=lambda r: r["measured"]):
        print(
            f"{row['workload']:<22}{str(row['optimizer']):<22}"
            f"{row['measured']:>8.2f}{row['local']:>8.2f}"
            f"{row['ray_service']:>8.2f}{row['marshalling']:>8.2f}"
            f"{row['smp']:>7.2f}{row['additive']:>8.2f}{row['lanes']:>8.2f}"
        )
    for name in ("additive", "lanes"):
        predicted = [row[name] for row in rows]
        ratios = [
            row[name] / row["measured"] for row in rows if row["measured"] > 0
        ]
        print(
            f"\n{name:<10} spearman={spearman(measured, predicted):.2f} "
            f"median(pred/meas)={statistics.median(ratios):.2f} "
            f"mean_abs_log_err="
            f"{statistics.mean(abs(math.log(r)) for r in ratios):.2f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
