"""Fit the parallel-stage concurrency factor from measured plans.

Every comparison cell stores the plan a planner produced, the throughput the
run measured, and (via ``pico_plan_costs_by_feature``) the isolated-cost score.
Scoring the same plan with different
``CEDAR_DP_STAGE_CONCURRENCY_{RAY,SMP}`` values shows which factor makes the
model's ranking agree with execution: the factor the runtime actually gets
from batched submissions and in-flight batches.

Usage (inside the container):
  python -u tmp_analysis/fit_stage_concurrency.py alpaca_cot [more workloads]
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path("/workspace/OptimalCedar")
sys.path.insert(0, str(ROOT))

os.environ.setdefault("CEDAR_MATCH_PROFILE_RESOURCES", "1")
os.environ.setdefault("CEDAR_PROFILE_MATCH_CPU_BUDGET", "64")
os.environ.setdefault("CEDAR_PROFILE_MATCH_RAY_CPU_BUDGET", "64")

import importlib  # noqa: E402

import yaml  # noqa: E402

from cedar.compose import OptimizerOptions  # noqa: E402
from cedar.compose.dp_optimizer import DpOptimizer  # noqa: E402
from cedar.compose.optimizer import PhysicalPlan  # noqa: E402


# Records in each workload's data set: a drained full pass processes each one
# exactly once (verified with per-operator call counters: parse_and_format
# calls == file lines), while the harness's sample counter can be inflated by
# its per-batch accounting, so the audit derives throughput from these counts.
WORKLOAD_RECORDS = {
    "simclr": 9469,
    "blip": 1000,
    "clip": 1000,
    "dino": 1000,
    "alpaca_cot": 74771,
    "pile_hackernews": 100000,
    "pile_pubmed_abstracts": 100000,
    "pile_uspto_backgrounds": 100000,
    "bloom_oscar": 50000,
}
DRAINED_RUN = "pico_drained_20260914"

RUNS = [
    # Drained full-pass measurements take precedence: stop-at-N runs report
    # the time to fill a deep in-flight window for buffered plans.
    ROOT / "outputs/pico_drained_20260914",
    ROOT / "outputs/pico_ten_workloads_20260913b",
    ROOT / "outputs/pico_djpecan_20260914",
    ROOT / "outputs/pico_missing_20260914",
]
GRID = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03, 0.015]

WORKLOADS = {
    "simclr": (
        "evaluation/pipelines/target_pipeline/simclr/cedar_dataset.py",
        {"workload": "simclr"},
        8000,
    ),
    "blip": (
        "evaluation/pipelines/target_pipeline/blip/cedar_dataset.py",
        {
            "workload": "blip",
            "dataset_path": str(ROOT / "datasets/target_pipeline_bench/blip.jsonl"),
        },
        2000,
    ),
    "clip": (
        "evaluation/pipelines/target_pipeline/clip/cedar_dataset.py",
        {
            "workload": "clip",
            "dataset_path": str(ROOT / "datasets/target_pipeline_bench/clip.jsonl"),
        },
        2000,
    ),
    "dino": (
        "evaluation/pipelines/target_pipeline/dino/cedar_dataset.py",
        {
            "workload": "dino",
            "dataset_path": str(ROOT / "datasets/target_pipeline_bench/dino.jsonl"),
            "views": "2",
        },
        2000,
    ),
    "alpaca_cot": ("evaluation/pipelines/alpaca_cot/cedar_dataset.py", {}, 2000),
    "pile_hackernews": (
        "evaluation/pipelines/pile_hackernews/cedar_dataset.py",
        {},
        2000,
    ),
    "pile_pubmed_abstracts": (
        "evaluation/pipelines/target_pipeline/hub/pile_pubmed_abstracts/cedar_dataset.py",
        {},
        2000,
    ),
    "pile_uspto_backgrounds": (
        "evaluation/pipelines/target_pipeline/hub/pile_uspto_backgrounds/cedar_dataset.py",
        {},
        2000,
    ),
}


def load_module(path: str):
    relative = Path(path).with_suffix("")
    return importlib.import_module(".".join(relative.parts))


def build_feature(workload: str, profile: Path):
    """Feature object for one workload, built the way the harness builds it."""
    from evaluation.cedar_utils import CedarEvalSpec
    from evaluation.eval_cedar import import_module_from_path

    dataset_file, kwargs, samples = WORKLOADS[workload]
    module = import_module_from_path(str((ROOT / dataset_file).resolve()))
    getter = getattr(module, "get_target_dataset", None) or module.get_dataset
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
    return next(iter(dataset.features.values()))


def collect_plans(workload: str):
    """(planner, plan dict, measured throughput) for one workload."""
    found = {}
    for run in reversed(RUNS):
        results = run / "results"
        if not results.is_dir():
            continue
        for path in sorted(results.glob(f"{workload}__*.json")):
            planner = path.stem.split("__", 1)[1]
            try:
                entry = json.loads(path.read_text())["runs"][0]
            except Exception:  # noqa: BLE001
                continue
            perf = entry.get("perf_time_sec")
            samples = entry.get("num_samples")
            if not perf or not samples or perf <= 0:
                continue
            plans = entry.get("physical_plans_by_feature") or {}
            if not plans:
                continue
            plan = next(iter(plans.values()))
            plan = json.loads(json.dumps(plan))
            plan["graph"] = {int(k): v for k, v in plan["graph"].items()}
            plan["pipes"] = {int(k): v for k, v in plan["pipes"].items()}
            for pipe in plan["pipes"].values():
                pipe["variant"] = pipe.get("variant") or "INPROCESS"
                pipe.setdefault(
                    "variant_ctx", {"variant_type": pipe["variant"]}
                )
            records = WORKLOAD_RECORDS.get(workload)
            if DRAINED_RUN in str(path) and records:
                found[planner] = (plan, records / perf)
            else:
                found[planner] = (plan, samples / perf)
    return found


def spearman(measured, predicted):
    n = len(measured)
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

    a = ranks(measured)
    b = ranks(predicted)
    mean = (n + 1) / 2.0
    num = sum((a[i] - mean) * (b[i] - mean) for i in range(n))
    den = sum((v - mean) ** 2 for v in a)
    return num / den if den else float("nan")


def main() -> None:
    workloads = sys.argv[1:] or ["alpaca_cot"]
    for workload in workloads:
        plans = collect_plans(workload)
        if not plans:
            print(f"{workload}: no plans")
            continue
        profile = None
        for run in RUNS:
            candidate = run / "profiles" / f"{workload}_profile.yaml"
            if candidate.is_file():
                profile = candidate
                break
        if profile is None:
            print(f"{workload}: no profile")
            continue
        feature = build_feature(workload, profile)
        optimizer = DpOptimizer()
        feature.set_optimizer(optimizer)
        _, _, samples = WORKLOADS[workload]
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
                reorder_timeout_sec=1800.0,
            ),
        )
        names = sorted(plans)
        measured = [plans[name][1] for name in names]
        print(f"\n=== {workload}  ({len(names)} plans)")
        # Per-plan comparison first, under the environment's current basis.
        os.environ.setdefault("CEDAR_DP_STAGE_CONCURRENCY_RAY", "1")
        os.environ.setdefault("CEDAR_DP_STAGE_CONCURRENCY_SMP", "1")
        basis = os.environ.get("CEDAR_DP_STAGE_CURVE_COST", "0")
        print(f"basis: stage_curve={basis}")
        print(
            f"{'planner':<30}{'W':>4}{'measured':>10}{'predicted':>11}"
            f"{'ratio':>7}"
        )
        for name in names:
            plan_dict, throughput = plans[name]
            plan = PhysicalPlan.from_dict(plan_dict)
            workers = max(1, int(plan.n_local_workers or 1))
            score = optimizer.calculate_dp_objective_cost(plan=plan)
            predicted = 1000.0 * workers / score if score > 0 else float("inf")
            ratio = predicted / throughput if throughput else float("nan")
            print(
                f"{name:<30}{workers:>4}{throughput:>10.0f}"
                f"{predicted:>11.0f}{ratio:>7.2f}"
            )
        header = f"{'gamma':>8} {'rho':>6}  model best / measured best"
        print(header)
        best_gamma, best_rho = None, -2.0
        for gamma in GRID:
            os.environ["CEDAR_DP_STAGE_CONCURRENCY_RAY"] = str(gamma)
            os.environ["CEDAR_DP_STAGE_CONCURRENCY_SMP"] = str(gamma)
            scores = []
            for name in names:
                plan = PhysicalPlan.from_dict(plans[name][0])
                scores.append(optimizer.calculate_dp_objective_cost(plan=plan))
            rho = spearman(measured, scores)
            model_best = names[scores.index(min(scores))]
            measured_best = names[measured.index(max(measured))]
            mark = ""
            if model_best == measured_best:
                mark = "  <- match"
            print(
                f"{gamma:>8} {rho:>6.2f}  {model_best} / {measured_best}{mark}"
            )
            if rho > best_rho:
                best_gamma, best_rho = gamma, rho
        print(f"  best gamma={best_gamma} rho={best_rho:.2f}")


if __name__ == "__main__":
    main()
