#!/usr/bin/env python3
"""Score formal optimizer plans with Cedar and DP cost models.

This is a read-only re-analysis of the canonical W=8 archive.  Physical plans
are canonicalized and deduplicated within each workload; execution time is
pooled across every three-round optimizer record that produced the same plan.
No data pipeline is executed.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from scipy.stats import spearmanr


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cedar.compose.optimizer import PhysicalPlan
from evaluation.chapter6_experiments.cost_model_accuracy_metrics import (
    pairwise_qerrors,
    qerror_summary,
)
from evaluation.cedar_utils import CedarEvalSpec
from evaluation.eval_cedar import import_module_from_path


FORMAL_ROOT = REPO_ROOT / "evaluation/chapter6_experiments/formal_results"
ARCHIVE = FORMAL_ROOT / "paper_artifacts" / "optimizer"
DATA_ROOT = ARCHIVE / "data"
PROFILE_ROOT = ARCHIVE / "profiles"


@dataclass(frozen=True)
class Workload:
    label: str
    dataset_file: str
    samples: int
    kwargs: dict[str, str]
    cache: bool = False


WORKLOADS = {
    "coco": Workload("COCO", "evaluation/pipelines/coco/cedar_dataset.py", 50_000, {}),
    "commonvoice": Workload(
        "CV",
        "evaluation/pipelines/commonvoice/cedar_dataset.py",
        160_000,
        {"max_samples": "10000"},
    ),
    "commonvoice_cache": Workload(
        "CV [Cache]",
        "evaluation/pipelines/commonvoice/cedar_cache_dataset.py",
        160_000,
        {"max_samples": "10000"},
        True,
    ),
    "llava_pretrain": Workload(
        "LLaVA",
        "evaluation/pipelines/llava_pretrain/cedar_dataset.py",
        5_000,
        {
            "dataset_path": "evaluation/datasets/llava_pretrain/blip_laion_cc_sbu_20000_dj_fmt_only_caption.jsonl",
            "image_root": "evaluation/datasets/llava_pretrain",
        },
    ),
    "redpajama_c4": Workload(
        "RP-C4",
        "evaluation/pipelines/redpajama_c4/cedar_dataset.py",
        20_000,
        {"dataset_path": "datasets/redpajama_c4/redpajama-c4-raw-829916.jsonl"},
    ),
    "simclrv2": Workload(
        "SimCLR", "evaluation/pipelines/simclrv2/cedar_dataset.py", 9_469, {}
    ),
    "simclrv2_cache": Workload(
        "SimCLR [Cache]",
        "evaluation/pipelines/simclrv2/cedar_cache_dataset.py",
        9_469,
        {},
        True,
    ),
    "stackexchange": Workload(
        "StackEx",
        "evaluation/pipelines/stackexchange/cedar_dataset.py",
        10_000,
        {"dataset_path": "datasets/stackexchange/redpajama-stackexchange-35000.jsonl"},
    ),
    "wikitext103": Workload(
        "Wiki", "evaluation/pipelines/wikitext103/cedar_dataset.py", 100_000, {"max_samples": "100000"}
    ),
    "wikitext103_cache": Workload(
        "Wiki [Cache]",
        "evaluation/pipelines/wikitext103/cedar_cache_dataset.py",
        100_000,
        {"max_samples": "100000"},
        True,
    ),
    "pile_europarl": Workload(
        "EuroParl",
        "evaluation/pipelines/pile_europarl/cedar_dataset.py",
        2_500,
        {"dataset_path": "datasets/pile_europarl/pile-europarl-raw.jsonl"},
    ),
}


def canonical_plan(path: Path) -> tuple[dict[str, Any], str]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    physical = payload["physical_plan"]
    encoded = json.dumps(physical, sort_keys=True, separators=(",", ":"))
    return physical, hashlib.sha256(encoded.encode()).hexdigest()


def plan_path(workload: str, optimizer: str) -> Path:
    replacement = DATA_ROOT / "dp_no_wall_clock" / workload / "plans" / f"{optimizer}.yaml"
    if optimizer == "dp_optimizer" and replacement.exists():
        return replacement
    if workload in ("coco", "commonvoice", "commonvoice_cache"):
        return DATA_ROOT / "enlarged_core" / "plans" / workload / f"{optimizer}.yaml"
    if workload == "pile_europarl":
        return DATA_ROOT / "europarl_formal" / workload / "plans" / f"{optimizer}.yaml"
    return DATA_ROOT / "standard_core" / "workloads" / workload / "plans" / f"{optimizer}.yaml"


def plan_optimizers(workload: str) -> list[str]:
    if workload in ("coco", "commonvoice", "commonvoice_cache"):
        directory = DATA_ROOT / "enlarged_core" / "plans" / workload
    elif workload == "pile_europarl":
        directory = DATA_ROOT / "europarl_formal" / workload / "plans"
    else:
        directory = DATA_ROOT / "standard_core" / "workloads" / workload / "plans"
    optimizers = {path.stem for path in directory.glob("*.yaml")}
    replacement = DATA_ROOT / "dp_no_wall_clock" / workload / "plans" / "dp_optimizer.yaml"
    if replacement.exists():
        optimizers.add("dp_optimizer")
    return sorted(optimizers)


def result_paths(workload: str, optimizer: str) -> list[Path]:
    replacement = DATA_ROOT / "dp_no_wall_clock" / workload / "results"
    if optimizer == "dp_optimizer" and replacement.exists():
        return sorted(replacement.glob("round*__dp_optimizer.json"))
    if workload in ("coco", "commonvoice", "commonvoice_cache"):
        directory = DATA_ROOT / "enlarged_core" / "cedar" / workload / optimizer
        return sorted(directory.glob("attempt1__round*.json"))
    if workload == "pile_europarl":
        directory = DATA_ROOT / "europarl_formal" / workload / "results"
    else:
        directory = DATA_ROOT / "standard_core" / "workloads" / workload / "results"
    return sorted(directory.glob(f"round*__{optimizer}.json"))


def read_execution(path: Path) -> tuple[float, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    times = [float(value) for value in payload.get("epoch_run_times", [])]
    counts = [int(value) for value in payload.get("epoch_num_samples", [])]
    if not times or not counts or sum(counts) <= 0:
        raise ValueError(f"Incomplete execution result: {path}")
    return sum(times), sum(counts)


def collect_candidates(workload: str) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for optimizer in plan_optimizers(workload):
        path = plan_path(workload, optimizer)
        physical, digest = canonical_plan(path)
        executions = [read_execution(result) for result in result_paths(workload, optimizer)]
        if len(executions) != 3:
            raise ValueError(f"{workload}/{optimizer} does not have three executions")
        output_counts = {count for _, count in executions}
        if len(output_counts) != 1:
            raise ValueError(f"{workload}/{optimizer} output counts differ")
        candidate = grouped.setdefault(
            digest,
            {
                "sha256": digest,
                "physical_plan": physical,
                "optimizers": [],
                "source_plans": [],
                "execution_times_sec": [],
                "output_counts": [],
            },
        )
        candidate["optimizers"].append(optimizer)
        candidate["source_plans"].append(str(path.relative_to(REPO_ROOT)))
        candidate["execution_times_sec"].extend(value for value, _ in executions)
        candidate["output_counts"].extend(count for _, count in executions)
    for index, candidate in enumerate(grouped.values(), start=1):
        candidate["candidate_id"] = f"p{index}"
        candidate["mean_runtime_sec"] = statistics.mean(candidate["execution_times_sec"])
        candidate["runtime_sd_sec"] = (
            statistics.stdev(candidate["execution_times_sec"])
            if len(candidate["execution_times_sec"]) > 1
            else 0.0
        )
    return list(grouped.values())


def load_optimizer(workload_name: str):
    workload = WORKLOADS[workload_name]
    profile = PROFILE_ROOT / f"{workload_name}.yaml"
    spec = CedarEvalSpec(
        batch_size=1,
        num_total_samples=workload.samples,
        num_epochs=1,
        kwargs=workload.kwargs,
        # Scoring only needs the logical graph and frozen profile.  Keeping the
        # dataset optimizer disabled avoids materializing or executing a plan.
        use_ray=False,
        profiled_stats=str(profile),
        disable_optimizer=True,
        disable_controller=True,
        disable_caching=not workload.cache,
        use_my_optimizer=2,
    )
    module = import_module_from_path(workload.dataset_file)
    dataset = module.get_dataset(spec)
    feature = next(iter(dataset.features.values()))
    optimizer = feature.optimizer
    optimizer.profiled_stats = yaml.safe_load(profile.read_text(encoding="utf-8"))
    optimizer.options = dataset.optimizer_options
    optimizer._init_stats()
    inner_ops = optimizer._get_linear_inner_ops()
    if inner_ops is None or not inner_ops:
        raise RuntimeError("Could not recover linear DP operators")
    dp_setup_error = None
    try:
        optimizer._prepare_dp_metadata(inner_ops)
    except Exception as exc:
        dp_setup_error = f"{type(exc).__name__}: {exc}"
    return dataset, optimizer, dp_setup_error


def score_candidates(workload_name: str, candidates: list[dict[str, Any]]) -> None:
    workload = WORKLOADS[workload_name]
    dataset, optimizer, dp_setup_error = load_optimizer(workload_name)
    try:
        for candidate in candidates:
            plan = PhysicalPlan.from_dict(candidate.pop("physical_plan"))
            fused = [
                list(desc.fused_pipes)
                for desc in plan.pipe_descs.values()
                if desc.fused_pipes and len(desc.fused_pipes) > 1
            ]
            try:
                cedar_cost = optimizer.calculate_cost(
                    plan.graph,
                    physical_specs=plan.pipe_descs,
                    fused_pipes=fused or None,
                    caching_on=workload.cache,
                    plan=plan,
                )
                candidate["cedar_cost"] = float(cedar_cost * 8)
            except Exception as exc:  # Preserve coverage rather than aborting.
                candidate["cedar_error"] = f"{type(exc).__name__}: {exc}"
            if dp_setup_error is not None:
                candidate["dp_objective_error"] = dp_setup_error
            else:
                try:
                    candidate["dp_objective_cost"] = float(
                        optimizer.calculate_dp_objective_cost(plan=plan)
                    )
                except Exception as exc:  # Preserve coverage rather than aborting.
                    candidate["dp_objective_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        del dataset
        gc.collect()
        try:
            import ray

            if ray.is_initialized():
                ray.shutdown()
        except Exception:
            pass


def model_metrics(candidates: list[dict[str, Any]], field: str) -> dict[str, Any]:
    scored = [item for item in candidates if field in item]
    if len(scored) < 2:
        return {"coverage": len(scored) / len(candidates), "num_scored": len(scored)}
    costs = [float(item[field]) for item in scored]
    times = [float(item["mean_runtime_sec"]) for item in scored]
    selected = min(scored, key=lambda item: item[field])
    fastest = min(scored, key=lambda item: item["mean_runtime_sec"])
    rho = None
    if len(scored) >= 3 and len(set(costs)) > 1 and len(set(times)) > 1:
        rho = float(spearmanr(costs, times).statistic)
    qerrors, ordered_correct, ordered_total = pairwise_qerrors(candidates, field)
    return {
        "coverage": len(scored) / len(candidates),
        "num_scored": len(scored),
        "spearman_rho": rho,
        "selected_candidate": selected["candidate_id"],
        "fastest_candidate": fastest["candidate_id"],
        "top1_correct": selected["candidate_id"] == fastest["candidate_id"],
        "selection_regret": selected["mean_runtime_sec"] / fastest["mean_runtime_sec"] - 1.0,
        "pairwise_qerror": qerror_summary(qerrors),
        "pairwise_order_accuracy": (
            ordered_correct / ordered_total if ordered_total else None
        ),
    }


def aggregate_model_metrics(
    workloads: dict[str, Any], field: str
) -> dict[str, Any]:
    all_qerrors: list[float] = []
    correct = 0
    total = 0
    workload_metrics = []
    for workload in workloads.values():
        candidates = workload["candidates"]
        values, workload_correct, workload_total = pairwise_qerrors(
            candidates, field
        )
        all_qerrors.extend(values)
        correct += workload_correct
        total += workload_total
        model_key = {
            "cedar_cost": "cedar",
            "dp_objective_cost": "dp",
            "reference_dp_cost": "reference_dp",
        }[field]
        metrics = workload["models"].get(model_key)
        if metrics and metrics.get("num_scored", 0) >= 2:
            workload_metrics.append(metrics)
    rhos = [
        item["spearman_rho"]
        for item in workload_metrics
        if item.get("spearman_rho") is not None
    ]
    return {
        "num_workloads": len(workload_metrics),
        "pairwise_qerror": qerror_summary(all_qerrors),
        "pairwise_order_accuracy": correct / total if total else None,
        "macro_spearman_rho": statistics.mean(rhos) if rhos else None,
        "top1_correct": sum(bool(item.get("top1_correct")) for item in workload_metrics),
        "mean_selection_regret": statistics.mean(
            item["selection_regret"] for item in workload_metrics
        ) if workload_metrics else None,
    }


def shared_coverage_summary(
    workloads: dict[str, Any], fields: tuple[str, ...]
) -> dict[str, Any]:
    """Evaluate every model on exactly the plans scorable by all models."""
    model_keys = {
        "cedar_cost": "cedar",
        "dp_objective_cost": "dp",
        "reference_dp_cost": "reference_dp",
    }
    shared: dict[str, Any] = {}
    for name, workload in workloads.items():
        candidates = [
            candidate
            for candidate in workload["candidates"]
            if all(field in candidate for field in fields)
        ]
        if len(candidates) < 2:
            continue
        shared[name] = {
            "candidates": candidates,
            "models": {
                model_keys[field]: model_metrics(candidates, field)
                for field in fields
            },
        }
    return {
        "protocol": "same workload-plan pairs scorable by every listed model",
        "models": {
            model_keys[field]: aggregate_model_metrics(shared, field)
            for field in fields
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=FORMAL_ROOT / "paper_artifacts" / "cost_model" / "analysis.json",
    )
    parser.add_argument("--workloads", nargs="+", choices=WORKLOADS, default=list(WORKLOADS))
    parser.add_argument(
        "--reference-analysis",
        type=Path,
        help=(
            "Optional older analysis.json. Matching plan costs are imported "
            "as a frozen reference model for an offline before/after comparison."
        ),
    )
    args = parser.parse_args()

    reference_costs: dict[tuple[str, str], float] = {}
    if args.reference_analysis is not None:
        reference = json.loads(args.reference_analysis.read_text())
        for workload_name, workload in reference.get("workloads", {}).items():
            for candidate in workload.get("candidates", []):
                if "sha256" in candidate and "dp_objective_cost" in candidate:
                    reference_costs[(workload_name, candidate["sha256"])] = float(
                        candidate["dp_objective_cost"]
                    )

    result: dict[str, Any] = {
        "protocol": {
            "archive": str(ARCHIVE.relative_to(REPO_ROOT)),
            "local_workers": 8,
            "cpu_budget": 64,
            "deduplication": "canonical physical-plan SHA-256 within workload",
            "runtime": "mean of all three-round records producing the canonical plan",
            "execution_rerun": False,
            "accuracy_metric": (
                "pairwise Q-error of predicted versus measured runtime ratios; "
                "1.0 is exact and workload-wide scale cancels"
            ),
            "reference_analysis": (
                str(args.reference_analysis) if args.reference_analysis else None
            ),
        },
        "workloads": {},
    }
    for workload_name in args.workloads:
        print(f"Scoring {workload_name}...", flush=True)
        candidates = collect_candidates(workload_name)
        score_candidates(workload_name, candidates)
        for candidate in candidates:
            reference_cost = reference_costs.get(
                (workload_name, candidate["sha256"])
            )
            if reference_cost is not None:
                candidate["reference_dp_cost"] = reference_cost
        result["workloads"][workload_name] = {
            "label": WORKLOADS[workload_name].label,
            "cache": WORKLOADS[workload_name].cache,
            "num_unique_candidates": len(candidates),
            "num_execution_measurements": sum(
                len(item["execution_times_sec"]) for item in candidates
            ),
            "models": {
                "cedar": model_metrics(candidates, "cedar_cost"),
                "dp": model_metrics(candidates, "dp_objective_cost"),
                "reference_dp": model_metrics(candidates, "reference_dp_cost"),
            },
            "candidates": candidates,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    result["summary"] = {
        "cedar": aggregate_model_metrics(result["workloads"], "cedar_cost"),
        "dp": aggregate_model_metrics(result["workloads"], "dp_objective_cost"),
        "reference_dp": aggregate_model_metrics(
            result["workloads"], "reference_dp_cost"
        ),
    }
    result["shared_coverage_summary"] = shared_coverage_summary(
        result["workloads"],
        ("cedar_cost", "reference_dp_cost", "dp_objective_cost"),
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
