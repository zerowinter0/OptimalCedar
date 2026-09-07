"""Isolated Cedar actions used by pilot and formal experiment drivers."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import yaml

from cedar.client import DataSet
from cedar.compose import OptimizerOptions, PhysicalPlan
from cedar.compose.optimizer import Optimizer
from evaluation.cedar_utils import CedarEvalSpec
from evaluation.motivation_multimodal.artifacts import (
    atomic_write_json,
    command_metadata,
    sha256_file,
)
from evaluation.pipelines.multimodal_running_example.cedar_dataset import (
    MultimodalRunningExampleFeature,
    Thresholds,
    get_dataset,
)
from cedar.sources import LocalLineSource


STAGED_OPTIMIZERS = (
    "staged_rfo",
    "staged_rof",
    "staged_fro",
    "staged_for",
    "staged_orf",
    "staged_ofr",
)
OPTIMIZER_SELECTORS = {
    **{name: 12 for name in STAGED_OPTIMIZERS},
    "joint": 13,
}


def _dataset_kwargs(
    dataset_path: Path,
    threshold_path: Path,
    image_root: Path,
) -> dict[str, str]:
    return {
        "dataset_path": str(dataset_path),
        "threshold_path": str(threshold_path),
        "image_root": str(image_root),
    }


def load_plan(path: str | Path) -> PhysicalPlan:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "physical_plan" not in payload:
        raise ValueError(f"Missing physical_plan in {path}")
    plan = PhysicalPlan.from_dict(payload["physical_plan"])
    if not plan.validate():
        raise ValueError(f"Invalid physical plan in {path}")
    return plan


def plan_backend_families(
    plan: PhysicalPlan,
    operator_ids: set[int] | None = None,
) -> list[str]:
    families = set()
    for pipe_id in plan.graph:
        desc = plan.pipe_descs[pipe_id]
        members = set(desc.fused_pipes or (pipe_id,))
        if operator_ids is not None and not members.intersection(operator_ids):
            continue
        variant = desc.variant_type.name
        if variant in {"RAY", "TF_RAY"}:
            family = "cuda-ray" if desc.execution_resource.value == "cuda" else "ray"
        elif variant == "SMP":
            family = "smp"
        elif variant != "INPROCESS" or desc.name != "LocalLineSourcePipe":
            family = "local"
        else:
            continue
        families.add(family)
    return sorted(families)


def profile(
    dataset_path: Path,
    threshold_path: Path,
    image_root: Path,
    output_path: Path,
    num_samples: int,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)
    spec = CedarEvalSpec(
        batch_size=1,
        num_total_samples=num_samples,
        num_epochs=1,
        kwargs=_dataset_kwargs(dataset_path, threshold_path, image_root),
        use_ray=True,
        profiled_stats=str(output_path),
        run_profiling=True,
        disable_optimizer=True,
        disable_controller=True,
        disable_prefetch=False,
        disable_offload=False,
        disable_parallelism=False,
        disable_reorder=False,
        disable_fusion=False,
        disable_caching=True,
    )
    started = time.perf_counter()
    try:
        get_dataset(spec)
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise
    if not output_path.is_file():
        raise RuntimeError("Cedar profiling returned without writing its profile")
    payload = yaml.safe_load(output_path.read_text(encoding="utf-8"))
    expected = payload.get("resource_config", {})
    if (
        expected.get("profile_local_workers") != 1
        or expected.get("ray_actors_per_stage") != 1
        or expected.get("smp_procs_per_stage") != 1
        or float(payload.get("profile_metadata", {}).get("stage_duration_sec", 0))
        != 10.0
    ):
        raise RuntimeError(f"Profile violates the fixed resource protocol: {expected}")
    return {
        "status": "success",
        "seconds": time.perf_counter() - started,
        "path": str(output_path),
        "sha256": sha256_file(output_path),
    }


def generate_plan(
    optimizer_name: str,
    dataset_path: Path,
    threshold_path: Path,
    image_root: Path,
    profile_path: Path,
    output_path: Path,
    num_samples: int,
) -> dict[str, Any]:
    selector = OPTIMIZER_SELECTORS[optimizer_name]
    stage_order = (
        optimizer_name.removeprefix("staged_")
        if optimizer_name in STAGED_OPTIMIZERS
        else None
    )
    generated = Path("/tmp/cedar_optimized_plan.yml")
    generated.unlink(missing_ok=True)
    spec = CedarEvalSpec(
        batch_size=1,
        num_total_samples=num_samples,
        num_epochs=1,
        kwargs=_dataset_kwargs(dataset_path, threshold_path, image_root),
        use_ray=True,
        profiled_stats=str(profile_path),
        run_profiling=False,
        disable_optimizer=False,
        disable_controller=True,
        disable_prefetch=False,
        disable_offload=False,
        disable_parallelism=True,
        disable_reorder=False,
        disable_fusion=False,
        disable_caching=True,
        use_my_optimizer=selector,
        generate_plan=True,
        reorder_timeout_sec=3600.0,
    )
    started = time.perf_counter()
    previous_stage_order = os.environ.get("CEDAR_STAGED_OPTIMIZATION_ORDER")
    try:
        if stage_order is not None:
            os.environ["CEDAR_STAGED_OPTIMIZATION_ORDER"] = stage_order
        get_dataset(spec)
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise
    finally:
        if previous_stage_order is None:
            os.environ.pop("CEDAR_STAGED_OPTIMIZATION_ORDER", None)
        else:
            os.environ["CEDAR_STAGED_OPTIMIZATION_ORDER"] = previous_stage_order
    if not generated.is_file():
        raise RuntimeError("Cedar optimizer returned without materializing a plan")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(generated, output_path)
    plan = load_plan(output_path)
    feature = _feature_for_plan_metadata(threshold_path, image_root, dataset_path)
    operator_ids_by_tag = {
        pipe.tag: pipe_id
        for pipe_id, pipe in feature.logical_pipes.items()
        if pipe.tag in {"normalize", "perplexity", "sharpness", "aesthetic", "clip", "blip"}
    }
    return {
        "status": "success",
        "optimizer": optimizer_name,
        "stage_order": stage_order,
        "seconds": time.perf_counter() - started,
        "path": str(output_path),
        "sha256": sha256_file(output_path),
        "backend_families": plan_backend_families(
            plan, set(operator_ids_by_tag.values())
        ),
        "operator_ids": operator_ids_by_tag,
        "n_local_workers": plan.n_local_workers,
        "parallelism_policy": "minimum_width",
    }


def _feature_for_plan_metadata(
    threshold_path: Path,
    image_root: Path,
    dataset_path: Path,
) -> MultimodalRunningExampleFeature:
    feature = MultimodalRunningExampleFeature(
        Thresholds.from_json(threshold_path),
        image_root=image_root,
    )
    feature.apply(LocalLineSource(str(dataset_path)))
    return feature


def score_plans_with_cedar(
    plan_paths: dict[str, Path],
    profile_path: Path,
    threshold_path: Path,
    image_root: Path,
    dataset_path: Path,
    num_samples: int,
) -> dict[str, Any]:
    """Replay materialized plans through Cedar's native cost model only."""

    profile_payload = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    feature = _feature_for_plan_metadata(
        threshold_path, image_root, dataset_path
    )
    optimizer = Optimizer()
    optimizer.init(feature.logical_pipes, feature.logical_adj_list)
    optimizer.profiled_stats = profile_payload
    optimizer.options = OptimizerOptions(
        enable_prefetch=True,
        available_local_cpus=1,
        enable_offload=True,
        enable_reorder=True,
        enable_local_parallelism=False,
        enable_fusion=True,
        enable_caching=False,
        num_samples=num_samples,
    )
    optimizer._validate_stats()
    optimizer._init_stats()
    costs = {}
    for name, path in sorted(plan_paths.items()):
        plan = load_plan(path)
        fused_blocks = [
            list(desc.fused_pipes)
            for desc in plan.pipe_descs.values()
            if desc.fused_pipes and len(desc.fused_pipes) > 1
        ]
        costs[name] = float(
            optimizer.calculate_cost(
                plan.graph,
                physical_specs=plan.pipe_descs,
                fused_pipes=fused_blocks or None,
                caching_on=False,
                plan=plan,
            )
        )
    return {
        "status": "success",
        "cost_model": "cedar.Optimizer.calculate_cost",
        "profile_sha256": sha256_file(profile_path),
        "costs": costs,
    }


def _consume(dataset: DataSet) -> tuple[list[str], float]:
    started = time.perf_counter()
    record_ids = []
    for record in dataset:
        if not isinstance(record, dict) or "record_id" not in record:
            raise ValueError(f"Unexpected pipeline output: {type(record)!r}")
        record_ids.append(str(record["record_id"]))
    return record_ids, time.perf_counter() - started


def execute_plan(
    plan_path: Path,
    dataset_path: Path,
    threshold_path: Path,
    image_root: Path,
    num_samples: int,
) -> dict[str, Any]:
    spec = CedarEvalSpec(
        batch_size=1,
        num_total_samples=num_samples,
        num_epochs=2,
        config=str(plan_path),
        kwargs=_dataset_kwargs(dataset_path, threshold_path, image_root),
        use_ray=True,
        disable_optimizer=True,
        disable_controller=True,
        disable_caching=True,
    )
    dataset = get_dataset(spec)
    try:
        warm_ids, warm_seconds = _consume(dataset)
        timed_ids, timed_seconds = _consume(dataset)
    finally:
        dataset.close()
    if set(warm_ids) != set(timed_ids):
        raise RuntimeError("Warm and timed epochs produced different record IDs")
    return {
        "status": "success",
        "seconds": timed_seconds,
        "warmup_seconds": warm_seconds,
        "output_count": len(timed_ids),
        "record_ids": sorted(timed_ids),
        "plan_sha256": sha256_file(plan_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("profile", "plan", "score", "execute"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--threshold", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--optimizer", choices=tuple(OPTIMIZER_SELECTORS))
    parser.add_argument(
        "--named-plan",
        action="append",
        default=[],
        metavar="NAME=PATH",
    )
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "profile":
        if args.profile is None:
            parser.error("profile action requires --profile")
        result = profile(
            args.dataset,
            args.threshold,
            args.image_root,
            args.profile,
            args.num_samples,
        )
    elif args.action == "plan":
        if args.profile is None or args.plan is None or args.optimizer is None:
            parser.error("plan action requires --profile, --plan, and --optimizer")
        result = generate_plan(
            args.optimizer,
            args.dataset,
            args.threshold,
            args.image_root,
            args.profile,
            args.plan,
            args.num_samples,
        )
    elif args.action == "score":
        if args.profile is None or not args.named_plan:
            parser.error("score action requires --profile and --named-plan")
        named_plans = {}
        for value in args.named_plan:
            if "=" not in value:
                parser.error("--named-plan must use NAME=PATH")
            name, raw_path = value.split("=", 1)
            if not name or name in named_plans:
                parser.error("--named-plan names must be nonempty and unique")
            named_plans[name] = Path(raw_path)
        result = score_plans_with_cedar(
            named_plans,
            args.profile,
            args.threshold,
            args.image_root,
            args.dataset,
            args.num_samples,
        )
    else:
        if args.plan is None:
            parser.error("execute action requires --plan")
        result = execute_plan(
            args.plan,
            args.dataset,
            args.threshold,
            args.image_root,
            args.num_samples,
        )
    atomic_write_json(
        args.result,
        {**result, "metadata": command_metadata()},
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
