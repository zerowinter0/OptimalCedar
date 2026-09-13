"""Isolated Cedar actions used by pilot and formal experiment drivers."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import yaml

from cedar.client import DataSet
from cedar.compose import OptimizerOptions, PhysicalPlan
from cedar.compose.feature import apply_profile_matched_resources
from cedar.compose.optimizer import Optimizer, PipeDesc
from cedar.compose.constants import FUSED_PIPE_NAME
from cedar.pipes import PipeExecutionResource, PipeVariantType
from cedar.compose.sequential_exhaustive_optimizer import (
    SingleWorkerCudaDpOptimizer,
)
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
    "baseline": 0,
    "cedar": 0,
    "pico": 14,
}
REPO_ROOT = Path(__file__).resolve().parents[2]


class SharedParallelismCudaCostModel(SingleWorkerCudaDpOptimizer):
    """Replay width-optimized plans over the single-worker CUDA search space."""

    joint_actor_allocation = True


def _stage_ray_image_package(
    dataset_path: Path,
    image_root: Path,
    package_root: Path | None = None,
) -> Path:
    """Materialize exactly the fixture images needed by remote Ray actors."""

    dataset_path = dataset_path.resolve()
    image_root = image_root.resolve()
    if package_root is None:
        package_root = (
            REPO_ROOT / "outputs/motivation_multimodal/ray_data_packages"
        )
    package = (
        package_root
        / sha256_file(dataset_path)[:16]
        / "pico_multimodal_data"
    )
    package.mkdir(parents=True, exist_ok=True)
    marker = package / "__init__.py"
    if not marker.exists():
        marker.write_text(
            '"""Images packaged for one immutable multimodal fixture."""\n',
            encoding="utf-8",
        )

    relative_paths: set[Path] = set()
    with dataset_path.open("r", encoding="utf-8") as source_file:
        for line_number, line in enumerate(source_file, start=1):
            record = json.loads(line)
            relative = Path(str(record["image_path"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(
                    f"Unsafe image_path at line {line_number}: {relative}"
                )
            relative_paths.add(relative)

    for relative in sorted(relative_paths):
        source = (image_root / relative).resolve()
        if not source.is_relative_to(image_root) or not source.is_file():
            raise FileNotFoundError(f"Fixture image does not exist: {source}")
        destination = package / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if destination.stat().st_size != source.stat().st_size:
                raise RuntimeError(
                    f"Stale Ray image package entry: {destination}"
                )
            continue
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
    return package


def _ray_runtime_env(
    dataset_path: Path,
    image_root: Path,
    package_root: Path | None = None,
) -> dict[str, list[str]]:
    """Distribute this checkout's actor code and exact fixture images."""

    os.environ["CEDAR_RAY_PY_MODULE_ROOT"] = str(REPO_ROOT)
    image_package = _stage_ray_image_package(
        dataset_path, image_root, package_root
    )
    return {
        "py_modules": [
            str(REPO_ROOT / "cedar"),
            str(REPO_ROOT / "pico_multimodal"),
            str(image_package),
        ]
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


def save_plan(plan: PhysicalPlan, path: str | Path) -> None:
    if not plan.validate():
        raise ValueError("Refusing to save an invalid physical plan")
    payload = {"physical_plan": plan.to_dict()}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        yaml.safe_dump(payload, sort_keys=True),
        encoding="utf-8",
    )


def apply_shared_parallelism(
    plan: PhysicalPlan,
    profile_path: Path,
    num_samples: int,
) -> dict[str, Any]:
    """Allocate every fixed plan structure with the same resource policy."""

    profile_payload = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    cpu_budget_raw = os.environ.get("CEDAR_PROFILE_MATCH_CPU_BUDGET")
    if cpu_budget_raw is None:
        raise RuntimeError(
            "Shared parallelism allocation requires "
            "CEDAR_PROFILE_MATCH_CPU_BUDGET"
        )
    fixed_workers_raw = os.environ.get(
        "CEDAR_PROFILE_MATCH_FIXED_LOCAL_WORKERS"
    )
    signature = apply_profile_matched_resources(
        plan,
        profile_payload,
        int(cpu_budget_raw),
        int(fixed_workers_raw) if fixed_workers_raw is not None else None,
        preserve_optimizer_widths=False,
        force_minimum_widths=False,
        num_samples=num_samples,
    )
    if signature["allocation_policy"] != "separate_pool_equal_share":
        raise RuntimeError(
            "Unexpected shared parallelism policy: "
            f"{signature['allocation_policy']}"
        )
    return signature


def materialize_manual_plan(
    baseline_path: Path,
    dp_path: Path,
    profile_path: Path,
    threshold_path: Path,
    image_root: Path,
    dataset_path: Path,
    output_path: Path,
    num_samples: int,
) -> dict[str, Any]:
    """Materialize the predeclared partially offloaded PICO alternative."""

    baseline = load_plan(baseline_path)
    dp_plan = load_plan(dp_path)
    feature = _feature_for_plan_metadata(
        threshold_path, image_root, dataset_path
    )
    ids = {pipe.tag: pipe_id for pipe_id, pipe in feature.logical_pipes.items()}
    required = {"parse", "normalize", "perplexity", "safety", "aesthetic", "clip", "blip"}
    if not required.issubset(ids):
        raise RuntimeError(f"Manual plan is missing logical tags: {required - set(ids)}")

    active_baseline = set(baseline.graph)
    source_ids = [
        pipe_id
        for pipe_id in active_baseline
        if baseline.pipe_descs[pipe_id].name in {"LocalLinePipe", "LocalLineSourcePipe"}
    ]
    if len(source_ids) != 1:
        raise RuntimeError(f"Expected one source in baseline plan, got {source_ids}")
    prefetch_ids = [
        pipe_id
        for pipe_id in dp_plan.graph
        if dp_plan.pipe_descs[pipe_id].name == "PrefetcherPipe"
    ]
    ray_templates = [
        desc
        for pipe_id, desc in dp_plan.pipe_descs.items()
        if pipe_id in dp_plan.graph
        and desc.variant_type in (PipeVariantType.RAY, PipeVariantType.TF_RAY)
    ]
    if len(prefetch_ids) != 1 or not ray_templates:
        raise RuntimeError("DP plan lacks a prefetch node or Ray context template")

    descs = copy.deepcopy(baseline.pipe_descs)
    prefetch_id = max(descs) + 1
    descs[prefetch_id] = copy.deepcopy(dp_plan.pipe_descs[prefetch_ids[0]])
    # Keep PICO's logical order but omit its CPU and GPU fusion decisions. The
    # shared allocator divides each resource pool across the resulting stages.
    gpu_context = copy.deepcopy(ray_templates[0].variant_ctx)
    gpu_context.n_actors = 1
    gpu_context.num_gpus = 1.0
    cpu_context = copy.deepcopy(ray_templates[0].variant_ctx)
    cpu_context.n_actors = 1
    cpu_context.num_gpus = 0.0
    normalize_id = ids["normalize"]
    descs[normalize_id].variant_type = PipeVariantType.RAY
    descs[normalize_id].variant_ctx = copy.deepcopy(cpu_context)
    descs[normalize_id].fused_pipes = None
    perplexity_id = ids["perplexity"]
    descs[perplexity_id].variant_type = PipeVariantType.RAY
    descs[perplexity_id].variant_ctx = copy.deepcopy(cpu_context)
    descs[perplexity_id].fused_pipes = None

    clip_id = ids["clip"]
    blip_id = ids["blip"]
    fused_id = max(prefetch_id, *descs) + 1
    descs[fused_id] = PipeDesc(
        name=FUSED_PIPE_NAME,
        variant_type=PipeVariantType.RAY,
        variant_ctx=copy.deepcopy(gpu_context),
        fused_pipes=[clip_id, blip_id],
        execution_resource=PipeExecutionResource.CUDA,
    )
    order = [
        source_ids[0],
        ids["parse"],
        ids["aesthetic"],
        ids["safety"],
        normalize_id,
        perplexity_id,
        fused_id,
        prefetch_id,
    ]
    graph = {
        pipe_id: ({order[index + 1]} if index + 1 < len(order) else set())
        for index, pipe_id in enumerate(order)
    }
    plan = PhysicalPlan(graph=graph, pipe_descs=descs, n_local_workers=1)
    resource_signature = apply_shared_parallelism(
        plan, profile_path, num_samples
    )
    save_plan(plan, output_path)
    return {
        "status": "success",
        "optimizer": "manual",
        "path": str(output_path),
        "sha256": sha256_file(output_path),
        "stage_widths": plan_stage_widths(plan),
        "backend_families": plan_backend_families(
            plan,
            {ids[tag] for tag in required if tag != "parse"},
        ),
        "operator_ids": {tag: ids[tag] for tag in required if tag != "parse"},
        "n_local_workers": 1,
        "parallelism_policy": resource_signature["allocation_policy"],
        "resource_signature": resource_signature,
        "declared_order": ["aesthetic", "safety", "normalize", "perplexity", "clip", "blip"],
        "declared_fusions": [["clip", "blip"]],
        "declared_placements": {
            "normalize": "ray-cpu",
            "perplexity": "ray-cpu",
            "clip+blip": "ray-gpu",
        },
    }


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


def plan_stage_widths(plan: PhysicalPlan) -> dict[str, int]:
    """Return widths of active parallel stages for protocol validation."""

    widths = {}
    for pipe_id in plan.graph:
        desc = plan.pipe_descs[pipe_id]
        if desc.variant_type.name in {"RAY", "TF_RAY"}:
            widths[str(pipe_id)] = int(desc.variant_ctx.n_actors)
        elif desc.variant_type.name == "SMP":
            widths[str(pipe_id)] = int(desc.variant_ctx.n_procs)
    return widths


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
        ray_runtime_env=_ray_runtime_env(dataset_path, image_root),
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
    ray_width = int(os.environ.get("CEDAR_PROFILE_RAY_ACTORS", "8"))
    smp_width = int(os.environ.get("CEDAR_PROFILE_SMP_PROCS", "8"))
    if (
        expected.get("profile_local_workers") != 1
        or expected.get("ray_actors_per_stage") != ray_width
        or expected.get("smp_procs_per_stage") != smp_width
        or float(payload.get("profile_metadata", {}).get("stage_duration_sec", 0))
        != 10.0
    ):
        raise RuntimeError(
            "Profile violates the declared resource protocol: "
            f"expected Ray={ray_width}, SMP={smp_width}; observed={expected}"
        )
    selectivities = payload.get("baseline", {}).get("selectivities")
    if os.environ.get("CEDAR_PROFILE_FILTER_SELECTIVITY") == "1" and not isinstance(
        selectivities, dict
    ):
        raise RuntimeError(
            "Profile protocol requested filter selectivity, but the profile "
            "contains no baseline.selectivities map"
        )
    return {
        "status": "success",
        "seconds": time.perf_counter() - started,
        "path": str(output_path),
        "sha256": sha256_file(output_path),
        "resource_config": expected,
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
    is_baseline = optimizer_name == "baseline"
    generated = Path("/tmp/cedar_optimized_plan.yml")
    generated.unlink(missing_ok=True)
    spec = CedarEvalSpec(
        batch_size=1,
        num_total_samples=num_samples,
        num_epochs=1,
        kwargs=_dataset_kwargs(dataset_path, threshold_path, image_root),
        use_ray=True,
        ray_runtime_env=_ray_runtime_env(dataset_path, image_root),
        profiled_stats=str(profile_path),
        run_profiling=False,
        disable_optimizer=False,
        disable_controller=True,
        disable_prefetch=False,
        disable_offload=is_baseline,
        disable_parallelism=True,
        disable_reorder=is_baseline,
        disable_fusion=is_baseline,
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
    resource_signature = apply_shared_parallelism(
        plan, profile_path, num_samples
    )
    save_plan(plan, output_path)
    stage_widths = plan_stage_widths(plan)
    feature = _feature_for_plan_metadata(threshold_path, image_root, dataset_path)
    operator_ids_by_tag = {
        pipe.tag: pipe_id
        for pipe_id, pipe in feature.logical_pipes.items()
        if pipe.tag in {"normalize", "perplexity", "safety", "aesthetic", "clip", "blip"}
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
        "stage_widths": stage_widths,
        "parallelism_policy": resource_signature["allocation_policy"],
        "resource_signature": resource_signature,
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
    shared_cpu_budget = int(
        os.environ.get("CEDAR_PROFILE_MATCH_CPU_BUDGET", "1")
    )
    feature = _feature_for_plan_metadata(
        threshold_path, image_root, dataset_path
    )
    optimizer = Optimizer()
    optimizer.init(feature.logical_pipes, feature.logical_adj_list)
    optimizer.profiled_stats = profile_payload
    optimizer.options = OptimizerOptions(
        enable_prefetch=True,
        available_local_cpus=shared_cpu_budget,
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
    pico_optimizer = SharedParallelismCudaCostModel()
    pico_optimizer.init(feature.logical_pipes, feature.logical_adj_list)
    pico_optimizer.profiled_stats = profile_payload
    pico_optimizer.options = OptimizerOptions(
        enable_prefetch=True,
        available_local_cpus=shared_cpu_budget,
        enable_offload=True,
        enable_reorder=True,
        enable_local_parallelism=False,
        enable_fusion=True,
        enable_caching=False,
        num_samples=num_samples,
    )
    pico_optimizer._validate_stats()
    pico_optimizer._init_stats()
    inner_ops = pico_optimizer._get_linear_inner_ops()
    if inner_ops is None or not inner_ops:
        raise RuntimeError("Could not recover linear operators for PICO scoring")
    pico_optimizer._prepare_dp_metadata(inner_ops)
    pico_costs = {
        name: float(
            pico_optimizer.calculate_dp_objective_cost(
                plan=load_plan(path),
                inner_ops=inner_ops,
            )
        )
        for name, path in sorted(plan_paths.items())
    }
    return {
        "status": "success",
        "cost_model": "cedar.Optimizer.calculate_cost",
        "pico_cost_model": "DpOptimizer.calculate_dp_objective_cost",
        "profile_sha256": sha256_file(profile_path),
        "costs": costs,
        "pico_costs": pico_costs,
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
        ray_runtime_env=_ray_runtime_env(dataset_path, image_root),
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
    parser.add_argument("action", choices=("profile", "plan", "manual", "score", "execute"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--threshold", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--baseline-plan", type=Path)
    parser.add_argument("--dp-plan", type=Path)
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
    elif args.action == "manual":
        if (
            args.plan is None
            or args.baseline_plan is None
            or args.dp_plan is None
            or args.profile is None
        ):
            parser.error(
                "manual action requires --plan, --baseline-plan, --dp-plan, "
                "and --profile"
            )
        result = materialize_manual_plan(
            args.baseline_plan,
            args.dp_plan,
            args.profile,
            args.threshold,
            args.image_root,
            args.dataset,
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
