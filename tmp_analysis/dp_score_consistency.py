"""Compare the DP's search-time objective with the re-scored objective.

    python tmp_analysis/dp_score_consistency.py <dataset_file> <kwargs> <profile> <plan.json>

For every planner's materialized plan in ``plan.json`` it prints
``search-style`` and ``re-scored`` objective values for the same plan, plus the
per-block contributions, so a mismatch can be attributed to a specific term.
"""

import importlib
import json
import logging
import os
import sys

import yaml

sys.path.insert(0, "/workspace/OptimalCedar")

from evaluation.cedar_utils import CedarEvalSpec  # noqa: E402

logging.basicConfig(level=logging.ERROR)


def build(dataset_file, kwargs_raw, profile):
    module = importlib.import_module(
        dataset_file.replace("/", ".").removesuffix(".py")
    )
    fields = {}
    for token in kwargs_raw.split(","):
        if token.strip():
            key, _, value = token.partition("=")
            fields[key.strip()] = value.strip()
    spec = CedarEvalSpec(1, None, 1, kwargs=fields, profiled_stats=profile)
    spec.use_my_optimizer = 2
    spec.disable_controller = True
    spec.disable_caching = True
    os.environ.setdefault("CEDAR_MATCH_PROFILE_RESOURCES", "1")
    os.environ.setdefault("CEDAR_PROFILE_MATCH_CPU_BUDGET", "64")
    os.environ.setdefault("CEDAR_PROFILE_MATCH_RAY_CPU_BUDGET", "64")
    dataset = module.get_dataset(spec)
    feature = next(iter(dataset.features.values()))
    optimizer = feature.optimizer
    if not getattr(optimizer, "profiled_stats", None):
        optimizer.profiled_stats = yaml.safe_load(open(profile))
        optimizer.options = dataset.optimizer_options
        optimizer._init_stats()
    inner = optimizer._get_linear_inner_ops()
    optimizer._prepare_dp_metadata(inner)
    return dataset, feature, optimizer, inner


def main() -> int:
    dataset_file, kwargs_raw, profile, plan_json = sys.argv[1:5]
    dataset, feature, optimizer, inner = build(
        dataset_file, kwargs_raw, profile
    )
    from cedar.compose.optimizer import PhysicalPlan

    payload = json.loads(open(plan_json).read())
    for run in payload.get("runs", []):
        plans = run.get("physical_plans_by_feature") or {}
        if not plans:
            print(f"== {run['optimizer']}: no plan recorded")
            continue
        plan = PhysicalPlan.from_dict(next(iter(plans.values())))
        optimizer.configure_lane_exposure()
        try:
            rescored = optimizer.calculate_dp_objective_cost(plan=plan)
        except Exception as exc:  # noqa: BLE001
            print(f"== {run['optimizer']}: rescoring failed: {exc}")
            continue
        block_specs = optimizer._dp_blocks_from_physical_plan(plan, inner)
        windows = optimizer._dp_plan_ray_windows(plan, block_specs)
        previous_widths = getattr(
            optimizer, "_dp_scoring_required_widths", None
        )
        optimizer._dp_scoring_required_widths = optimizer._dp_plan_stage_widths(
            plan
        )
        try:
            cost = optimizer._replay_concurrency_aware_objective(
                block_specs,
                inner,
                workers=max(1, int(plan.n_local_workers or 1)),
                window_bytes_by_mask=windows,
            )
        finally:
            optimizer._dp_scoring_required_widths = previous_widths
        factor = optimizer._dp_worker_contention_factor(
            max(1, int(plan.n_local_workers or 1))
        )
        print(
            f"== {run['optimizer']}: nw={plan.n_local_workers} "
            f"blocks={[list(spec[0]) for spec in block_specs]}"
        )
        print(
            f"   raw cost.local={cost.local_serial:.3f} "
            f"ray={cost.ray_serial:.3f} smp={cost.smp_serial:.3f} "
            f"gpu={cost.gpu_serial:.3f} score={cost.score:.3f}"
        )
        print(
            f"   contention factor={factor:.3f} "
            f"rescored={rescored:.3f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
