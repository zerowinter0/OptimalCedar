"""Four-plan motivation experiment: baseline, Cedar, and two manual plans."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from evaluation.motivation_multimodal.artifacts import atomic_write_json, sha256_file
from evaluation.motivation_multimodal.calibrate import run_calibration
from evaluation.motivation_multimodal.pilot import (
    ThresholdConfig,
    _run_worker,
    _shared_args,
    iter_threshold_configs,
)


PLAN_NAMES = ("unoptimized", "cedar", "pico", "manual")
MIN_RELATIVE_COST_GAP = 0.05
MIN_RELATIVE_RUNTIME_GAP = 0.10


def _meaningfully_less(left: float, right: float) -> bool:
    """Require a visible model-cost gap so near-ties do not pass."""

    return float(left) < float(right) * (1.0 - MIN_RELATIVE_COST_GAP)


def optimized_plan_runtime_checks(
    medians: dict[str, float],
    cedar_costs: dict[str, float],
) -> dict[str, bool]:
    """Validate the predeclared runtime witness for the three optimized plans."""

    runtime_order_has_clear_gaps = (
        medians["pico"]
        < medians["manual"] * (1.0 - MIN_RELATIVE_RUNTIME_GAP)
        and medians["manual"]
        < medians["cedar"] * (1.0 - MIN_RELATIVE_RUNTIME_GAP)
    )
    cedar_reverses_other_pair = (
        (medians["manual"] - medians["cedar"])
        * (
            float(cedar_costs["manual"])
            - float(cedar_costs["cedar"])
        )
        < 0
    )
    return {
        "optimized_runtime_order_has_clear_gaps": runtime_order_has_clear_gaps,
        "cedar_reverses_manual_vs_cedar": cedar_reverses_other_pair,
    }


def plans_share_parallelism_protocol(plans: dict[str, dict[str, Any]]) -> bool:
    """Return whether every plan used the same budgeted width allocator."""

    if set(plans) != set(PLAN_NAMES):
        return False
    protocols = set()
    for result in plans.values():
        signature = result.get("resource_signature")
        if not isinstance(signature, dict):
            return False
        policy = result.get("parallelism_policy")
        cpu_budget = int(signature.get("cpu_budget", -1))
        ray_budget = int(signature.get("ray_cpu_budget", -1))
        local_used = int(signature.get("total_accounted_local_cpus", -1))
        ray_used = int(signature.get("total_accounted_ray_cpus", -1))
        gpu_used = float(signature.get("total_accounted_gpus", -1.0))
        if (
            policy != "separate_pool_equal_share"
            or cpu_budget < 1
            or ray_budget < 1
            or not 0 <= local_used <= cpu_budget
            or not 0 <= ray_used <= ray_budget
            or not 0.0 <= gpu_used <= 1.0
        ):
            return False
        protocols.add((policy, cpu_budget, ray_budget))
    return len(protocols) == 1


def _run_configuration(
    config: ThresholdConfig,
    fixture_root: Path,
    image_root: Path,
    calibration_root: Path,
    output_root: Path,
    pilot_num_samples: int,
    plan_only: bool,
) -> dict[str, Any]:
    root = output_root / "configurations" / config.key
    root.mkdir(parents=True, exist_ok=True)
    threshold = calibration_root / "thresholds" / f"{config.key}.json"
    calibration_data = fixture_root / "calibration.jsonl"
    pilot_data = fixture_root / "pilot.jsonl"
    profile = root / "profile.yml"
    paths = {name: root / f"{name}_plan.yml" for name in PLAN_NAMES}
    base = {
        "key": config.key,
        "threshold_path": str(threshold),
        "threshold_sha256": sha256_file(threshold),
    }
    common = _shared_args(calibration_data, threshold, image_root, 500)
    profile_result = _run_worker(
        ["profile", *common, "--profile", str(profile)],
        root / "profile_result.json",
        root / "profile.log",
        4 * 3600,
    )
    plans = {}
    for output_name, optimizer_name in (
        ("unoptimized", "baseline"),
        ("cedar", "cedar"),
        ("pico", "pico"),
    ):
        plans[output_name] = _run_worker(
            [
                "plan",
                *common,
                "--profile",
                str(profile),
                "--optimizer",
                optimizer_name,
                "--plan",
                str(paths[output_name]),
            ],
            root / f"{output_name}_plan_result.json",
            root / f"{output_name}_plan.log",
            3600,
        )
    plans["manual"] = _run_worker(
        [
            "manual",
            *common,
            "--baseline-plan",
            str(paths["unoptimized"]),
            "--dp-plan",
            str(paths["pico"]),
            "--profile",
            str(profile),
            "--plan",
            str(paths["manual"]),
        ],
        root / "manual_plan_result.json",
        root / "manual_plan.log",
        600,
    )
    score_args = ["score", *common, "--profile", str(profile)]
    for name in PLAN_NAMES:
        score_args.extend(["--named-plan", f"{name}={paths[name]}"])
    scores = _run_worker(
        score_args,
        root / "costs.json",
        root / "costs.log",
        600,
    )
    cedar_costs = scores["costs"]
    pico_costs = scores["pico_costs"]
    static_checks = {
        "four_distinct_plans": len({plans[name]["sha256"] for name in PLAN_NAMES}) == 4,
        "shared_budgeted_parallelism_policy": plans_share_parallelism_protocol(plans),
        "cedar_ranks_pico_first": _meaningfully_less(
            cedar_costs["pico"],
            min(float(cedar_costs[name]) for name in PLAN_NAMES if name != "pico"),
        ),
        "cedar_prefers_own_plan_to_manual": _meaningfully_less(
            cedar_costs["cedar"], cedar_costs["manual"]
        ),
        "pico_orders_optimized_plans": _meaningfully_less(
            pico_costs["pico"], pico_costs["manual"]
        ) and _meaningfully_less(
            pico_costs["manual"], pico_costs["cedar"]
        ),
    }
    if not all(static_checks.values()) or plan_only:
        result = {
            **base,
            "status": "screen_passed" if all(static_checks.values()) else "screened_out",
            "profile": profile_result,
            "plans": plans,
            "cedar_costs": cedar_costs,
            "pico_costs": pico_costs,
            "static_checks": static_checks,
        }
        atomic_write_json(root / "result.json", result)
        return result

    executions = {name: [] for name in PLAN_NAMES}
    pilot_args = _shared_args(pilot_data, threshold, image_root, pilot_num_samples)
    for repetition in range(3):
        offset = repetition % len(PLAN_NAMES)
        order = PLAN_NAMES[offset:] + PLAN_NAMES[:offset]
        for name in order:
            run = _run_worker(
                ["execute", *pilot_args, "--plan", str(paths[name])],
                root / f"{name}_run_{repetition}.json",
                root / f"{name}_run_{repetition}.log",
                3600,
            )
            executions[name].append(run)
    id_sets = [
        set(run["record_ids"])
        for name in PLAN_NAMES
        for run in executions[name]
    ]
    equivalent = bool(id_sets) and all(ids == id_sets[0] for ids in id_sets[1:])
    medians = {
        name: statistics.median(run["seconds"] for run in executions[name])
        for name in PLAN_NAMES
    }
    runtime_checks = optimized_plan_runtime_checks(medians, cedar_costs)
    qualifies = equivalent and all(runtime_checks.values())
    result = {
        **base,
        "status": "success",
        "qualifies": qualifies,
        "profile": profile_result,
        "plans": plans,
        "cedar_costs": cedar_costs,
        "pico_costs": pico_costs,
        "executions": executions,
        "median_seconds": medians,
        "equivalent": equivalent,
        "output_count": len(id_sets[0]) if equivalent else 0,
        "static_checks": static_checks,
        "runtime_checks": runtime_checks,
    }
    atomic_write_json(root / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot-num-samples", type=int, default=500)
    parser.add_argument("--grid", choices=("standard", "distinct-staged"), default="distinct-staged")
    parser.add_argument("--max-configs", type=int)
    parser.add_argument("--config-key")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed calibration and per-configuration results.",
    )
    args = parser.parse_args()
    configs = list(iter_threshold_configs(args.grid))
    if args.config_key is not None:
        configs = [config for config in configs if config.key == args.config_key]
        if not configs:
            parser.error(f"--config-key is not present in the {args.grid} grid")
    if args.max_configs is not None:
        configs = configs[: args.max_configs]
    args.output.mkdir(parents=True, exist_ok=True)
    calibration_root = args.output / "calibration"
    calibration_complete = (calibration_root / "calibration_summary.json").is_file() and all(
        (calibration_root / "thresholds" / f"{config.key}.json").is_file()
        for config in configs
    )
    if not (args.resume and calibration_complete):
        run_calibration(args.fixture, args.image_root, calibration_root, configs=configs)
    results = []
    for index, config in enumerate(configs, start=1):
        result_path = args.output / "configurations" / config.key / "result.json"
        if args.resume and result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            progress_status = f"resumed:{result['status']}"
        else:
            result = _run_configuration(
                config,
                args.fixture,
                args.image_root,
                calibration_root,
                args.output,
                args.pilot_num_samples,
                args.plan_only,
            )
            progress_status = result["status"]
        results.append(result)
        atomic_write_json(args.output / "results.json", {"status": "running", "results": results})
        print(
            f"four-plan progress={index}/{len(configs)} key={config.key} "
            f"status={progress_status}",
            flush=True,
        )
    qualifying = [result for result in results if result.get("qualifies") is True]
    atomic_write_json(
        args.output / "results.json",
        {"status": "complete", "qualifying": len(qualifying), "results": results},
    )


if __name__ == "__main__":
    main()
