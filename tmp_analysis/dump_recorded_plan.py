"""Write one recorded comparison cell's physical plan to YAML.

    python tmp_analysis/dump_recorded_plan.py <workload> <optimizer> <out.yaml>

The recorded runs are searched newest-protocol first, so the plan is the one
the formal comparison actually executed.  Baseline aliases (``cedar`` for
``optimizer``, ``dj`` for ``dj_optimizer``, ``pecan`` for
``pecan_optimizer``) are accepted.
"""

import json
import sys
from pathlib import Path

import yaml

ROOT = Path("/workspace/OptimalCedar")
RUNS = [
    ROOT / "outputs/pico_drained_20260914",
    ROOT / "outputs/pico_djpecan_20260914",
    ROOT / "outputs/pico_missing_20260914",
    ROOT / "outputs/pico_ten_workloads_20260913b",
]
ALIASES = {
    "cedar": ["optimizer"],
    "dj": ["dj_optimizer", "dj_two_stage_optimizer"],
    "pecan": ["pecan_optimizer", "pecan_two_stage_optimizer"],
    "plumber": ["plumber_optimizer"],
    "raydata": ["raydata_optimizer"],
    "simple_dp": ["simple_dp_optimizer"],
}


def restore_int_keys(value):
    if isinstance(value, dict):
        restored = {}
        for key, item in value.items():
            new_key = key
            if isinstance(key, str):
                try:
                    new_key = int(key)
                except ValueError:
                    pass
            restored[new_key] = restore_int_keys(item)
        return restored
    if isinstance(value, list):
        return [restore_int_keys(item) for item in value]
    return value


def normalize_pipes(plan):
    """Fill in the fields older recorded plans omit.

    Plans recorded before the variant context was serialized only carry a pipe
    name and an id; ``PhysicalPlan.from_dict`` expects ``variant`` and
    ``variant_ctx`` to exist.
    """
    for pipe in plan.get("pipes", {}).values():
        pipe["variant"] = pipe.get("variant") or "INPROCESS"
        pipe.setdefault("variant_ctx", {"variant_type": pipe["variant"]})
        pipe["variant_ctx"].setdefault("variant_type", pipe["variant"])
    return plan


def main() -> int:
    workload, requested, out_path = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    names = ALIASES.get(requested, [requested])
    for run in RUNS:
        for name in names:
            path = run / "results" / f"{workload}__{name}.json"
            if not path.is_file():
                continue
            payload = json.loads(path.read_text())
            run_entry = (payload.get("runs") or [None])[0]
            if not run_entry:
                continue
            plans = run_entry.get("physical_plans_by_feature") or {}
            if not plans:
                continue
            plan = restore_int_keys(next(iter(plans.values())))
            plan = normalize_pipes(plan)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(yaml.safe_dump({"physical_plan": plan}))
            print(f"{workload}/{name}: {out_path}  (from {run.name})")
            return 0
    print(f"{workload}/{requested}: no recorded plan")
    return 1


if __name__ == "__main__":
    sys.exit(main())
