"""Write the physical plans of one workload's comparison run to YAML files."""
import argparse
import json
from pathlib import Path

import yaml


def _restore_int_keys(value):
    """Turn JSON's stringified integer keys back into integers.

    ``PhysicalPlan.to_dict`` round-trips through JSON, so a dumped plan has
    ``"0"``-style pipe ids while the loader looks operators up by ``int``.
    """
    if isinstance(value, dict):
        restored = {}
        for key, item in value.items():
            new_key = key
            if isinstance(key, str):
                try:
                    new_key = int(key)
                except ValueError:
                    new_key = key
            restored[new_key] = _restore_int_keys(item)
        return restored
    if isinstance(value, list):
        return [_restore_int_keys(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    payload = json.loads(args.results.read_text())
    for run in payload["runs"]:
        plans = run.get("physical_plans_by_feature")
        if not plans:
            print(f"{run['optimizer']}: no plan")
            continue
        plan = next(iter(plans.values()))
        plan = _restore_int_keys(plan)
        target = args.output / f"{run['optimizer']}.yaml"
        target.write_text(yaml.safe_dump({"physical_plan": plan}))
        print(f"{run['optimizer']}: {target}")


if __name__ == "__main__":
    main()
