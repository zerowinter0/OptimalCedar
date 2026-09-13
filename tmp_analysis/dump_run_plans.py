"""Write the physical plans of one workload's comparison run to YAML files."""
import argparse
import json
from pathlib import Path

import yaml


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
        target = args.output / f"{run['optimizer']}.yaml"
        target.write_text(yaml.safe_dump({"physical_plan": plan}))
        print(f"{run['optimizer']}: {target}")


if __name__ == "__main__":
    main()
