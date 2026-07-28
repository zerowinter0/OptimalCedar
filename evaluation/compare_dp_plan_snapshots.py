#!/usr/bin/env python3
"""Compare archived and regenerated DpOptimizer physical plans."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _differences(before: Any, after: Any, path: str = "") -> List[str]:
    if type(before) is not type(after):
        return [path or "$"]
    if isinstance(before, dict):
        result = []
        for key in sorted(set(before) | set(after), key=str):
            child = f"{path}.{key}" if path else str(key)
            if key not in before or key not in after:
                result.append(child)
            else:
                result.extend(_differences(before[key], after[key], child))
        return result
    if isinstance(before, list):
        if len(before) != len(after):
            return [path or "$"]
        result = []
        for index, (left, right) in enumerate(zip(before, after)):
            result.extend(_differences(left, right, f"{path}[{index}]"))
        return result
    return [] if before == after else [path or "$"]


def compare(before_root: Path, after_root: Path) -> Dict[str, Any]:
    workloads = sorted(path.stem for path in before_root.glob("*.yaml"))
    result: Dict[str, Any] = {"schema_version": 1, "workloads": {}}
    for workload in workloads:
        before_path = before_root / f"{workload}.yaml"
        after_path = after_root / f"{workload}.yaml"
        if not after_path.exists():
            result["workloads"][workload] = {
                "identical": False,
                "error": "regenerated plan is missing",
            }
            continue
        before = yaml.safe_load(before_path.read_text())
        after = yaml.safe_load(after_path.read_text())
        changed_paths = _differences(before, after)
        result["workloads"][workload] = {
            "identical": not changed_paths,
            "before_sha256": _digest(before_path),
            "after_sha256": _digest(after_path),
            "changed_path_count": len(changed_paths),
            "changed_paths": changed_paths[:200],
        }
    result["all_identical"] = all(
        item.get("identical", False)
        for item in result["workloads"].values()
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.before, args.after)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
