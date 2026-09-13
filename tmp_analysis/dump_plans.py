"""Write per-repeat physical plans from a comparison.json into plans/<name>/."""
import json
import sys
from pathlib import Path

import yaml

run_dir = Path(sys.argv[1] if len(sys.argv) > 1 else
               "/workspace/OptimalCedar/outputs/simclrv2_scaling_20260911")
data = json.loads((run_dir / "results" / "comparison.json").read_text())
for run in data["runs"]:
    dest = run_dir / "plans" / run["optimizer"]
    dest.mkdir(parents=True, exist_ok=True)
    repeats = run.get("repeat_results") or []
    if not repeats and run.get("physical_plans_by_feature"):
        repeats = [run]
    for index, trial in enumerate(repeats, 1):
        plans = trial.get("physical_plans_by_feature")
        if plans:
            (dest / f"round_{index}.yaml").write_text(yaml.safe_dump(plans))
    print(f"{run['optimizer']}: {len(repeats)} plan file(s) written to {dest}")
