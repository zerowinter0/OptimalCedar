"""Execute arbitrary plan files ({"physical_plan": ...}) and report the time."""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path("/workspace/OptimalCedar")
REF = ROOT / "outputs/simclrv2_four_remote_20260911"
OUT = ROOT / "tmp_analysis/plan_runs"
DATA = "evaluation/pipelines/target_pipeline/simclr/cedar_dataset.py"
ADDRESS = "172.23.166.105:6379"
PROFILE = ROOT / "outputs/simclrv2_scaling_20260911/profile.yaml"


def prepare(name, path):
    """Normalize a plan file so the runtime accepts it."""
    data = yaml.safe_load(Path(path).read_text())
    plan = data.get("physical_plan", data)
    payload = plan.get("feature_r0", plan)
    payload["graph"] = {int(k): v for k, v in payload["graph"].items()}
    payload["pipes"] = {int(k): v for k, v in payload["pipes"].items()}
    for pipe in payload["pipes"].values():
        pipe.setdefault("variant", "INPROCESS")
    # The formal experiment fixes the local worker count at 8; plans that were
    # produced without Feature.optimize() keep the unconstrained default.
    payload["n_local_workers"] = int(os.environ.get("CEDAR_PLAN_WORKERS", "8"))
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / f"{name}.yaml"
    target.write_text(yaml.safe_dump({"physical_plan": payload}))
    return target


def run(name, path):
    log = OUT / f"{name}.log"
    env = dict(os.environ)
    env.update(
        {
            "CEDAR_RAY_PLACEMENT_RESOURCE": "cedar_remote",
            "CEDAR_MATCH_PROFILE_RESOURCES": "1",
            "CEDAR_PROFILE_MATCH_FIXED_LOCAL_WORKERS": "8",
            "CEDAR_PROFILE_MATCH_CPU_BUDGET": "64",
            "CEDAR_PROFILE_MATCH_RAY_CPU_BUDGET": "64",
        }
    )
    cmd = [
        sys.executable,
        "-u",
        str(REF / "entry.py"),
        "evaluation/eval_cedar.py",
        "--dataset_file",
        DATA,
        "--batch_size",
        "1",
        "--num_total_samples",
        "9469",
        "--profiled_stats",
        str(PROFILE),
        "--master_feature_config",
        str(path),
        "--use_ray",
        "--ray_ip",
        ADDRESS,
        "--disable_controller",
        "--disable_caching",
    ]
    with log.open("w") as handle:
        subprocess.run(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, env=env)
    text = log.read_text(errors="replace")
    match = re.search(r"Total time: ([0-9.]+)s", text)
    return {
        "plan": name,
        "total_time_sec": float(match.group(1)) if match else None,
        "log": str(log),
    }


def main():
    specs = []
    for arg in sys.argv[1:]:
        name, _, path = arg.partition("=")
        specs.append((name, path))
    results = []
    for name, path in specs:
        target = prepare(name, path)
        print(f"[plans] running {name}", flush=True)
        result = run(name, target)
        print("[plans] " + json.dumps(result), flush=True)
        results.append(result)
        (OUT / "plan_runs.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
