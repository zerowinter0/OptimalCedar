"""Count the Ray actors a fixed-plan cell actually creates.

The runner does not retain per-worker actor registrations, so the actor count
is observed directly: the plan is executed as a subprocess while this process
polls ``ray.util.list_actors()`` on the same cluster.

Usage (inside the container):
  python -u scripts/block_actor_probe.py --plan <plan.yaml> --label R-U_w1 \
      --num-samples 2000 --out outputs/<run>/actor_probe.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from block_mechanism_common import SIMCLRV2_DATASET  # noqa: E402

DATASET_FILE = ROOT / "evaluation/pipelines/simclrv2/cedar_dataset.py"
RAY_IP = "172.23.166.105:6379"
DEFAULT_PROFILE = (
    ROOT / "outputs/ultimate_eight_optimizers_fix_20260921/simclrv2/profiles/shared.yaml"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import ray
    from ray._private import state as ray_state

    ray.init(address=RAY_IP, ignore_reinit_error=True)
    env = dict(os.environ)
    env.update(
        CEDAR_RAY_PLACEMENT_RESOURCE="cedar_remote",
        CEDAR_RAY_REQUIRE_REMOTE="1",
        CEDAR_RAY_ACTOR_READY_TIMEOUT_SEC="900",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    command = [
        sys.executable, "-u", str(ROOT / "scripts/run_fixed_plan_throughput.py"),
        "--plan", str(args.plan),
        "--label", args.label,
        "--results-path", str(args.out.parent / f"actor_probe_{args.label}.json"),
        "--dataset-file", str(DATASET_FILE),
        "--dataset-kwargs", f"dataset_path={SIMCLRV2_DATASET}",
        "--batch-size", "4",
        "--num-epochs", str(args.epochs),
        "--num-total-samples", str(args.num_samples),
        "--profiled-stats", str(args.profile),
        "--ray-ip", RAY_IP,
    ]
    process = subprocess.Popen(
        command, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    peak = 0
    names: Counter = Counter()
    locations: Counter = Counter()
    try:
        while process.poll() is None:
            actors = ray_state.actors()
            # Only the Cedar stage actors of the plan under test; the cluster
            # keeps dead actors of earlier runs in this table.
            relevant = [
                info for info in actors.values()
                if info.get("State") == "ALIVE"
                and info.get("ActorClassName", "")
                in ("RayActorMapperPipeVariant",
                    "RayActorFusedOptimizerPipeVariant")
            ]
            peak = max(peak, len(relevant))
            for actor in relevant:
                names[actor.get("ActorClassName", "?")] += 1
                address = actor.get("Address") or ""
                ip = ""
                if "'IPAddress':" in address:
                    ip = address.split("'IPAddress':")[1].split(",")[0].strip(" '\"")
                locations[ip or "?"] += 1
            time.sleep(2.0)
    finally:
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=60)

    payload = {
        "label": args.label,
        "plan": str(args.plan),
        "peak_alive_actors": peak,
        "actor_names_observed": {k: v for k, v in names.items()},
        "actor_locations": dict(locations),
        "returncode": process.returncode,
        "num_samples": args.num_samples,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    ray.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
