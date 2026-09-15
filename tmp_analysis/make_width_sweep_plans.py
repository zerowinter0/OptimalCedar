"""Generate one fused-Ray plan per (worker count, actors per worker) pair.

Used to measure how a stage's achieved per-record rate depends on the total
number of actors a host runs, which is what a plan's declared width really
buys.  Every plan here is the same work (all operators fused into a single Ray
stage), so the only difference is how the actor pool is distributed over
workers.

Usage (inside the container):
  python tmp_analysis/make_width_sweep_plans.py <template.yaml> <outdir>
"""

import copy
import sys
from pathlib import Path

import yaml

SWEEP = [
    ("w32_a1", 32, 1),
    ("w32_a2", 32, 2),
    ("w16_a2", 16, 2),
    ("w8_a4", 8, 4),
    ("w8_a7", 8, 7),
    ("w8_a2", 8, 2),
    ("w8_a1", 8, 1),
]


def submit_batch_for(samples: int, workers: int, actors: int) -> int:
    """The submit batch the harness materializes for a finite workload.

    ``apply_profile_matched_resources`` bounds the batch so that a finite
    workload still spreads over the whole actor pool: three batches per actor.
    Measuring with the plan's own (huge) declared batch instead would submit
    one batch per worker, which silently uses a single actor per worker.
    """
    return max(1, -(-samples // max(1, workers * actors * 3)))


def main() -> int:
    template = yaml.safe_load(Path(sys.argv[1]).read_text())["physical_plan"]
    outdir = Path(sys.argv[2])
    samples = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
    outdir.mkdir(parents=True, exist_ok=True)
    for name, workers, actors in SWEEP:
        plan = copy.deepcopy(template)
        plan["n_local_workers"] = workers
        for pipe in plan["pipes"].values():
            ctx = pipe.get("variant_ctx") or {}
            if ctx.get("variant_type") == "RAY":
                ctx["n_actors"] = actors
                batch = submit_batch_for(samples, workers, actors)
                ctx["submit_batch_size"] = batch
                ctx["max_inflight"] = max(
                    int(ctx.get("max_inflight") or 0),
                    batch * actors * 3,
                )
            elif ctx.get("variant_type") == "SMP":
                ctx["n_procs"] = actors
        target = outdir / f"{name}.yaml"
        target.write_text(yaml.safe_dump({"physical_plan": plan}, sort_keys=False))
        print(
            f"{target}  workers={workers} actors={actors} "
            f"submit_batch={submit_batch_for(samples, workers, actors)}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
