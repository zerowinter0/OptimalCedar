"""Build the plans that discriminate the two offload cost structures.

The joint objective currently charges a Ray stage's service *and* its
marshalling to the worker lane (``local + ray``), while an SMP stage gets its
own lane (``max``).  The two structures disagree only when the local chain and
the offloaded stage are comparable in size:

  additive:  cycle = local_work + ray_service
  lanes:     cycle = max(local_work, ray_service)

This script splits an all-in-process plan's operator chain into a local part
and a Ray part at a chosen boundary, so the same workload can be measured
with both structures' predictions side by side.

Usage (inside the container):
  python tmp_analysis/make_offload_split_plans.py <plan.yaml> <outdir> \
      --split <n_local_prefix_ops> [--actors N] [--workers W]
"""

import argparse
import copy
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("plan")
    parser.add_argument("outdir")
    parser.add_argument("--split", type=int, required=True,
                        help="number of operators to keep before the Ray stage")
    parser.add_argument("--actors", type=int, default=1)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    payload = yaml.safe_load(Path(args.plan).read_text())
    plan = payload["physical_plan"]
    graph = {int(k): (int(v) if v not in ("", None) else None)
             for k, v in plan["graph"].items()}
    pipes = {int(k): v for k, v in plan["pipes"].items()}

    # Walk the materialized chain from the source.
    targets = {v for v in graph.values() if v is not None}
    current = next(k for k in graph if k not in targets)
    chain = []
    while current is not None:
        chain.append(current)
        current = graph.get(current)

    # Operators in execution order (fused pipes expand to their members).
    order = []
    for p_id in chain:
        desc = pipes[p_id]
        fused = desc.get("fused_pipes")
        if fused:
            order.extend(int(x) for x in fused)
        else:
            order.append(p_id)

    local_part = order[: args.split]
    ray_part = order[args.split :]
    if not ray_part:
        raise SystemExit("split leaves no operator for the Ray stage")

    out = copy.deepcopy(plan)
    out_pipes = {int(k): v for k, v in out["pipes"].items()}
    source_id = chain[0]
    sink_ids = [p for p in chain if graph.get(p) is None]

    # Rebuild the chain: source -> [local ops] -> fused Ray stage -> sink.
    fused_id = max(out_pipes) + 1
    for p_id in list(out_pipes):
        if p_id == source_id or p_id in sink_ids:
            continue
        out_pipes.pop(p_id)

    if local_part:
        out_pipes[local_part[0]] = {
            "name": f"FusedPipeLocal",
            "variant": "INPROCESS",
            "variant_ctx": {"variant_type": "INPROCESS"},
            "fused_pipes": local_part if len(local_part) > 1 else None,
            "execution_resource": "cpu",
        }
    out_pipes[fused_id] = {
        "name": "FusedPipeRay",
        "variant": "RAY",
        "variant_ctx": {
            "variant_type": "RAY",
            "n_actors": args.actors,
            "max_inflight": 100,
            "max_prefetch": 100,
            "submit_batch_size": 32,
            "use_threads": True,
            "num_gpus": 0.0,
        },
        "fused_pipes": ray_part if len(ray_part) > 1 else None,
        "execution_resource": "cpu",
    }

    head = local_part[0] if local_part else None
    new_graph = {}
    if head is not None:
        new_graph[source_id] = str(head)
        new_graph[head] = str(fused_id)
    else:
        new_graph[source_id] = str(fused_id)
    new_graph[fused_id] = str(sink_ids[0])
    new_graph[sink_ids[0]] = ""
    out["graph"] = new_graph
    out["pipes"] = {str(k): v for k, v in out_pipes.items()}
    if args.workers:
        out["n_local_workers"] = args.workers

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    name = args.name or f"split{args.split}_a{args.actors}"
    target = outdir / f"{name}.yaml"
    target.write_text(yaml.safe_dump({"physical_plan": out}, sort_keys=False))
    print(
        f"{target}: local={local_part} ray={ray_part} "
        f"actors={args.actors} workers={out['n_local_workers']}"
    )
    return 0


if __name__ == "__main__":
    main()
