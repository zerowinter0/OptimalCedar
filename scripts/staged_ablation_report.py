"""Staged-vs-DP ablation report: throughput, planning time, model score, plan.

Usage (inside the container):
  python -u scripts/staged_ablation_report.py outputs/staged_ablation_20260922
"""
import argparse
import glob
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

PAIRS = [
    ("boundary", "staged_boundary", "old_dp_boundary"),
    ("+affine", "staged_boundary_affine", "simple_dp_boundary"),
    ("+W", "staged_workers_boundary_affine", "simple_dp_workers_boundary"),
]
WORKLOADS = ["simclrv2", "simclrv2_cache", "commonvoice", "coco",
             "llava_pretrain"]


def plan_chain(plan: dict) -> str:
    pipes = {int(k): v for k, v in plan["pipes"].items()}
    graph = {int(k): str(v) for k, v in plan["graph"].items()}
    predecessors = {
        int(item)
        for value in graph.values()
        for item in value.split(",")
        if item.strip()
    }
    starts = [p_id for p_id in graph if p_id not in predecessors]
    if len(starts) != 1:
        return f"<{len(starts)} sources>"
    node, order = starts[0], []
    while True:
        order.append(node)
        successors = [int(x) for x in graph[node].split(",") if x.strip()]
        if not successors:
            break
        node = successors[0]
    parts = []
    for p_id in order:
        desc = pipes.get(p_id, {})
        name = desc.get("name") or f"pipe{p_id}"
        fused = desc.get("fused_pipes")
        if fused:
            name = "Fused{%s}" % ",".join(str(item) for item in fused)
        variant = desc.get("variant")
        ctx = desc.get("variant_ctx") or {}
        if variant and variant != "INPROCESS":
            width = ctx.get("n_actors") or ctx.get("n_procs")
            name = f"{name}[{variant} w={width}]"
        parts.append(name)
    return " -> ".join(parts)


def load_cell(run: Path, workload: str, method: str):
    pattern = run / workload / "results" / f"round1__{method}.json"
    files = sorted(glob.glob(str(pattern)))
    if not files:
        return None
    payload = json.loads(Path(files[-1]).read_text())
    runs = payload.get("runs") or []
    if not runs:
        return None
    record = runs[0]
    plans = record.get("physical_plans_by_feature") or {}
    scores = record.get("plan_costs_by_feature") or {}
    first = next(iter(plans), None)
    return {
        "throughput": record.get("throughput_samples_per_sec"),
        "samples": record.get("num_samples"),
        "setup_sec": record.get("setup_time_sec"),
        "score": scores.get(first) if first else None,
        "workers": plans[first]["n_local_workers"] if first else None,
        "chain": plan_chain(plans[first]) if first else "<no plan>",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--workloads", nargs="+", default=WORKLOADS)
    args = parser.parse_args()
    run = args.run.resolve()

    for workload in args.workloads:
        print(f"\n===== {workload}")
        for tier, staged, dp in PAIRS:
            rows = [
                (f"staged {tier}", load_cell(run, workload, staged)),
                (f"dp     {tier}", load_cell(run, workload, dp)),
            ]
            for label, cell in rows:
                if cell is None:
                    print(f"  {label:16s} <missing>")
                    continue
                print(
                    "  %-16s W=%-3s %10.1f rec/s  n=%-7s setup=%7.1fs  "
                    "score=%.4f" % (
                        label, cell["workers"],
                        cell["throughput"] or float("nan"),
                        cell["samples"], cell["setup_sec"] or float("nan"),
                        cell["score"] if cell["score"] is not None
                        else float("nan"),
                    )
                )
                print(f"        {cell['chain']}")
            staged_cell, dp_cell = rows[0][1], rows[1][1]
            if staged_cell and dp_cell:
                ratio = (dp_cell["throughput"] or 0) / (
                    staged_cell["throughput"] or 1
                )
                print(f"        -> dp/staged throughput = {ratio:.3f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
