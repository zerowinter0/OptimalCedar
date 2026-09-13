"""Compare the reference run with the layered-profile run."""
import json
import sys
from pathlib import Path

import yaml

ROOT = Path("/home/xieruiyang/OptimalCedar")
REF = ROOT / "outputs/simclrv2_four_remote_20260911"
NEW = ROOT / "outputs/simclrv2_scaling_20260911"


def load_runs(run_dir):
    path = run_dir / "results" / "comparison.json"
    data = json.loads(path.read_text())
    return {run["optimizer"]: run for run in data["runs"]}


def plan_of(run_dir, name):
    path = run_dir / "plans" / name / "round_1.yaml"
    if not path.exists():
        return None
    return yaml.safe_load(path.read_text())["feature_r0"]


def describe(plan):
    if plan is None:
        return "missing"
    graph = {int(k): v for k, v in plan["graph"].items()}
    order = []
    current = next(p for p in graph if p not in {int(x) for x in graph.values() if x})
    while current is not None:
        order.append(current)
        nxt = graph.get(current) or ""
        current = int(nxt) if str(nxt).strip() else None
    rows = []
    for pid in order:
        desc = plan["pipes"][str(pid)] if str(pid) in plan["pipes"] else plan["pipes"][pid]
        ctx = desc.get("variant_ctx") or {}
        width = ctx.get("n_actors", ctx.get("n_procs"))
        fused = desc.get("fused_pipes")
        label = desc.get("name")
        rows.append(
            f"{pid}:{label}{'(' + str(width) + ')' if width else ''}"
            f"{str(fused) if fused else ''}:{desc.get('variant')}"
        )
    return " -> ".join(rows)


def main():
    ref = load_runs(REF)
    new = load_runs(NEW)
    names = sorted(set(ref) | set(new))
    print("===== measured (reference -> layered profile) =====")
    for name in names:
        a, b = ref.get(name), new.get(name)
        def fmt(run, key):
            return "n/a" if run is None else f"{run[key]:.3f}"
        print(
            f"  {name:<18} perf {fmt(a, 'perf_time_sec')} -> "
            f"{fmt(b, 'perf_time_sec')} s | tps "
            f"{fmt(a, 'throughput_samples_per_sec')} -> "
            f"{fmt(b, 'throughput_samples_per_sec')} | "
            f"pico(DP objective) {fmt(a, 'pico_plan_cost')} -> "
            f"{fmt(b, 'pico_plan_cost')}"
        )
        if a is not None and b is not None:
            for run in (a, b):
                reps = run.get("repeat_results") or []
                if reps:
                    run["_reps"] = [
                        round(item.get("perf_time_sec", 0.0), 3) for item in reps
                    ]
        print(f"      repeats ref={a.get('_reps') if a else None} "
              f"new={b.get('_reps') if b else None}")

    print()
    print("===== plans =====")
    for name in names:
        print(f"  [{name}] ref: {describe(plan_of(REF, name))}")
        print(f"  [{name}] new: {describe(plan_of(NEW, name))}")


if __name__ == "__main__":
    sys.exit(main())
