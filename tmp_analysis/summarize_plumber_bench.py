"""Summarize the Cedar/Plumber/ours comparison into a table and CSV."""
import argparse
import csv
import json
from pathlib import Path

NAMES = {
    "optimizer": "Cedar",
    "plumber_optimizer": "Plumber-style",
    "dp_optimizer": "PICO (ours)",
}


def plan_summary(plan):
    """Return a compact description of the physical plan."""
    graph = {int(k): v for k, v in plan["graph"].items()}
    fused = {
        int(k): v["fused_pipes"]
        for k, v in plan["pipes"].items()
        if v.get("fused_pipes") and len(v["fused_pipes"]) > 1
    }
    parts = []
    for pid, desc in sorted(plan["pipes"].items(), key=lambda kv: int(kv[0])):
        pid = int(pid)
        if pid not in graph and pid not in fused:
            continue
        variant = desc.get("variant", "INPROCESS")
        if variant == "INPROCESS":
            continue
        ctx = desc.get("variant_ctx") or {}
        width = ctx.get("n_actors", ctx.get("n_procs"))
        label = desc.get("name", "?")
        if pid in fused:
            label = "fuse" + str(fused[pid])
        parts.append(f"{label}@{variant}" + (f"x{width}" if width else ""))
    return ", ".join(parts) if parts else "all INPROCESS"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    out = args.output
    rows = []
    for path in sorted((out / "results").glob("*.json")):
        if path.name.endswith("_probe.json"):
            continue
        payload = json.loads(path.read_text())
        if "runs" not in payload:
            continue
        workload = path.stem
        runs = {run["optimizer"]: run for run in payload["runs"]}
        ours = runs.get("dp_optimizer", {}).get("perf_time_sec")
        row = {
            "workload": workload,
            "samples": payload.get("data_num_total_samples"),
        }
        for key, name in NAMES.items():
            run = runs.get(key) or {}
            row[f"{name}_perf_s"] = round(run.get("perf_time_sec", 0.0), 3)
            row[f"{name}_tps"] = round(
                run.get("throughput_samples_per_sec", 0.0), 1
            )
            plan = (run.get("physical_plans_by_feature") or {})
            if plan:
                row[f"{name}_plan"] = plan_summary(next(iter(plan.values())))
        if ours:
            for key, name in NAMES.items():
                other = runs.get(key, {}).get("perf_time_sec")
                row[f"speedup_vs_{name}"] = (
                    round(other / ours, 2) if other else None
                )
        rows.append(row)

    if not rows:
        print(f"no results under {out}")
        return
    columns = list(rows[0].keys())
    (out / "comparison_table.csv").write_text(
        "\n".join(
            [",".join(columns)]
            + [",".join(str(row.get(c, "")) for c in columns) for row in rows]
        )
        + "\n"
    )
    header = (
        f"{'workload':<9}{'samples':>8}"
        f"{'Cedar(s)':>10}{'Plumber(s)':>12}{'PICO(s)':>10}"
        f"{'vs Cedar':>10}{'vs Plumber':>12}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['workload']:<9}{row.get('samples', ''):>8}"
            f"{row.get('Cedar_perf_s', 0):>10.3f}"
            f"{row.get('Plumber-style_perf_s', 0):>12.3f}"
            f"{row.get('PICO (ours)_perf_s', 0):>10.3f}"
            f"{str(row.get('speedup_vs_Cedar')):>10}"
            f"{str(row.get('speedup_vs_Plumber-style')):>12}"
        )
    print()
    for row in rows:
        print(f"[{row['workload']}] PICO plan: {row.get('PICO (ours)_plan')}")
        print(f"[{row['workload']}] Plumber plan: {row.get('Plumber-style_plan')}")
        print(f"[{row['workload']}] Cedar plan: {row.get('Cedar_plan')}")
    print(f"\nwrote {out / 'comparison_table.csv'}")


if __name__ == "__main__":
    main()
