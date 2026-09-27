"""Collect the uniform six-arm cost-model ablation into one table.

Reads ``outputs/pico_final_w_only_20260924/<workload>/results/ablation_matrix*.json``
and writes ``ablation_matrix.csv`` / ``ablation_matrix.md`` next to the other
delivery tables (plus a copy in ``docs/final_w_only_20260924/``).

Usage (inside the container):
  python -m tmp_analysis.ablation_matrix_table
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

ROOT = Path("/workspace/OptimalCedar")
OUT = ROOT / "outputs/pico_final_w_only_20260924"
DOCS = ROOT / "docs/final_w_only_20260924"

WORKLOADS = (
    "simclrv2",
    "simclrv2_cache",
    "commonvoice",
    "coco",
    "llava_pretrain",
    "wikitext103",
)
ARMS = (
    "pico_final",
    "staged_final",
    "old_dp_boundary",
    "pico_final_no_boundary",
    "pico_byte_proportional",
    "optimizer",
)
ARM_IDENTITY = {
    "pico_final": (
        "39",
        "joint W-only DP search + representation-aware affine compute "
        "(k_(op,class)*elements+b) + stage boundary; width fixed at 1",
    ),
    "staged_final": (
        "42",
        "Cedar's staged search priced by the final PICO cost model "
        "(Cedar cost function replaced; same compute/boundary/backend/W rules)",
    ),
    "old_dp_boundary": (
        "28",
        "DP search priced by Cedar's legacy profile entries (baseline latency "
        "+ whole-pipeline offload throughput) with Cedar's fixed+bytes "
        "boundary; no W search",
    ),
    "pico_final_no_boundary": (
        "40",
        "same W-only search and element/representation affine compute, explicit "
        "boundary term removed (operator-level change only)",
    ),
    "pico_byte_proportional": (
        "41",
        "byte-proportional compute (y = x in bytes, i.e. Cedar's operator "
        "layer) + boundary model + the same W-only search (boundary-level "
        "change only)",
    ),
    "optimizer": ("0", "Cedar native staged optimizer (baseline)"),
}
ARM_LABEL = {
    "pico_final": "PICO（全模型）",
    "staged_final": "Cedar 搜索 + PICO 成本模型",
    "old_dp_boundary": "PICO 搜索 + 旧 Cedar 成本模型",
    "pico_final_no_boundary": "PICO − 边界项（只改算子层）",
    "pico_byte_proportional": "PICO − 算子层（只改边界）",
    "optimizer": "Cedar 原生",
}


def _plan_summary(plan: dict) -> str:
    if not plan:
        return ""
    fused = [desc.get("fused_pipes") for desc in (plan.get("pipes") or {}).values()
             if desc.get("fused_pipes")]
    stages = [f"{desc.get('name')}:{desc.get('variant')}"
              for desc in (plan.get("pipes") or {}).values()
              if desc.get("variant") not in (None, "INPROCESS")]
    return f"W={plan.get('n_local_workers')}; fused={fused}; stages={stages}"


def main() -> int:
    rows = []
    missing = []
    for workload in WORKLOADS:
        results = sorted((OUT / workload / "results").glob("ablation_matrix*.json"))
        results = [path for path in results if not path.name.endswith(".failed.json")]
        if not results:
            missing.append(workload)
        for path in results:
            data = json.loads(path.read_text())
            cell = path.stem
            for run in data.get("runs", []):
                plans = run.get("physical_plans_by_feature") or {}
                key = "feature" if "feature" in plans else (
                    sorted(plans)[0] if plans else None
                )
                plan = plans.get(key) if key else {}
                repeats = run.get("repeat_results") or [run]
                for index, rep in enumerate(repeats):
                    rows.append(
                        {
                            "workload": workload,
                            "cell": cell,
                            "arm": run.get("optimizer"),
                            "selector": ARM_IDENTITY.get(run.get("optimizer"), ("", ""))[0],
                            "model_identity": ARM_IDENTITY.get(run.get("optimizer"), ("", ""))[1],
                            "round": index + 1,
                            "num_samples": rep.get("num_samples"),
                            "perf_time_sec": rep.get("perf_time_sec"),
                            "throughput_samples_per_sec": rep.get(
                                "throughput_samples_per_sec"
                            ),
                            "setup_time_sec": rep.get("setup_time_sec"),
                            "plan_cost": rep.get("plan_cost"),
                            "plan_summary": _plan_summary(plan or {}),
                            "results_path": str(path.relative_to(ROOT)),
                        }
                    )
        for path in sorted((OUT / workload / "results").glob(
            "ablation_matrix*.failed.json"
        )):
            rows.append(
                {
                    "workload": workload,
                    "cell": path.stem.replace(".failed", ""),
                    "arm": "(cell failed)",
                    "selector": "",
                    "model_identity": "recorded failure; see the cell log",
                    "round": 0,
                    "num_samples": "",
                    "perf_time_sec": "",
                    "throughput_samples_per_sec": "",
                    "setup_time_sec": "",
                    "plan_cost": "",
                    "plan_summary": "",
                    "results_path": str(path.relative_to(ROOT)),
                }
            )
    fields = list(rows[0].keys()) if rows else [
        "workload", "cell", "arm", "selector", "model_identity", "round",
        "num_samples", "perf_time_sec", "throughput_samples_per_sec",
        "setup_time_sec", "plan_cost", "plan_summary", "results_path",
    ]
    with (OUT / "ablation_matrix.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # Markdown view: one row per workload, one column per arm (normalised to
    # that workload's PICO arm so the models can be compared across loads).
    lines = [
        "# 统一六臂成本模型消融（2026-09-27）",
        "",
        "每个负载跑同一组臂、同一份 profile、同一资源预算、同一数据量（= 对应 main_fast 的量）。",
        "括号内为相对该负载 `pico_final` 的比例。",
        "",
        "| 负载 | " + " | ".join(ARM_LABEL[arm] for arm in ARMS) + " |",
        "| --- | " + " | ".join("---:" for _ in ARMS) + " |",
    ]
    for workload in WORKLOADS:
        values = {}
        for row in rows:
            if row["workload"] != workload or not row["throughput_samples_per_sec"]:
                continue
            try:
                values[row["arm"]] = float(row["throughput_samples_per_sec"])
            except (TypeError, ValueError):
                continue
        base = values.get("pico_final")
        cells = []
        for arm in ARMS:
            value = values.get(arm)
            if value is None:
                cells.append("—")
            elif base:
                cells.append(f"{value:.1f}（{value / base:.2f}×）")
            else:
                cells.append(f"{value:.1f}")
        lines.append(f"| {workload} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "臂身份：",
        "",
        "| 臂 | selector | 含义 |",
        "| --- | --- | --- |",
    ]
    for arm in ARMS:
        selector, identity = ARM_IDENTITY[arm]
        lines.append(f"| {ARM_LABEL[arm]} | {selector} | {identity} |")
    if missing:
        lines += ["", f"未产出结果：{', '.join(missing)}"]
    (OUT / "ablation_matrix.md").write_text("\n".join(lines))
    DOCS.mkdir(parents=True, exist_ok=True)
    for name in ("ablation_matrix.csv", "ablation_matrix.md"):
        (DOCS / name).write_bytes((OUT / name).read_bytes())
    print(f"ablation matrix rows={len(rows)} workloads_with_results="
          f"{len(WORKLOADS) - len(missing)}/{len(WORKLOADS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
