"""Assemble the fusion-discount experiment data (experiments A, B, C).

Reads the three experiment outputs plus the Cedar cost export and writes
``figure_data.json`` / ``figure_data.csv`` / ``figure_data.md`` with:

  * experiment A: per compute level rho, T_U, T_F, measured retention,
    member-compute retention, other-overhead retention, unfused member share
    and the rule prediction T_U * rho;
  * experiment B: the same quantities for the real block (U/P/F);
  * experiment C: per-plan throughput, actors, Cedar costs, predicted and
    measured speedups, and the block-external cost check.

Usage:
  python -u scripts/fusion_discount_figure_data.py \
      --run-dir outputs/fusion_discount_20260923
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def mean(values: List[float]) -> Optional[float]:
    return statistics.fmean(values) if values else None


def stdev(values: List[float]) -> Optional[float]:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def group_summary(rows: List[Dict[str, str]], keys: List[str]) -> Dict[Any, Dict[str, float]]:
    grouped: Dict[Any, List[Dict[str, str]]] = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        key = key[0] if len(key) == 1 else key
        grouped.setdefault(key, []).append(row)
    out = {}
    for key, items in grouped.items():
        out[key] = {
            "elapsed_ms_per_record": mean(
                [float(i["elapsed_ms_per_record_mean"]) for i in items]
            ),
            "elapsed_stdev": stdev(
                [float(i["elapsed_ms_per_record_mean"]) for i in items]
            ),
            "compute_ms_per_record": mean(
                [float(i["compute_ms_per_record_mean"]) for i in items]
            ),
            "compute_stdev": stdev(
                [float(i["compute_ms_per_record_mean"]) for i in items]
            ),
            "other_ms_per_record": mean(
                [float(i["other_ms_per_record_mean"]) for i in items]
            ),
            "other_stdev": stdev(
                [float(i["other_ms_per_record_mean"]) for i in items]
            ),
            "rounds": len(items),
            "actors": int(items[0].get("actors") or 0),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir

    # ---------------- experiment A ----------------
    config_a = json.loads((run / "expA" / "expA_config.json").read_text())
    summary_a = group_summary(
        read_csv(run / "expA" / "expA_summary.csv"), ["config", "level"]
    )
    levels = sorted({int(k[1]) for k in summary_a if k[0] == "U"})
    rho_U = config_a["rho_U"]
    rho_P = config_a["rho_P"]
    experiment_a = []
    for level in levels:
        u = summary_a.get(("U", str(level)))
        f = summary_a.get(("F", str(level)))
        p = summary_a.get(("P", str(level)))
        if not u or not f:
            continue
        entry = {
            "level": level,
            "rho": rho_U,
            "T_U_ms_per_record": u["elapsed_ms_per_record"],
            "T_U_stdev": u["elapsed_stdev"],
            "T_F_ms_per_record": f["elapsed_ms_per_record"],
            "T_F_stdev": f["elapsed_stdev"],
            "measured_retention_T_F_over_T_U": (
                f["elapsed_ms_per_record"] / u["elapsed_ms_per_record"]
            ),
            "member_compute_retention": (
                f["compute_ms_per_record"] / u["compute_ms_per_record"]
            ),
            "other_overhead_retention": (
                f["other_ms_per_record"] / u["other_ms_per_record"]
            ),
            "unfused_member_share": (
                u["compute_ms_per_record"] / u["elapsed_ms_per_record"]
            ),
            "rule_prediction_T_F": u["elapsed_ms_per_record"] * rho_U,
            "rule_vs_measured_ratio": (
                (u["elapsed_ms_per_record"] * rho_U)
                / f["elapsed_ms_per_record"]
            ),
            "U_compute_ms_per_record": u["compute_ms_per_record"],
            "U_other_ms_per_record": u["other_ms_per_record"],
            "F_compute_ms_per_record": f["compute_ms_per_record"],
            "F_other_ms_per_record": f["other_ms_per_record"],
        }
        if p:
            entry.update(
                {
                    "rho_partial": rho_P,
                    "T_P_ms_per_record": p["elapsed_ms_per_record"],
                    "measured_retention_T_P_over_T_U": (
                        p["elapsed_ms_per_record"] / u["elapsed_ms_per_record"]
                    ),
                    "rule_prediction_T_P": u["elapsed_ms_per_record"] * rho_P,
                }
            )
        experiment_a.append(entry)

    # ---------------- experiment B ----------------
    summary_b = group_summary(
        read_csv(run / "expB" / "expB_summary.csv"), ["config"]
    )
    cedar_costs_path = run / "cedar_costs.json"
    cedar_costs = (
        json.loads(cedar_costs_path.read_text())
        if cedar_costs_path.exists() else {}
    )
    block_info = cedar_costs.get("declared_order_block", {})
    u_b = summary_b.get("U")
    experiment_b = []
    for org in ("U", "P", "F"):
        payload = summary_b.get(org)
        if not payload or not u_b:
            continue
        experiment_b.append(
            {
                "organisation": org,
                "elapsed_ms_per_record": payload["elapsed_ms_per_record"],
                "elapsed_stdev": payload["elapsed_stdev"],
                "compute_ms_per_record": payload["compute_ms_per_record"],
                "other_ms_per_record": payload["other_ms_per_record"],
                "actors": payload["actors"],
                "retention_vs_U": (
                    payload["elapsed_ms_per_record"]
                    / u_b["elapsed_ms_per_record"]
                ),
                "compute_retention_vs_U": (
                    payload["compute_ms_per_record"] / u_b["compute_ms_per_record"]
                ),
                "other_retention_vs_U": (
                    payload["other_ms_per_record"] / u_b["other_ms_per_record"]
                ),
            }
        )
    experiment_b_summary = {
        "rho": block_info.get("rho"),
        "io_base_bytes": block_info.get("io_base_bytes"),
        "io_fused_bytes": block_info.get("io_fused_bytes"),
        "member_ray_costs_ms_per_record": block_info.get(
            "member_ray_costs_ms_per_record"
        ),
        "member_ray_cost_sum_ms_per_record": block_info.get(
            "member_ray_cost_sum_ms_per_record"
        ),
        "formula_block_cost_ms_per_record": block_info.get(
            "charged_block_cost_ms_per_record"
        ),
        "formula_block_cost_from_rho": (
            (block_info.get("member_ray_cost_sum_ms_per_record") or 0.0)
            * (block_info.get("rho") or 0.0)
        ),
        "rule_prediction_T_F": (
            u_b["elapsed_ms_per_record"] * block_info["rho"]
            if u_b and block_info.get("rho") else None
        ),
        "organisations": experiment_b,
    }

    # ---------------- experiment C ----------------
    plans = cedar_costs.get("plans", [])
    throughputs: Dict[str, List[float]] = {}
    for path in sorted((run / "expC" / "throughput").glob("*.json")) if (
        run / "expC" / "throughput"
    ).exists() else []:
        payload = json.loads(path.read_text())
        label = path.stem.rsplit("_r", 1)[0]
        throughputs.setdefault(label, []).append(
            payload["throughput_samples_per_sec"]
        )
    experiment_c = []
    base_rate = mean(throughputs.get("U", []))
    base_cost = next(
        (p["whole_plan_cost_ms_per_record"] for p in plans if p["label"] == "U"),
        None,
    )
    for plan in plans:
        label = plan["label"]
        rates = throughputs.get(label, [])
        experiment_c.append(
            {
                "label": label,
                "plan_path": plan["plan_path"],
                "workers": plan["workers"],
                "whole_plan_cost_ms_per_record": plan[
                    "whole_plan_cost_ms_per_record"
                ],
                "block_cost_ms_per_record": plan["block"][
                    "charged_block_cost_ms_per_record"
                ],
                "block_external_cost_ms_per_record": plan[
                    "block_external_cost_ms_per_record"
                ],
                "predicted_speedup_vs_U": (
                    base_cost / plan["whole_plan_cost_ms_per_record"]
                    if base_cost else None
                ),
                "throughput_mean_rec_s": mean(rates),
                "throughput_stdev": stdev(rates),
                "throughput_runs": rates,
                "measured_speedup_vs_U": (
                    (mean(rates) / base_rate)
                    if rates and base_rate else None
                ),
                "repeats_ok": len(rates),
            }
        )

    figure = {
        "experiment_A": {
            "config": config_a,
            "levels": experiment_a,
            "verification": json.loads(
                (run / "expA" / "expA_verification.json").read_text()
            ),
        },
        "experiment_B": experiment_b_summary,
        "experiment_C": {
            "block_external_identical": cedar_costs.get(
                "block_external_identical"
            ),
            "plans": experiment_c,
        },
        "candidate_blocks": json.loads(
            (run / "candidate_blocks.json").read_text()
        )["usable_sorted"][:10]
        if (run / "candidate_blocks.json").exists() else [],
        "experiment_A_instrument_off": {
            "cells": {
                f"{config}@{level}": payload
                for (config, level), payload in group_summary(
                    read_csv(
                        run / "expA_instrument_off" / "expA_off_summary.csv"
                    ),
                    ["config", "level"],
                ).items()
            },
            "note": (
                "instrument=none: member timings disabled, only the batch "
                "end-to-end clock runs; the same protocol and CPU pins."
            ),
        },
    }
    (run / "figure_data.json").write_text(json.dumps(figure, indent=2))

    # long-format CSV for plotting
    rows: List[Dict[str, Any]] = []
    for entry in experiment_a:
        rows.append({"panel": "A", "x": entry["level"], "series": "measured",
                     "metric": "retention", "value":
                     entry["measured_retention_T_F_over_T_U"]})
        rows.append({"panel": "A", "x": entry["level"], "series": "cedar_rho",
                     "metric": "retention", "value": entry["rho"]})
        rows.append({"panel": "A", "x": entry["level"], "series": "rule",
                     "metric": "retention", "value":
                     entry["rule_prediction_T_F"] / entry["T_U_ms_per_record"]})
        rows.append({"panel": "A", "x": entry["level"], "series": "member_share",
                     "metric": "share", "value": entry["unfused_member_share"]})
    for entry in experiment_b:
        rows.append({"panel": "B", "x": entry["organisation"], "series": "measured",
                     "metric": "retention_vs_U", "value": entry["retention_vs_U"]})
    for entry in experiment_c:
        rows.append({"panel": "C", "x": entry["label"], "series": "measured",
                     "metric": "speedup_vs_U",
                     "value": entry["measured_speedup_vs_U"]})
        rows.append({"panel": "C", "x": entry["label"], "series": "predicted",
                     "metric": "speedup_vs_U",
                     "value": entry["predicted_speedup_vs_U"]})
    with (run / "figure_data.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = ["# Fusion 折扣实验数据（A/B/C）", ""]
    lines.append("## 实验 A：受控块（ρ 固定）")
    lines.append("")
    lines.append("| 计算档 K | ρ | T_U | T_F | 实测保留 | 成员计算保留 | 其他开销保留 | U 成员占比 | 规则预测 T_U×ρ | 预测/实测 |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for entry in experiment_a:
        lines.append(
            "| %d | %.4f | %.3f | %.3f | %.4f | %.4f | %.4f | %.4f | %.3f | %.3f |" % (
                entry["level"], entry["rho"], entry["T_U_ms_per_record"],
                entry["T_F_ms_per_record"], entry["measured_retention_T_F_over_T_U"],
                entry["member_compute_retention"], entry["other_overhead_retention"],
                entry["unfused_member_share"], entry["rule_prediction_T_F"],
                entry["rule_vs_measured_ratio"],
            )
        )
    lines.append("")
    lines.append("## 实验 B：真实块 to_float→crop→flip")
    lines.append("")
    lines.append(
        f"ρ = {experiment_b_summary['rho']:.4f}，成员 Ray 成本 = "
        f"{experiment_b_summary['member_ray_costs_ms_per_record']}，"
        f"Σ成员×ρ = {experiment_b_summary['formula_block_cost_from_rho']:.4f} ms/record；"
        f"声明（未融合）计划里这三段按 INPROCESS 计价，块内合计 "
        f"{experiment_b_summary['formula_block_cost_ms_per_record']:.4f} ms/record"
    )
    lines.append("")
    lines.append("| 组织 | 总时间 | 成员计算 | 其他开销 | actor | 保留 vs U |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for entry in experiment_b:
        lines.append("| %s | %.3f | %.3f | %.3f | %d | %.4f |" % (
            entry["organisation"], entry["elapsed_ms_per_record"],
            entry["compute_ms_per_record"], entry["other_ms_per_record"],
            entry["actors"], entry["retention_vs_U"]))
    lines.append("")
    lines.append("## 实验 C：完整流水线（W=1，4,000 条/轮 ×3）")
    lines.append("")
    lines.append("| 计划 | Cedar cost | 块成本 | 块外成本 | 预测加速 | 实测吞吐 | 实测加速 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for entry in experiment_c:
        lines.append("| %s | %.4f | %.4f | %.4f | %.3f | %s | %s |" % (
            entry["label"], entry["whole_plan_cost_ms_per_record"],
            entry["block_cost_ms_per_record"], entry["block_external_cost_ms_per_record"],
            entry["predicted_speedup_vs_U"] or float("nan"),
            ("%.2f ± %.2f" % (entry["throughput_mean_rec_s"], entry["throughput_stdev"]))
            if entry["throughput_mean_rec_s"] else "n/a",
            ("%.3f" % entry["measured_speedup_vs_U"])
            if entry["measured_speedup_vs_U"] else "n/a"))
    lines.append("")
    lines.append(f"块外模型贡献一致：{figure['experiment_C']['block_external_identical']}")
    (run / "figure_data.md").write_text("\n".join(lines))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
