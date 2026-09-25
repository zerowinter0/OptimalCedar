"""Assemble the final W-only delivery from whatever cells completed.

Reads outputs/pico_final_w_only_20260924/<workload>/results/*.json and the
chapter-3 evidence files, then writes the CSV/JSON/Markdown set the chapter
needs.  Safe to re-run: every file is regenerated from raw results, so a later
run after the campaign finishes produces the complete tables.

Usage (inside the container):
  python -u tmp_analysis/assemble_final_delivery.py
"""

import csv
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path("/workspace/OptimalCedar")
OUT = ROOT / "outputs/pico_final_w_only_20260924"
REPR = ROOT / "outputs/affine_repr_model_20260924"
DIAG = ROOT / "outputs/affine_reorder_diagnosis_20260924"
FUSION = ROOT / "outputs/fusion_discount_20260923"
PROFILE = ROOT / "outputs/affine_repr_profile_20260924"

BASELINE_KIND = {
    "optimizer": "Cedar native staged optimizer",
    "plumber_optimizer": "Plumber strategy inside Cedar",
    "raydata_optimizer": "Ray Data strategy inside Cedar",
    "unopti": "declared plan, unoptimized",
    "pico_final": "final PICO (W-only, representation-aware compute)",
    "pico_final_no_boundary": "final PICO without the boundary term",
    "pico_byte_proportional": "byte-proportional compute, same W-only search",
    "simple_dp_workers_boundary": "byte affine compute, same W-only search",
    "staged_workers_boundary_affine": "staged search, same compute model",
    "staged_boundary_affine": "staged search, byte affine compute",
}


def commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()


def plan_summary(plan: dict):
    pipes = plan.get("pipes") or {}
    fused = [
        desc.get("fused_pipes")
        for desc in pipes.values()
        if desc.get("fused_pipes")
    ]
    stages = [
        f"{desc.get('name')}:{desc.get('variant')}"
        for desc in pipes.values()
        if desc.get("variant") not in (None, "INPROCESS")
    ]
    widths = [
        desc.get("variant_ctx", {}).get("n_actors")
        or desc.get("variant_ctx", {}).get("n_procs")
        for desc in pipes.values()
        if desc.get("variant") not in (None, "INPROCESS")
    ]
    return fused, stages, widths


def collect_throughput():
    rows = []
    for workload_dir in sorted(OUT.glob("*")):
        if not workload_dir.is_dir():
            continue
        for result in sorted((workload_dir / "results").glob("*.json")):
            try:
                data = json.loads(result.read_text())
            except Exception:  # noqa: BLE001
                continue
            for run in data.get("runs", []):
                plans = run.get("physical_plans_by_feature") or {}
                key = "feature" if "feature" in plans else (
                    sorted(plans)[0] if plans else None
                )
                plan = plans.get(key) if key else {}
                fused, stages, widths = plan_summary(plan or {})
                repeats = run.get("repeat_results") or [run]
                for index, rep in enumerate(repeats):
                    rows.append(
                        {
                            "workload": workload_dir.name,
                            "cell": result.stem,
                            "optimizer": run.get("optimizer"),
                            "identity": BASELINE_KIND.get(
                                run.get("optimizer"), "unknown"
                            ),
                            "round": index + 1,
                            "throughput_samples_per_sec": rep.get(
                                "throughput_samples_per_sec"
                            ),
                            "perf_time_sec": rep.get("perf_time_sec"),
                            "setup_time_sec": rep.get("setup_time_sec"),
                            "num_samples": rep.get("num_samples"),
                            "n_local_workers": plan.get("n_local_workers"),
                            "stage_widths": ",".join(
                                str(value) for value in widths if value
                            ),
                            "fused": json.dumps(fused),
                            "stages": json.dumps(stages),
                            "status": "completed",
                            "commit": commit(),
                            "results_path": str(result.relative_to(ROOT)),
                        }
                    )
    return rows


def collect_planning():
    """Planning time and DP state counts, parsed from the per-cell logs."""
    import re

    pattern = re.compile(
        r"Exact layer (\d+)/(\d+) masks=(\d+) states=(\d+) "
        r"max_frontier=(\d+) layer_sec=([\d.eE+-]+) total_sec=([\d.eE+-]+)"
    )
    rows = []
    for log in sorted(OUT.glob("*/logs/*.log")):
        workload = log.parts[-3]
        cell = log.stem
        masks = states = 0
        total = 0.0
        operators = 0
        for line in log.read_text(errors="replace").splitlines():
            match = pattern.search(line)
            if not match:
                continue
            operators = int(match.group(2))
            masks = max(masks, int(match.group(3)))
            states = max(states, int(match.group(4)))
            total = max(total, float(match.group(7)))
        if operators:
            rows.append(
                {
                    "workload": workload,
                    "cell": cell,
                    "operators": operators,
                    "masks": masks,
                    "dp_states_max": states,
                    "dp_total_sec": total,
                    "unit": "seconds / maximum states in the exact layer",
                    "source": str(log.relative_to(ROOT)),
                }
            )
    return rows


def collect_w_scaling():
    """W evidence: fixed-structure sweep cells when present, plus the two
    complete stage-B runs that measured the same workload at W=1 and W=64."""
    rows = []
    # Preferred source: the W cells that restricted the final optimizer to one
    # W candidate (deployable arm, measured end to end by the normal harness).
    for result in sorted(OUT.glob("*/results/w_cell_W*.json")):
        requested = result.stem.replace("w_cell_W", "")
        try:
            data = json.loads(result.read_text())
        except Exception:  # noqa: BLE001
            continue
        for run in data.get("runs", []):
            plans = run.get("physical_plans_by_feature") or {}
            key = "feature" if "feature" in plans else (sorted(plans)[0] if plans else None)
            plan = plans.get(key, {}) if key else {}
            for index, rep in enumerate(run.get("repeat_results") or [run]):
                rows.append(
                    {
                        "workload": result.parts[-3],
                        "requested_workers": requested,
                        "n_local_workers": plan.get("n_local_workers"),
                        "optimizer": run.get("optimizer"),
                        "round": index + 1,
                        "throughput_samples_per_sec": rep.get(
                            "throughput_samples_per_sec"
                        ),
                        "perf_time_sec": rep.get("perf_time_sec"),
                        "num_samples": rep.get("num_samples"),
                        "provenance": (
                            "final PICO restricted to this single W candidate "
                            "(structure re-optimised at that W); one complete run"
                        ),
                        "source": str(result.relative_to(ROOT)),
                    }
                )
    stage_b = {
        "cheap": {
            "W1": ("simple_dp_boundary", "outputs/stage_b_repr_20260924/results_cheap.json"),
            "W64": ("simple_dp_repr_affine", "outputs/stage_b_repr_20260924/results_cheap.json"),
        },
        "pico": {
            "W64": ("simple_dp_workers_width_boundary", "outputs/stage_b_repr_20260924/results_pico.json"),
        },
    }
    for label, entries in stage_b.items():
        for w_label, (optimizer, path) in entries.items():
            result = ROOT / path
            if not result.exists():
                continue
            data = json.loads(result.read_text())
            run = next(
                (r for r in data.get("runs", []) if r["optimizer"] == optimizer), None
            )
            if run is None:
                continue
            rows.append(
                {
                    "workload": "simclrv2",
                    "n_local_workers": w_label.replace("W", ""),
                    "optimizer": optimizer,
                    "round": "mean_of_3",
                    "throughput_samples_per_sec": run.get(
                        "throughput_samples_per_sec"
                    ),
                    "perf_time_sec": run.get("perf_time_sec"),
                    "num_samples": run.get("num_samples"),
                    "provenance": (
                        "complete run of the same plan family at this W "
                        "(confounded with the plan structure; not a "
                        "fixed-structure sweep)"
                    ),
                    "source": path,
                }
            )
    for result in sorted(OUT.glob("*/w_scaling/results_W*.json")):
        try:
            data = json.loads(result.read_text())
        except Exception:  # noqa: BLE001
            continue
        workers = result.stem.replace("results_W", "")
        for run in data.get("runs", []):
            for index, rep in enumerate(run.get("repeat_results") or [run]):
                rows.append(
                    {
                        "workload": result.parts[-4],
                        "n_local_workers": workers,
                        "optimizer": run.get("optimizer"),
                        "round": index + 1,
                        "throughput_samples_per_sec": rep.get(
                            "throughput_samples_per_sec"
                        ),
                        "perf_time_sec": rep.get("perf_time_sec"),
                        "num_samples": rep.get("num_samples"),
                        "source": str(result.relative_to(ROOT)),
                    }
                )
    return rows


def write_csv(path: Path, rows, fields=None):
    if not rows:
        path.write_text("")
        return
    fields = fields or list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def chapter3_tables():
    """Recompute the M1..M5 plan table from the scored evidence."""
    summary = {}
    plan_rows = []
    plan_summary_csv = REPR / "plan_summary.csv"
    if plan_summary_csv.exists():
        with plan_summary_csv.open() as handle:
            plan_rows = list(csv.DictReader(handle))
    prediction_rows = []
    operator_csv = REPR / "operator_results.csv"
    if operator_csv.exists():
        with operator_csv.open() as handle:
            prediction_rows = list(csv.DictReader(handle))
    for row in plan_rows:
        measured = float(row["measured_mean_ms"])
        for model in ("M1", "M2", "M3", "M4", "M5"):
            summary.setdefault(model, []).append(float(row[model]) / measured)
    table = []
    for model, ratios in summary.items():
        table.append(
            {
                "model": model,
                "plans": len(ratios),
                "min_ratio": min(ratios),
                "max_ratio": max(ratios),
                "mape": statistics.fmean(abs(value - 1.0) for value in ratios),
                "rmse": (
                    statistics.fmean((value - 1.0) ** 2 for value in ratios)
                )
                ** 0.5,
            }
        )
    return table, plan_rows, prediction_rows


def fusion_components():
    rows = []
    source = FUSION / "expB/expB_summary.csv"
    if source.exists():
        with source.open() as handle:
            for row in csv.DictReader(handle):
                rows.append(
                    {
                        "kind": "serial_diagnostic_real_block",
                        "organisation": row["config"],
                        "round": row["round"],
                        "elapsed_ms_per_record_mean": row[
                            "elapsed_ms_per_record_mean"
                        ],
                        "compute_ms_per_record_mean": row[
                            "compute_ms_per_record_mean"
                        ],
                        "other_ms_per_record_mean": row[
                            "other_ms_per_record_mean"
                        ],
                        "units": "ms per source record",
                        "window": "serial service, no overlap, 120 batches",
                        "source": str(source.relative_to(ROOT)),
                    }
                )
    return rows


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = collect_throughput()
    write_csv(OUT / "throughput.csv", rows)
    if rows:
        ablations = [
            row
            for row in rows
            if "ablation" in row["cell"] or "staged" in row["cell"]
        ]
        write_csv(OUT / "ablation.csv", ablations)
        staging = [row for row in rows if "staged" in row["cell"]]
        write_csv(OUT / "search_comparison.csv", staging)
    table, plan_rows, operator_rows = chapter3_tables()
    planning = collect_planning()
    write_csv(OUT / "planning.csv", planning)
    write_csv(OUT / "staged_joint.csv", collect_staged_joint())
    write_csv(OUT / "oracle_results.csv", collect_oracle())
    write_csv(OUT / "w_scaling.csv", collect_w_scaling())
    write_csv(OUT / "operator_predictions.csv", operator_rows)
    write_csv(OUT / "plan_predictions.csv", plan_rows)
    write_csv(OUT / "ranking.csv", table)
    write_csv(OUT / "fusion_components.csv", fusion_components())
    (OUT / "figure_data.json").write_text(
        json.dumps(
            {
                "throughput": rows,
                "models": table,
                "plans": plan_rows,
                "operators": operator_rows,
                "fusion_components": fusion_components(),
            },
            indent=1,
            default=float,
        )
    )
    print(f"throughput rows: {len(rows)}; plan rows: {len(plan_rows)}; "
          f"operator rows: {len(operator_rows)}")
    write_reports(rows, table, plan_rows, operator_rows)
    write_manifest()
    print(f"wrote {OUT}/throughput.csv, the chapter-3 tables and the reports")
    return 0


def throughput_summary(rows):
    summary = {}
    for row in rows:
        if row.get("throughput_samples_per_sec") in (None, ""):
            continue
        key = (row["workload"], row["cell"], row["optimizer"])
        summary.setdefault(key, []).append(float(row["throughput_samples_per_sec"]))
    out = []
    for (workload, cell, optimizer), values in sorted(summary.items()):
        out.append(
            {
                "workload": workload,
                "cell": cell,
                "optimizer": optimizer,
                "identity": BASELINE_KIND.get(optimizer, "unknown"),
                "rounds": len(values),
                "throughput_mean": statistics.fmean(values),
                "throughput_min": min(values),
                "throughput_max": max(values),
                "throughput_stdev": (
                    statistics.stdev(values) if len(values) > 1 else 0.0
                ),
            }
        )
    return out


def collect_staged_joint():
    path = OUT / "staged_joint.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    rows = []
    for name, entry in (data.get("plans") or {}).items():
        rows.append(
            {
                "plan": name,
                "same_scorer_objective": entry.get("hammer_objective"),
                "workers": entry.get("workers"),
                "fused": json.dumps(entry.get("fused")),
                "parallel_stages": json.dumps(entry.get("parallel_stages")),
                "throughput_samples_per_sec": entry.get("throughput_samples_per_sec"),
                "planning_sec": entry.get("setup_time_sec"),
                "plan_file": entry.get("plan_file"),
                "staged_selected_workers": (
                    (data.get("staged_stages", {}).get("selected") or {}).get("workers")
                ),
                "staged_selected_cost_ms_per_record": (
                    (data.get("staged_stages", {}).get("selected") or {}).get(
                        "cost_ms_per_record"
                    )
                ),
            }
        )
    return rows


def collect_oracle():
    path = OUT / "oracle_results.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    rows = []
    for key, entry in data.items():
        if not isinstance(entry, dict):
            continue
        instance = entry.get("instance") if isinstance(entry.get("instance"), dict) else {}
        dp = entry.get("dp") if isinstance(entry.get("dp"), dict) else {}
        oracle = entry.get("oracle") if isinstance(entry.get("oracle"), dict) else {}
        rows.append(
            {
                "experiment": key,
                "operators": len(instance.get("operators", []) or []),
                "candidates_evaluated": entry.get("orders_evaluated"),
                "workers_enumerated": ",".join(
                    str(w) for w in (instance.get("workers") or [])
                ),
                "dp_score": dp.get("score"),
                "dp_workers": dp.get("workers"),
                "oracle_score": oracle.get("score"),
                "match": entry.get("match"),
                "status": entry.get("status", ""),
            }
        )
    return rows


def write_reports(rows, table, plan_rows, operator_rows) -> None:
    summary = throughput_summary(rows)
    written = sorted({row["workload"] for row in summary})
    models = {entry["model"]: entry for entry in table}
    lines = [
        "# Final W-only PICO：交付与完成矩阵",
        "",
        f"commit: `{commit()}`",
        "",
        "## 身份",
        "",
        "最终 PICO = `pico_final`（selector 39，`SimpleDpWorkersBoundaryAffineReprOptimizer`）：",
        "联合搜索顺序/融合/后端/缓存与 W；**不搜索 stage width**（每个并行 stage 固定 width=1）；",
        "计算项 = `k_(算子, 表示类)·元素数 + b_(算子, 表示类)`；边界项沿用字节模型。详见 `identity.json`。",
        "",
        "## 完成矩阵（由已完成 cell 自动生成）",
        "",
        "| 负载 | cells | optimizers |",
        "| --- | ---: | --- |",
    ]
    per_workload = {}
    for entry in summary:
        per_workload.setdefault(entry["workload"], set()).add(entry["cell"])
    for workload in sorted(per_workload):
        cells = ", ".join(sorted(per_workload[workload]))
        optimizers = sorted(
            {
                entry["optimizer"]
                for entry in summary
                if entry["workload"] == workload
            }
        )
        lines.append(f"| {workload} | {cells} | {', '.join(optimizers)} |")
    if not per_workload:
        lines.append("| （尚无完成的 cell） | — | — |")
    lines += [
        "",
        "## 吞吐（均值 ± 范围，samples/s）",
        "",
        "| 负载 | cell | optimizer | 轮数 | 均值 | 范围 |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for entry in summary:
        lines.append(
            f"| {entry['workload']} | {entry['cell']} | {entry['optimizer']} | "
            f"{entry['rounds']} | {entry['throughput_mean']:.1f} | "
            f"{entry['throughput_min']:.1f}–{entry['throughput_max']:.1f} |"
        )
    lines += [
        "",
        "## 计算模型（第三章，来自 `outputs/affine_repr_model_20260924`）",
        "",
        "| 模型 | 计划数 | 比值范围 | MAPE | RMSE |",
        "| --- | ---: | --- | ---: | ---: |",
    ]
    for name in ("M1", "M2", "M3", "M4", "M5"):
        entry = models.get(name)
        if entry:
            lines.append(
                f"| {name} | {entry['plans']} | {entry['min_ratio']:.2f}–"
                f"{entry['max_ratio']:.2f} | {entry['mape'] * 100:.1f}% | "
                f"{entry['rmse'] * 100:.1f}% |"
            )
    w_rows = collect_w_scaling()
    planning_rows = collect_planning()
    lines += [
        "",
        "## W 证据（`w_scaling.csv`，含来源与局限）",
        "",
        "| 负载 | W | optimizer | 吞吐 /s | 来源 |",
        "| --- | ---: | --- | ---: | --- |",
    ]
    for entry in w_rows:
        requested = entry.get("requested_workers", entry.get("n_local_workers"))
        lines.append(
            f"| {entry['workload']} | {requested} | "
            f"{entry['optimizer']} | {float(entry['throughput_samples_per_sec']):.1f} | "
            f"{entry.get('provenance', entry.get('source', ''))} |"
        )
    if not w_rows:
        lines.append("| （无） | — | — | — | — |")
    lines += [
        "",
        "局限：W=1 与 W=64 两组来自**不同计划结构**的完整运行（stage B），"
        "不是固定结构 W 扫描；固定结构扫描脚本已修好但本轮未跑成，写作时按上表标注。",
        "",
        "## 规划成本与 DP 状态数（`planning.csv`）",
        "",
        "| 负载 | cell | 算子数 | mask 数 | 最大状态数 | DP 秒 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for entry in planning_rows:
        lines.append(
            f"| {entry['workload']} | {entry['cell']} | {entry['operators']} | "
            f"{entry['masks']} | {entry['dp_states_max']} | {entry['dp_total_sec']:.1f} |"
        )
    if not planning_rows:
        lines.append("| （无） | — | — | — | — | — |")
    lines += [
        "",
        "## 失败与负结果（必须保留）",
        "",
        "- 未完成 cell（原样保留，不冒充成功）："
        + ("、".join(sorted(p.name for p in OUT.glob("*/results/*.failed.json"))) or "无"),
        "- 语义：移动 `to_float` 会改变 torchvision 的值域解释（float 图像 clamp 到 [0,1]），",
        "  SimCLRv2 上不存在语义等价重排；重排吞吐差不得写成等价优化收益（`semantic_scope.md`）。",
        "- 完整 PICO 上字节 affine 与表示感知模型的吞吐在运行噪声内不可区分",
        "  （2404 vs 2320 samples/s，区间重叠）→ 本轮不宣称吞吐提升。",
        "- 联合 oracle（顺序×融合×后端×W 的独立枚举）只完成顺序×W 轴与",
        "  `state_sufficiency`；plan-replay 入口的联合枚举被阻塞，原因记录在 `oracle_results.json`。",
        "",
        "## 复现",
        "",
        "```bash",
        "bash scripts/pico_final_all_20260924.sh        # profiles -> smoke -> campaign -> W scaling -> assembly",
        "python -u tmp_analysis/assemble_final_delivery.py   # 任何时候重跑都可以重建本目录的表格",
        "```",
    ]
    (OUT / "README.md").write_text("\n".join(lines))

    claims = [
        "# 主张 → 证据（最终 W-only 交付）",
        "",
        "| 主张 | 实验 | 原始路径 | 关键数值 | 可用范围 |",
        "| --- | --- | --- | --- | --- |",
        f"| 字节不是计算规模的充分统计量 | 表示/几何受控对照 | `docs/mechanism_20260923/affine_reorder/blur_geometry.json` | 同约 6 万字节：0.850 vs 2.606 ms | 仅该负载的算子族 |",
        f"| 元素+表示类显著降低计划成本误差 | M1–M5 消融（11 计划 × 3 轮） | `outputs/affine_repr_model_20260924/` | MAPE 41.2% → 13.5% | 计划计算量口径，非吞吐 |",
        f"| 截距 b 无独立贡献 | M4 vs M5 | 同上 | 13.6% → 13.5% | 同负载 |",
        f"| DP 在顺序×W 空间最优 | 1260 顺序独立参考 | `outputs/affine_reorder_diagnosis_20260924/dp_optimality_test.json` | 命中 argmin (11.717960) | 该受限空间 |",
        f"| 表示状态可由算子集合决定 | 前缀一致性 + 实测 payload | `outputs/pico_final_w_only_20260924/state_sufficiency.json` | 71 个 mask 无冲突 | SimCLRv2 DAG |",
        f"| 最终 PICO 的吞吐 | 主对比（每 cell 3 轮） | `throughput.csv` | 见 README 表 | 合法计划，非等价语义 |",
    ]
    (OUT / "claim_evidence.md").write_text("\n".join(claims))


def write_manifest() -> None:
    import hashlib

    small = {}
    docs_dir = ROOT / "docs/final_w_only_20260924"
    docs_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(OUT.glob("*")):
        if path.is_file() and path.suffix in (".csv", ".json", ".md"):
            data = path.read_bytes()
            entry = {
                "path": str(path.relative_to(ROOT)),
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            if path.suffix == ".csv" and data:
                entry["rows"] = len(data.decode(errors="replace").splitlines()) - 1
            small[path.name] = entry
            try:
                (docs_dir / path.name).write_bytes(data)
            except Exception:  # noqa: BLE001
                pass
    large = {}
    for path in sorted(OUT.glob("*/*/results/*.json")):
        data = path.read_bytes()
        large[str(path.relative_to(ROOT))] = {
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    for profile in sorted(PROFILE.glob("*/shared.yaml")):
        data = profile.read_bytes()
        large[str(profile.relative_to(ROOT))] = {
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    (OUT / "MANIFEST.json").write_text(
        json.dumps(
            {
                "small_files": small,
                "large_artifacts": large,
                "note": (
                    "小文件同时抄送到 docs/final_w_only_20260924/；逐 cell 原始日志在 "
                    "outputs/pico_final_w_only_20260924/<workload>/logs/，"
                    "profile 在 outputs/affine_repr_profile_20260924/<workload>/shared.yaml。"
                ),
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
