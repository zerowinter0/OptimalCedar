"""Collect every number the result figures need into one document.

Three blocks:
  1. steady-state throughput  = num_samples / perf_time_sec  (the campaign's
     "数据量 / 稳态时间" convention);
  2. optimization time        = setup_time_sec of the cell (for optimizer cells
     this is the optimizer's planning + plan materialisation wall time);
  3. model accuracy           = the three cost models (cedar, plumber, PICO)
     scoring every plan, plus their ranking against the measured throughput.

Usage (inside the container):
  python -u scripts/collect_figure_data_20260921.py

Writes outputs/figure_data_20260921/figure_data.json and
docs/figure_data_20260921.md.
"""

import json
import logging
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import yaml  # noqa: E402

from score_plan_cost_models import (  # noqa: E402
    build_scorers,
    load_plan,
    plan_chain,
    plumber_cost,
)

CAMPAIGN = ROOT / "outputs/ultimate_eight_optimizers_fix_20260921"
LLAVA_PICO_RUN = ROOT / "outputs/pico_w_only_20260921"
OUT_JSON = ROOT / "outputs/figure_data_20260921/figure_data.json"
OUT_MD = ROOT / "docs/figure_data_20260921.md"

WORKLOADS = [
    "simclrv2",
    "simclrv2_cache",
    "commonvoice",
    "coco",
    "llava_pretrain",
    "stackexchange",
]

# (figure label, status.json method, results/plans file stem, results-doc name, note)
CELLS = [
    ("unopt", "unopti", "unopti", "unopti", "未优化，直接执行原计划"),
    ("plumber", "plumber-opt", "plumber_optimizer", "plumber-opt", "Plumber 基线"),
    ("raydata", "ray-opt", "raydata_optimizer", "ray-opt", "Ray Data 基线"),
    (
        "cedar",
        "cedar-opt",
        "optimizer",
        "cedar-opt",
        "Cedar 原始分阶段优化器",
    ),
    (
        "cedar-dp",
        "old-dp-opt",
        "old_dp_legacy_optimizer",
        "old-dp-opt",
        "DP，但只读 Cedar 旧 profile 属性（SimpleDpOptimizer）",
    ),
    (
        "PICO-Resource",
        "old_dp_boundary",
        "old_dp_boundary",
        "dp-boundary",
        "只用 stage boundary 项（OldDpBoundaryOptimizer）",
    ),
    (
        "PICO-Resource-Op",
        "simple_dp_boundary",
        "simple_dp_boundary",
        "dp-boundary-affine",
        "boundary + 每算子 kx+b（SimpleDpBoundaryOptimizer）",
    ),
    (
        "PICO",
        "simple_dp_workers_width_boundary",
        "simple_dp_workers_width_boundary",
        "dp-boundary-affine-W-width",
        "boundary + kx+b + Workers + width 搜索（SimpleDpWorkersWidthBoundaryOptimizer）",
    ),
]

# llava PICO comes from the W-only run (the joint W x width search does not
# finish in the cell budget); stackexchange has no PICO plan at all.
PICO_PLAN_OVERRIDE = {
    "llava_pretrain": LLAVA_PICO_RUN / "llava_pretrain/plans/pico_w_only.yaml",
}
PICO_RESULT_OVERRIDE = {
    "llava_pretrain": LLAVA_PICO_RUN / "llava_pretrain/results/round1__pico_w_only.json",
}
# Records behind one counted sample in the W-only llava harness (4 images per
# record).  The campaign cells count records, so the figure needs both.
PICO_NUM_SAMPLES_PER_RECORD = {"llava_pretrain": 4}


def read_status(workload: str) -> dict:
    path = CAMPAIGN / "status.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    out = {}
    for cell in payload.get(workload, {}).get("cells", []):
        method = cell.get("method")
        if method:
            out[method] = {
                "status": cell.get("status"),
                "reason": cell.get("reason"),
            }
    return out


def read_cell(workload: str, method: str, stem: str, status: dict) -> dict:
    """Measured numbers of one cell (throughput, times, state)."""
    result = CAMPAIGN / workload / f"results/round1__{stem}.json"
    plan = CAMPAIGN / workload / f"plans/round1__{stem}.yaml"
    override = workload in PICO_RESULT_OVERRIDE and method == (
        "simple_dp_workers_width_boundary"
    )
    if override:
        result = PICO_RESULT_OVERRIDE[workload]
        plan = PICO_PLAN_OVERRIDE[workload]
    entry = {
        "method": method,
        "status": status.get(method, {}).get("status", "missing"),
        "reason": status.get(method, {}).get("reason"),
        "plan_path": str(plan.relative_to(ROOT)) if plan.exists() else None,
    }
    if not result.exists():
        return entry
    run = json.loads(result.read_text())["runs"][0]
    # Only the W-only override counts images, every campaign cell counts records.
    per_record = PICO_NUM_SAMPLES_PER_RECORD.get(workload, 1) if override else 1
    entry.update(
        {
            "status": "completed",
            "num_samples": run.get("num_samples"),
            "num_records": (run.get("num_samples") or 0) / per_record,
            "perf_time_sec": run.get("perf_time_sec"),
            "setup_time_sec": run.get("setup_time_sec"),
            "total_time_sec": run.get("total_time_sec"),
            "throughput_samples_per_sec": run.get("throughput_samples_per_sec"),
            "throughput_records_per_sec": (
                (run.get("throughput_samples_per_sec") or 0.0) / per_record
            ),
            "counts_images_per_record": per_record > 1,
        }
    )
    return entry


def score_plans(workload: str) -> dict:
    """Three-model scores for every plan of this workload."""
    profile_path = CAMPAIGN / workload / "profiles/shared.yaml"
    profile = yaml.safe_load(profile_path.read_text())
    cedar, pico, inner_ops = build_scorers(
        workload, profile, enable_caching=workload.endswith("_cache")
    )
    scores = {}
    for label, _method, stem, _doc_name, _note in CELLS:
        plan_path = CAMPAIGN / workload / f"plans/round1__{stem}.yaml"
        if workload in PICO_PLAN_OVERRIDE and stem == "simple_dp_workers_width_boundary":
            plan_path = PICO_PLAN_OVERRIDE[workload]
        if not plan_path.exists():
            continue
        plan = load_plan(plan_path)
        workers = max(1, int(plan.n_local_workers or 1))
        entry = {
            "plan_path": str(plan_path.relative_to(ROOT)),
            "workers": workers,
            "chain": plan_chain(plan),
            "cedar_cost_ms_per_source_record": round(
                cedar.calculate_cost(plan.graph, plan=plan), 4
            ),
            "plumber_cost_ms_per_source_record": plumber_cost(plan, profile)[
                "cost_ms_per_source_record"
            ],
        }
        try:
            score = pico.calculate_dp_objective_cost(plan=plan)
            entry["pico_score"] = round(score, 4)
            entry["pico_cost_ms_per_source_record"] = round(score / workers, 4)
        except Exception as exc:  # noqa: BLE001
            entry["pico_error"] = f"{type(exc).__name__}: {exc}"
        scores[label] = entry
    return scores, inner_ops


def _ranks(values, reverse: bool):
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=reverse)
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            out[order[k]] = rank
        i = j + 1
    return out


def _pearson(a, b) -> float:
    ma, mb = statistics.mean(a), statistics.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = math.sqrt(
        sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)
    )
    return num / den if den else float("nan")


def _kendall_tau_b(x, y) -> float:
    n0 = n1 = n2 = conc = disc = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            n0 += 1
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx == 0 and dy == 0:
                n1 += 1
                n2 += 1
            elif dx == 0:
                n1 += 1
            elif dy == 0:
                n2 += 1
            elif (dx > 0) == (dy > 0):
                conc += 1
            else:
                disc += 1
    den = math.sqrt((n0 - n1) * (n0 - n2))
    return (conc - disc) / den if den else float("nan")


def rank_agreement(scores: dict, measured: dict) -> dict:
    """Compare each model's cost ranking with the measured throughput ranking."""
    labels = [
        label
        for label, _m, _s, _d, _n in CELLS
        if label in scores
        and measured.get(label, {}).get("throughput_records_per_sec")
        and label in scores
        and "pico_cost_ms_per_source_record" in scores[label]
    ]
    excluded = []
    for label, _m, _s, _d, _n in CELLS:
        if label in labels:
            continue
        cell = measured.get(label, {})
        score = scores.get(label)
        if not score:
            continue
        if score.get("pico_error"):
            excluded.append((label, score["pico_error"]))
    if len(labels) < 4:
        return {
            "labels": labels,
            "excluded": excluded,
            "note": "comparable cells < 4, ranking statistics not computed",
        }
    throughput = [measured[label]["throughput_records_per_sec"] for label in labels]
    out = {
        "labels": labels,
        "excluded": excluded,
        "throughput_rank": _ranks(throughput, reverse=True),
    }
    for model, key in (
        ("cedar", "cedar_cost_ms_per_source_record"),
        ("plumber", "plumber_cost_ms_per_source_record"),
        ("pico", "pico_cost_ms_per_source_record"),
    ):
        cost = [scores[label][key] for label in labels]
        cost_rank = _ranks(cost, reverse=False)
        out[f"{model}_cost"] = cost
        out[f"{model}_rank"] = cost_rank
        out[f"{model}_spearman"] = round(
            _pearson(cost_rank, out["throughput_rank"]), 4
        )
        out[f"{model}_kendall"] = round(
            _kendall_tau_b(cost_rank, out["throughput_rank"]), 4
        )
        out[f"{model}_top1"] = labels[cost_rank.index(min(cost_rank))]
    out["measured_top1"] = labels[out["throughput_rank"].index(min(out["throughput_rank"]))]
    return out


def fmt(value, digits=2):
    if value is None:
        return "—"
    if isinstance(value, float) and not math.isfinite(value):
        return "—"
    return f"{value:.{digits}f}"


def main() -> int:
    logging.disable(logging.INFO)
    data = {"workloads": {}, "cells": [c[0] for c in CELLS], "notes": {}}
    md = []
    for workload in WORKLOADS:
        status = read_status(workload)
        measured = {
            label: read_cell(workload, method, stem, status)
            for label, method, stem, _doc_name, _note in CELLS
        }
        print(f"[{workload}] measuring done, scoring plans ...", flush=True)
        scores, inner_ops = score_plans(workload)
        agreement = rank_agreement(scores, measured)
        data["workloads"][workload] = {
            "measured": measured,
            "scores": scores,
            "rank_agreement": agreement,
            "inner_ops": inner_ops,
        }
        print(
            f"[{workload}] scored {len(scores)} plans; "
            f"agreement={ {k: v for k, v in agreement.items() if k.endswith('spearman')} }",
            flush=True,
        )

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(data, indent=2))

    # ---------------------------------------------------------------- markdown
    labels = [c[0] for c in CELLS]
    md.append("# 实验结果图所需数据（2026-09-21）\n")
    md.append(
        "本文件由 `scripts/collect_figure_data_20260921.py` 生成，包含三张图的数据：\n"
        "1. 每个负载上各 optimizer 的**稳态吞吐量**（柱状图）；\n"
        "2. 每个 optimizer 在各负载上的**优化时间**（cedar 在 llava/stackexchange 上无结果）；\n"
        "3. 用 **cedar / plumber / PICO** 三个 cost model 给每个 plan 打分，"
        "并与实测吞吐排序对比（模型准确率）。\n"
    )
    md.append("## 0. 命名映射\n")
    md.append("| 图中的名字 | 结果文档里的名字 | 实现 | 说明 |")
    md.append("| --- | --- | --- | --- |")
    for label, _method, stem, doc_name, note in CELLS:
        md.append(f"| {label} | {doc_name} | `{stem}` | {note} |")
    md.append("")
    md.append(
        "数据来源：`outputs/ultimate_eight_optimizers_fix_20260921`（正式放大 campaign，"
        "simclrv2 189,380 条 = 9,469 × 20；commonvoice 300,000；coco 50,000；"
        "llava_pretrain 50,000 输入 / 43,940 条过过滤；stackexchange 20,000 输入）；"
        "llava 的 PICO 取 `outputs/pico_w_only_20260921`（W-only，见 §2 注）。\n"
    )
    md.append("### 0.1 使用说明与注意事项\n")
    md.append(
        "- **缺失单元**：`unopt@commonvoice` / `unopt@coco` 执行超 2 h 未产出 plan，"
        "既无吞吐也不参与打分；`cedar@llava_pretrain` / `cedar@stackexchange` 为已知的 Cedar "
        "优化超时（`skipped_user_requested`，reason = *Known Cedar optimization timeout for "
        "this workload*）；`PICO@stackexchange` 规划超过 2 h cell 上限（`skipped_previous_timeout`），"
        "没有 plan。\n"
        "- **llava 的 PICO** 只有 W-only 结果：`CEDAR_DP_WIDTH_LADDER=1` 只约束搜索候选，最终资源分配"
        "仍把 stage 扩宽到 `SMP w=63`，因此不是严格的 width=1 消融；该次 harness 以 **4 张图/记录**"
        "计数（175,760 个计数样本），文档已折算成 records/s（÷4 → 43,940 条）。\n"
        "- **吞吐口径**：稳态吞吐 = 数据量 / 稳态时间，其中稳态时间取 cell 的 `perf_time_sec`"
        "（= Σ epoch_run_times）。`summary.csv` 里的 `mean_input_records_per_sec` 用的是"
        "「数据量 / workload wall」（含每个 epoch 的启动与排空），数值更高，两者不要混用。\n"
        "- **cost 口径**：cedar 是单 worker 的 ms/source-record；plumber 已按 plan 的 W 折算"
        "（`1000/(瓶颈单 worker 速率 × W)`）；PICO 的 `calculate_dp_objective_cost` 是 W-conditioned 的 "
        "S，表里同时给 S 与 S/W。\n"
        "- **模型覆盖率**：只有 PICO 会拒绝 plan（coco 3 个、原因见 §3.0）。被拒绝的 cell 从排序统计里"
        "剔除，**三个模型都在同一子集上比较**，避免“谁覆盖得多谁占便宜”。\n"
    )

    md.append("## 1. 稳态吞吐量（柱状图）\n")
    md.append("单位：records/s（= 数据量 / 稳态时间 `perf_time_sec`）。\n")
    md.append("| 负载 | " + " | ".join(labels) + " |")
    md.append("| --- |" + " ---: |" * len(labels))
    for workload in WORKLOADS:
        row = [workload]
        for label in labels:
            cell = data["workloads"][workload]["measured"][label]
            value = cell.get("throughput_records_per_sec")
            row.append(fmt(value, 1) if value else f"— *{cell['status']}*")
        md.append("| " + " | ".join(row) + " |")
    md.append("")
    md.append("每负载明细（数据量、稳态时间、优化时间、总时长、状态）：\n")
    for workload in WORKLOADS:
        md.append(f"### {workload}\n")
        md.append(
            "| optimizer | 数据量(条) | 稳态时间(s) | 稳态吞吐(rec/s) | "
            "优化时间(s) | 总时长(s) | 状态 |"
        )
        md.append("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
        for label in labels:
            cell = data["workloads"][workload]["measured"][label]
            md.append(
                "| {label} | {n} | {perf} | {thr} | {setup} | {total} | {status} |".format(
                    label=label,
                    n=f"{(cell.get('num_records') or 0):,.0f}" if cell.get("num_samples") else "—",
                    perf=fmt(cell.get("perf_time_sec")),
                    thr=fmt(cell.get("throughput_records_per_sec"), 1),
                    setup=fmt(cell.get("setup_time_sec"), 1),
                    total=fmt(cell.get("total_time_sec"), 1),
                    status=cell["status"],
                )
            )
        md.append("")

    md.append("## 2. 优化时间\n")
    md.append("单位：秒，取 cell 的 `setup_time_sec`（optimizer 规划 + 计划物化；"
              "unopti 只有构建开销）。\n")
    md.append("| 负载 | " + " | ".join(labels) + " |")
    md.append("| --- |" + " ---: |" * len(labels))
    for workload in WORKLOADS:
        row = [workload]
        for label in labels:
            cell = data["workloads"][workload]["measured"][label]
            value = cell.get("setup_time_sec")
            row.append(fmt(value, 1) if value is not None else f"— *{cell['status']}*")
        md.append("| " + " | ".join(row) + " |")
    md.append("")

    md.append("## 3. 三个 cost model 的估计与排序\n")
    md.append(
        "口径：cedar = `Optimizer.calculate_cost`（单 worker、ms/source-record）；"
        "plumber = plan 宽度瓶颈 + `W`（ms/source-record，1/rate）；"
        "PICO = `SimpleDpWorkersWidthBoundaryOptimizer.calculate_dp_objective_cost`（S），"
        "表中同时给出 S/W。\n"
    )
    md.append("**模型覆盖率**（每个负载有多少 plan 能被该模型定价）：\n")
    md.append("| 负载 | 有 plan 的 cell | cedar | plumber | PICO | PICO 打不了分的 cell |")
    md.append("| --- | ---: | ---: | ---: | ---: | --- |")
    for workload in WORKLOADS:
        block = data["workloads"][workload]
        scores = block["scores"]
        measured = block["measured"]
        pico_ok = [label for label in scores if "pico_score" in scores[label]]
        excluded = [
            f"{label}（{scores[label]['pico_error'].split(':')[0]}）"
            for label in scores
            if "pico_error" in scores[label]
        ]
        planned_but_unrun = [
            label
            for label in scores
            if not measured.get(label, {}).get("throughput_records_per_sec")
        ]
        md.append(
            "| {w} | {n} | {n} | {n} | {k} | {notes} |".format(
                w=workload,
                n=len(scores),
                k=len(pico_ok),
                notes="、".join(excluded + planned_but_unrun) or "—",
            )
        )
    md.append("")

    md.append("**汇总：模型给出的 cost 排序与实测吞吐排序的一致性**\n")
    md.append(
        "| 负载 | 可比 cell | cedar ρ | plumber ρ | PICO ρ | cedar τ | plumber τ | PICO τ | "
        "实测最优 | cedar 最优 | plumber 最优 | PICO 最优 |"
    )
    md.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |")
    for workload in WORKLOADS:
        agreement = data["workloads"][workload]["rank_agreement"]
        if "note" in agreement:
            md.append(f"| {workload} | {len(agreement.get('labels', []))} | — | — | — | — | — | — | — | — | — | — |")
            continue
        md.append(
            "| {w} | {n} | {cs} | {ps} | {ks} | {ck} | {pk} | {kk} | {best} | {c} | {p} | {k} |".format(
                w=workload,
                n=len(agreement["labels"]),
                cs=agreement["cedar_spearman"],
                ps=agreement["plumber_spearman"],
                ks=agreement["pico_spearman"],
                ck=agreement["cedar_kendall"],
                pk=agreement["plumber_kendall"],
                kk=agreement["pico_kendall"],
                best=agreement["measured_top1"],
                c=agreement["cedar_top1"],
                p=agreement["plumber_top1"],
                k=agreement["pico_top1"],
            )
        )
    md.append("")
    for workload in WORKLOADS:
        block = data["workloads"][workload]
        md.append(f"### 3.{WORKLOADS.index(workload) + 1} {workload}\n")
        md.append(
            "| optimizer | cedar cost (ms) | plumber cost (ms) | PICO score S | PICO S/W | "
            "实测吞吐(rec/s) | 实测排名 | cedar 排名 | plumber 排名 | PICO 排名 |"
        )
        md.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        agreement = block["rank_agreement"]
        ranks = {
            model: dict(zip(agreement.get("labels", []), agreement.get(f"{model}_rank", [])))
            for model in ("cedar", "plumber", "pico")
        }
        measured_rank = dict(
            zip(agreement.get("labels", []), agreement.get("throughput_rank", []))
        )
        for label in labels:
            score = block["scores"].get(label)
            cell = block["measured"][label]
            if not score:
                md.append(
                    f"| {label} | — | — | — | — | "
                    f"{fmt(cell.get('throughput_records_per_sec'), 1)} | "
                    f"{fmt(measured_rank.get(label), 1)} | — | — | — |"
                )
                continue
            pico_score = score.get("pico_score")
            pico_note = "" if pico_score is not None else " *(pico error)*"
            md.append(
                "| {label} | {cedar} | {plumber} | {pico} | {pico_w} | {thr} | {mr} | "
                "{cr} | {pr} | {kr}{note} |".format(
                    label=label,
                    cedar=fmt(score["cedar_cost_ms_per_source_record"]),
                    plumber=fmt(score["plumber_cost_ms_per_source_record"]),
                    pico=fmt(pico_score),
                    pico_w=fmt(score.get("pico_cost_ms_per_source_record")),
                    thr=fmt(cell.get("throughput_records_per_sec"), 1),
                    mr=fmt(measured_rank.get(label), 1),
                    cr=fmt(ranks["cedar"].get(label), 1),
                    pr=fmt(ranks["plumber"].get(label), 1),
                    kr=fmt(ranks["pico"].get(label), 1),
                    note=pico_note,
                )
            )
        if "note" in agreement:
            md.append(f"\n*{agreement['note']}*\n")
        else:
            md.append(
                "\n可比 cell {n} 个（三者都能定价的）；实测最优 = `{best}`；"
                "cedar 最优 = `{c}`；plumber 最优 = `{p}`；"
                "PICO 最优 = `{k}`。Spearman ρ（cost 排名 vs 吞吐排名，越接近 1 越好）："
                "cedar {cs}、plumber {ps}、PICO {ks}；Kendall τ：cedar {ck}、"
                "plumber {pk}、PICO {kk}。\n".format(
                    n=len(agreement.get("labels", [])),
                    best=agreement.get("measured_top1"),
                    c=agreement.get("cedar_top1"),
                    p=agreement.get("plumber_top1"),
                    k=agreement.get("pico_top1"),
                    cs=agreement.get("cedar_spearman"),
                    ps=agreement.get("plumber_spearman"),
                    ks=agreement.get("pico_spearman"),
                    ck=agreement.get("cedar_kendall"),
                    pk=agreement.get("plumber_kendall"),
                    kk=agreement.get("pico_kendall"),
                )
            )
        if agreement.get("excluded"):
            md.append(
                "被排除在排序之外的 cell（有 plan 但 PICO 无法定价）："
                + "；".join(
                    f"`{label}` — {reason}"
                    for label, reason in agreement["excluded"]
                )
                + "。\n"
            )

    md.append("## 4. 复现\n")
    md.append("```bash\n# 容器内\npython -u scripts/collect_figure_data_20260921.py\n```\n")
    md.append(
        f"JSON（每个 plan 的完整打分与计划链）：`{OUT_JSON.relative_to(ROOT)}`\n"
    )
    OUT_MD.write_text("\n".join(md))
    print(f"wrote {OUT_JSON} and {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
