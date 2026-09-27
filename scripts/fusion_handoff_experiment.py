"""§3.3 fusion-cost transfer validation against the deployed PICO model.

The decomposition shown in the paper is produced by the *deployed* code:
``cedar.compose.my_optimizer.stage_handoff_ms`` — the same function
``MyOptimizer._dp_stage_boundary_components`` calls for every DP transition and
for plan replay.  This script never re-implements a boundary formula.

Phases (run in this order, so parameters are frozen before validation):

  stage-profile   U measured per stage: T_i (client), C_i (member, instrumented)
                  and boundary byte volumes, on the profiling batch set
  predict         C_i from the stage profile, H_i from the deployed estimator,
                  overall-I/O-discount and PICO decomposition for U/P/F
  validate        >=5 interleaved U/P/F rounds, member instrumentation OFF
                  (main) plus one paired instrumented pass (diagnostic)
  summarise       comparison, figure data, implementation parity

Usage (inside the container):
  python -u scripts/fusion_handoff_experiment.py stage-profile --run-dir DIR \
      --inputs outputs/fusion_discount_20260923/expB
  python -u scripts/fusion_handoff_experiment.py predict --run-dir DIR \
      --profile outputs/affine_repr_profile_20260924/simclrv2/shared.yaml
  python -u scripts/fusion_handoff_experiment.py validate --run-dir DIR
  python -u scripts/fusion_handoff_experiment.py summarise --run-dir DIR
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from cedar.compose.my_optimizer import stage_handoff_ms  # noqa: E402
from fusion_cost_split_harness import (  # noqa: E402
    BLOCK,
    ORGANISATIONS,
    StageRunner,
    _batch_records,
    _block_callables,
    _setup,
    _write_csv,
)

PIPE_IDS = {name: p_id for name, p_id in BLOCK}          # to_float=7 crop=6 flip=5
OP_ORDER = [name for name, _ in BLOCK]
BATCH_SIZE = 4
GROUP_OF_STAGE = {"U": [(name,) for name in OP_ORDER],
                  "P": [("to_float", "crop"), ("flip",)],
                  "F": [tuple(OP_ORDER)]}


# ------------------------------------------------------------ stage profile --


def cmd_stage_profile(args) -> int:
    ray, actor_classes, options = _setup(args)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    pool = list(
        torch.load(Path(args.inputs) / "inputs" / "block_inputs.pt")["records"]
    )
    callables = dict(_block_callables(args.batch_size))
    runner = StageRunner(
        "U", callables, args.batch_size, options, actor_classes,
        args.remote_cpu, args.same_remote_cpu,
    )
    rows: List[Dict[str, Any]] = []
    try:
        runner.set_instrument(True)
        for index in range(args.warmup):
            runner.run_batch(_batch_records(pool, index, args.batch_size))
        for index in range(args.batches):
            records = _batch_records(pool, index, args.batch_size)
            runner.reset_events()
            result = runner.run_batch(records)
            events = runner.member_events()
            per_stage = [0.0] * len(runner.stages)
            for event in events:
                per_stage[event["stage_index"]] += event["wall_ms"]
            for stage_index, stages in enumerate(runner.stages):
                t_ms = result["stage_times"][stage_index] * 1000.0
                c_ms = per_stage[stage_index]
                in_bytes, out_bytes = result["stage_bytes"][stage_index]
                rows.append(
                    {
                        "config": "U",
                        "batch_id": index,
                        "stage_index": stage_index,
                        "stage_members": "+".join(stages),
                        "source_records": args.batch_size,
                        "T_ms": t_ms,
                        "T_ms_per_record": t_ms / args.batch_size,
                        "C_ms": c_ms,
                        "C_ms_per_record": c_ms / args.batch_size,
                        "H_ms": t_ms - c_ms,
                        "H_ms_per_record": (t_ms - c_ms) / args.batch_size,
                        "in_bytes": in_bytes,
                        "out_bytes": out_bytes,
                        "in_bytes_per_record": in_bytes / args.batch_size,
                        "out_bytes_per_record": out_bytes / args.batch_size,
                        "batch_whole_T_ms_per_record": (
                            result["whole_time"] * 1000.0 / args.batch_size
                        ),
                        "instrument": "on",
                    }
                )
    finally:
        runner.shutdown()
    _write_csv(run_dir / "baseline_stage_profile.csv", rows)
    print(f"stage profile rows={len(rows)}", flush=True)
    return 0


# ------------------------------------------------------------------ predict --


def _profile_handoff_entry(physical_model: Dict[str, Any], p_id: int) -> Dict[str, Any]:
    operators = (
        (physical_model.get("object_boundary") or {}).get("RAY") or {}
    ).get("operators") or {}
    entry = operators.get(p_id, operators.get(str(p_id)))
    if not isinstance(entry, dict):
        raise RuntimeError(f"profile has no object_boundary entry for pipe {p_id}")
    return entry


def _stage_means(run_dir: Path) -> Dict[str, Dict[str, float]]:
    with (run_dir / "baseline_stage_profile.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    out: Dict[str, Dict[str, float]] = {}
    for name in OP_ORDER:
        subset = [row for row in rows if row["stage_members"] == name]
        out[name] = {
            "T_ms_per_record": statistics.fmean(
                float(row["T_ms_per_record"]) for row in subset
            ),
            "C_ms_per_record": statistics.fmean(
                float(row["C_ms_per_record"]) for row in subset
            ),
            "in_bytes_per_record": statistics.fmean(
                float(row["in_bytes_per_record"]) for row in subset
            ),
            "out_bytes_per_record": statistics.fmean(
                float(row["out_bytes_per_record"]) for row in subset
            ),
        }
    out["_whole_block_U_ms_per_record"] = {
        "value": statistics.fmean(
            float(row["batch_whole_T_ms_per_record"]) for row in rows
        )
    }
    return out


def _handoff(
    physical_model: Dict[str, Any],
    first_name: str,
    last_name: str,
) -> Dict[str, Any]:
    """Call the deployed estimator with the profile's real submit batch size."""
    first_entry = _profile_handoff_entry(physical_model, PIPE_IDS[first_name])
    submit_batch = int(
        first_entry["input_identity_stage"].get("submit_batch_size") or BATCH_SIZE
    )
    return stage_handoff_ms(
        physical_model,
        "RAY",
        PIPE_IDS[first_name],
        PIPE_IDS[last_name],
        input_records=1.0,
        output_records=1.0,
        submit_batch=submit_batch,
        transported_bytes_ms=0.0,
        fixed_ms=0.0,
        inflight_inflation=1.0,
    )


def cmd_predict(args) -> int:
    run_dir = Path(args.run_dir).resolve()
    profile = yaml.safe_load(Path(args.profile).read_text())
    physical = profile["physical_model"]
    stages = _stage_means(run_dir)
    compute = {name: stages[name]["C_ms_per_record"] for name in OP_ORDER}
    measured_h = {
        name: stages[name]["T_ms_per_record"] - compute[name] for name in OP_ORDER
    }
    in_a = stages["to_float"]["in_bytes_per_record"]
    out_a = stages["to_float"]["out_bytes_per_record"]
    in_b = stages["crop"]["in_bytes_per_record"]
    out_b = stages["crop"]["out_bytes_per_record"]
    in_c = stages["flip"]["in_bytes_per_record"]
    out_c = stages["flip"]["out_bytes_per_record"]

    io_base = (in_a + out_a) + (in_b + out_b) + (in_c + out_c)
    io_ab_base = (in_a + out_a) + (in_b + out_b)
    io_ab_fused = in_a + out_b
    io_abc_fused = in_a + out_c
    rho = {
        "U": 1.0,
        "P": io_ab_fused / io_ab_base,
        "F": io_abc_fused / io_base,
    }

    rows: List[Dict[str, Any]] = []
    parity: List[Dict[str, Any]] = []
    for organisation, groups in GROUP_OF_STAGE.items():
        compute_term = sum(compute[name] for name in OP_ORDER)
        handoffs = []
        for group in groups:
            result = _handoff(physical, group[0], group[-1])
            handoffs.append(
                {
                    "group": list(group),
                    "model": result["model"],
                    "ms_per_record": result["local_ms"] + result["parallel_ms"],
                    "terms": result["terms"],
                }
            )
            parity.append(
                {
                    "organisation": organisation,
                    "group": "+".join(group),
                    "first_pipe": PIPE_IDS[group[0]],
                    "last_pipe": PIPE_IDS[group[-1]],
                    "model": result["model"],
                    "submit_ms_per_sample": result["terms"].get(
                        "submit_ms_per_sample"
                    ),
                    "fetch_ms_per_sample": result["terms"].get(
                        "fetch_ms_per_sample"
                    ),
                    "handoff_ms_per_record": result["local_ms"]
                    + result["parallel_ms"],
                }
            )
        handoff_term = sum(item["ms_per_record"] for item in handoffs)
        measured_handoff = sum(measured_h[name] for name in OP_ORDER)
        rows.append(
            {
                "organisation": organisation,
                "method": "A_overall_discount",
                "rho": rho[organisation],
                "compute_ms_per_record": rho[organisation] * compute_term,
                "handoff_ms_per_record": rho[organisation] * handoff_term,
                "total_ms_per_record": rho[organisation]
                * (compute_term + handoff_term),
                "terms": json.dumps(
                    {
                        "rho": rho[organisation],
                        "io_base_per_record": io_base,
                        "io_ab_base": io_ab_base,
                        "io_ab_fused": io_ab_fused,
                        "io_abc_fused": io_abc_fused,
                        "compute_per_stage": compute,
                        "handoff_groups": handoffs,
                    }
                ),
            }
        )
        rows.append(
            {
                "organisation": organisation,
                "method": "B_pico_decomposition",
                "rho": 1.0,
                "compute_ms_per_record": compute_term,
                "handoff_ms_per_record": handoff_term,
                "total_ms_per_record": compute_term + handoff_term,
                "terms": json.dumps(
                    {
                        "compute_per_stage": compute,
                        "handoff_groups": handoffs,
                        "boundary_model": handoffs[0]["model"] if handoffs else None,
                    }
                ),
            }
        )
    fields = [
        "organisation", "method", "rho", "compute_ms_per_record",
        "handoff_ms_per_record", "total_ms_per_record", "terms",
    ]
    with (run_dir / "predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    _write_csv(run_dir / "implementation_parity.csv", parity)
    frozen = {
        "profile": str(Path(args.profile).resolve()),
        "profile_handoff_model": (
            ((physical.get("object_boundary") or {}).get("RAY") or {}).get(
                "handoff_model"
            )
        ),
        "boundary_function": "cedar.compose.my_optimizer.stage_handoff_ms",
        "compute_source": "measured member compute from baseline_stage_profile.csv",
        "baseline_stages": stages,
        "measured_handoff_ms_per_record": measured_h,
        "measured_handoff_sum_ms_per_record": measured_handoff,
        "predictions": rows,
    }
    (run_dir / "frozen_predictions.json").write_text(json.dumps(frozen, indent=1))
    print(f"predictions rows={len(rows)}", flush=True)
    return 0


# ----------------------------------------------------------------- validate --


def cmd_validate(args) -> int:
    ray, actor_classes, options = _setup(args)
    run_dir = Path(args.run_dir)
    pool = list(
        torch.load(Path(args.inputs) / "inputs" / "block_inputs.pt")["records"]
    )
    callables = dict(_block_callables(args.batch_size))
    runners: Dict[str, StageRunner] = {}
    rows: List[Dict[str, Any]] = []
    try:
        for organisation in ORGANISATIONS:
            runners[organisation] = StageRunner(
                organisation, callables, args.batch_size, options, actor_classes,
                args.remote_cpu, args.same_remote_cpu,
            )
        for instrument in ("off", "on"):
            for runner in runners.values():
                runner.set_instrument(instrument == "on")
            for runner in runners.values():
                for index in range(args.warmup):
                    runner.run_batch(_batch_records(pool, index, args.batch_size))
            rounds = args.rounds if instrument == "off" else args.instrument_rounds
            for repeat in range(rounds):
                order = list(args.order)
                if repeat % 2 == 1:
                    order = list(reversed(order))
                # rotate the starting point so no arm always runs first
                shift = repeat % len(order)
                order = order[shift:] + order[:shift]
                for organisation in order:
                    runner = runners[organisation]
                    stage_compute: List[float] = []
                    for index in range(args.batches):
                        records = _batch_records(pool, index, args.batch_size)
                        runner.reset_events()
                        result = runner.run_batch(records)
                        if instrument == "on":
                            events = runner.member_events()
                            stage_compute = [
                                sum(
                                    event["wall_ms"]
                                    for event in events
                                    if event["stage_index"] == stage_index
                                )
                                for stage_index in range(len(runner.stages))
                            ]
                        rows.append(
                            {
                                "instrument": instrument,
                                "repeat": repeat,
                                "order_index": order.index(organisation),
                                "organisation": organisation,
                                "batch_id": index,
                                "source_records": args.batch_size,
                                "whole_T_ms": result["whole_time"] * 1000.0,
                                "whole_T_ms_per_record": (
                                    result["whole_time"] * 1000.0 / args.batch_size
                                ),
                                "stage_T_ms_per_record": json.dumps(
                                    [
                                        value * 1000.0 / args.batch_size
                                        for value in result["stage_times"]
                                    ]
                                ),
                                "stage_C_ms_per_record": json.dumps(
                                    [
                                        value / args.batch_size
                                        for value in stage_compute
                                    ]
                                ),
                                "stages": json.dumps(
                                    [list(g) for g in runner.stages]
                                ),
                            }
                        )
                print(
                    f"instrument={instrument} repeat={repeat} rows={len(rows)}",
                    flush=True,
                )
    finally:
        for runner in runners.values():
            runner.shutdown()
    _write_csv(run_dir / "measurements.csv", rows)
    print(f"validation rows={len(rows)}", flush=True)
    return 0


# ---------------------------------------------------------------- summarise --


def cmd_summarise(args) -> int:
    run_dir = Path(args.run_dir).resolve()
    with (run_dir / "measurements.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    predictions = list(csv.DictReader((run_dir / "predictions.csv").open()))
    frozen = json.loads((run_dir / "frozen_predictions.json").read_text())
    summary: List[Dict[str, Any]] = []
    actual: Dict[str, Dict[str, float]] = {}
    for instrument in ("off", "on"):
        for organisation in ("U", "P", "F"):
            subset = [
                float(row["whole_T_ms_per_record"])
                for row in rows
                if row["instrument"] == instrument
                and row["organisation"] == organisation
            ]
            if not subset:
                continue
            per_repeat: Dict[str, List[float]] = {}
            for row in rows:
                if (
                    row["instrument"] == instrument
                    and row["organisation"] == organisation
                ):
                    per_repeat.setdefault(row["repeat"], []).append(
                        float(row["whole_T_ms_per_record"])
                    )
            repeat_means = [statistics.fmean(v) for v in per_repeat.values()]
            stdev = statistics.pstdev(repeat_means) if len(repeat_means) > 1 else 0.0
            half_width = (
                1.96 * stdev / max(len(repeat_means) ** 0.5, 1.0)
                if stdev > 0
                else 0.0
            )
            entry = {
                "instrument": instrument,
                "organisation": organisation,
                "batches": len(subset),
                "rounds": len(repeat_means),
                "mean_ms_per_record": statistics.fmean(subset),
                "median_ms_per_record": statistics.median(subset),
                "p10_ms_per_record": float(np.quantile(subset, 0.10)),
                "p90_ms_per_record": float(np.quantile(subset, 0.90)),
                "repeat_means": json.dumps(repeat_means),
                "repeat_mean_stdev_ms": stdev,
                "repeat_mean_ci95_ms": half_width,
            }
            summary.append(entry)
            actual.setdefault(instrument, {})[organisation] = entry[
                "mean_ms_per_record"
            ]
    _write_csv(run_dir / "summary.csv", summary)
    check = []
    for organisation in ("U", "P", "F"):
        off = actual.get("off", {}).get(organisation)
        on = actual.get("on", {}).get(organisation)
        if off is None or on is None:
            continue
        check.append(
            {
                "organisation": organisation,
                "instrument_off_ms": off,
                "instrument_on_ms": on,
                "instrument_effect_pct": 100.0 * (on - off) / off,
            }
        )
    _write_csv(run_dir / "instrumentation_check.csv", check)
    comparison = []
    for row in predictions:
        organisation = row["organisation"]
        predicted = float(row["total_ms_per_record"])
        measured = actual["off"][organisation]
        entry = next(
            item
            for item in summary
            if item["instrument"] == "off" and item["organisation"] == organisation
        )
        comparison.append(
            {
                "organisation": organisation,
                "method": row["method"],
                "predicted_compute_ms": row["compute_ms_per_record"],
                "predicted_handoff_ms": row["handoff_ms_per_record"],
                "predicted_total_ms": predicted,
                "measured_ms": measured,
                "abs_error_ms": predicted - measured,
                "rel_error": (predicted - measured) / measured,
                "measured_repeat_stdev_ms": entry["repeat_mean_stdev_ms"],
                "error_over_stdev": (
                    (predicted - measured) / entry["repeat_mean_stdev_ms"]
                    if entry["repeat_mean_stdev_ms"]
                    else float("nan")
                ),
                "rho": row["rho"],
                "terms": row["terms"],
            }
        )
    fields = list(comparison[0].keys())
    with (run_dir / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in comparison:
            writer.writerow(row)
    figure = {
        "measured_ms_per_record": actual.get("off", {}),
        "measured_instrument_on_ms_per_record": actual.get("on", {}),
        "summary": summary,
        "predictions": comparison,
        "baseline_stages": frozen["baseline_stages"],
        "measured_handoff_ms_per_record": frozen["measured_handoff_ms_per_record"],
        "profile_handoff_model": frozen["profile_handoff_model"],
        "boundary_function": frozen["boundary_function"],
    }
    (run_dir / "figure_data.json").write_text(json.dumps(figure, indent=1, default=float))
    _write_readme(run_dir, comparison, summary, frozen, check)
    print(f"comparison rows={len(comparison)}", flush=True)
    return 0


def _pct(value: float) -> str:
    return f"{100.0 * value:+.1f}%"


def _write_readme(run_dir, comparison, summary, frozen, check) -> None:
    stages = frozen["baseline_stages"]
    predicted = {(row["organisation"], row["method"]): row for row in comparison}
    lines = [
        "# §3.3 融合成本迁移验证（部署 PICO 边界模型统一版，2026-09-27）",
        "",
        "## 1. 模型身份（图里的预测就是部署实现）",
        "",
        "分项预测直接调用 `cedar.compose.my_optimizer.stage_handoff_ms`——",
        "`MyOptimizer._dp_stage_boundary_components` 在每次 DP 转移和计划回放中调用的同一个函数。",
        "实验不复制任何边界公式；`implementation_parity.csv` 记录每个块的调用参数与返回的模型名。",
        "",
        f"本轮冻结 profile 的 handoff 模型：`{frozen['profile_handoff_model']}`。",
        "",
        "边界模型的实现（相对上一版的变化）：",
        "",
        "- profiler 现在用真实合法对象测**单向**代价：submit（driver 序列化+提交+actor 接收）与",
        "  fetch（`ray.get`：取回+反序列化），分别按样本摊销；",
        "- DP 的块边界 = `submit(块首算子的输入对象) × 输入记录数 + fetch(块末算子的输出对象) × 输出记录数`；",
        "  这是可加组合，不需要「半程对称」假设；",
        "- 旧 profile 仍可运行，但会以 `identity_half_legacy` 记录并打印警告；",
        "  `CEDAR_REQUIRE_VALIDATED_BOUNDARY=1` 时缺新字段直接报错。",
        "",
        "## 2. 基准剖析（U，逐阶段，同一批 40 个批次）",
        "",
        "| 阶段 | 实测 T (ms/记录) | 实测 C (ms/记录) | 实测 T−C | 请求字节/记录 | 响应字节/记录 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in OP_ORDER:
        entry = stages[name]
        lines.append(
            f"| {name} | {entry['T_ms_per_record']:.3f} | "
            f"{entry['C_ms_per_record']:.3f} | "
            f"{entry['T_ms_per_record'] - entry['C_ms_per_record']:.3f} | "
            f"{entry['in_bytes_per_record']:.0f} | {entry['out_bytes_per_record']:.0f} |"
        )
    whole = stages["_whole_block_U_ms_per_record"]["value"]
    total = sum(stages[name]["T_ms_per_record"] for name in OP_ORDER)
    lines += [
        "",
        f"逐阶段之和 ΣT_i = {total:.3f} ms/记录；同会话整块 U = {whole:.3f} ms/记录，"
        f"差 {100*(whole-total)/total:+.1f}%（阶段之间的客户端胶水，属于同一计时边界内）。",
        "",
        "## 3. 冻结预测 vs 独立验证",
        "",
        "| 组织 | 实测 (ms/记录) | 方法 A 整体折扣 | 误差 | 方法 B PICO 分项 | 误差 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for organisation in ("U", "P", "F"):
        a = predicted.get((organisation, "A_overall_discount"))
        b = predicted.get((organisation, "B_pico_decomposition"))
        lines.append(
            f"| {organisation} | {a['measured_ms']:.2f} | "
            f"{a['predicted_total_ms']:.2f} | {_pct(a['rel_error'])} | "
            f"{b['predicted_total_ms']:.2f} | {_pct(b['rel_error'])} |"
        )
    instrument = {row["organisation"]: row for row in check}
    lines += [
        "",
        "轮间不确定性与插桩影响见 `summary.csv` 与 `instrumentation_check.csv`；",
        "主结果一律取无成员插桩值。",
        "",
        "## 4. 结论边界",
        "",
        "- 这是**串行服务时间**（一个批次在飞）的比较，不是完整流水线吞吐预测。",
        "- U 的两种方法使用同一基准（ρ=1），因此预测严格相同。",
        "- P/F 的实测时间不参与任何拟合或模型选择：预测在 `predict` 阶段冻结后才运行 `validate`。",
        "- 若分项方法在某个组织上误差仍大，README 直接记录，不调整口径。",
    ]
    if instrument:
        lines.append(
            "- 插桩影响："
            + "、".join(
                f"{org} {instrument[org]['instrument_effect_pct']:+.1f}%"
                for org in ("U", "P", "F")
                if org in instrument
            )
            + "。"
        )
    (run_dir / "README.md").write_text("\n".join(lines))


SMALL_FILES = (
    "README.md",
    "model_identity.json",
    "baseline_stage_profile.csv",
    "boundary_profile.csv",
    "predictions.csv",
    "frozen_predictions.json",
    "measurements.csv",
    "summary.csv",
    "instrumentation_check.csv",
    "comparison.csv",
    "implementation_parity.csv",
    "figure_data.json",
)


def cmd_manifest(args) -> int:
    import hashlib
    import shutil

    run_dir = Path(args.run_dir).resolve()
    profile_path = Path(args.profile).resolve()
    docs = ROOT / "docs/fusion_handoff_20260927"
    docs.mkdir(parents=True, exist_ok=True)
    frozen_dir = run_dir / "frozen_profile"
    frozen_dir.mkdir(exist_ok=True)
    shutil.copy2(profile_path, frozen_dir / profile_path.name)
    profile_bytes = profile_path.read_bytes()
    parity = list(csv.DictReader((run_dir / "implementation_parity.csv").open()))
    identity = {
        "commit": _git_commit(),
        "final_pico": {
            "class": (
                "cedar.compose.simple_dp_ablation_optimizer."
                "SimpleDpWorkersBoundaryAffineReprOptimizer"
            ),
            "selector": 39,
            "boundary_entry": "MyOptimizer._dp_stage_boundary_components",
            "boundary_function": "cedar.compose.my_optimizer.stage_handoff_ms",
            "handoff_model_used": sorted({row["model"] for row in parity}),
        },
        "profile": {
            "path": str(profile_path.relative_to(ROOT)),
            "sha256": hashlib.sha256(profile_bytes).hexdigest(),
            "frozen_copy": str((frozen_dir / profile_path.name).relative_to(ROOT)),
        },
        "experiment_parity": (
            "predictions call stage_handoff_ms directly with the profile's real "
            "submit batch size; implementation_parity.csv lists every call"
        ),
        "units": "ms per source record, serial service (one batch in flight)",
    }
    (run_dir / "model_identity.json").write_text(json.dumps(identity, indent=1))
    # boundary_profile.csv is the profile's own handoff entries for this block.
    rows = []
    profile = yaml.safe_load(profile_bytes)
    operators = (
        ((profile.get("physical_model") or {}).get("object_boundary") or {})
        .get("RAY", {})
        .get("operators", {})
    )
    for name in OP_ORDER:
        entry = operators.get(PIPE_IDS[name], operators.get(str(PIPE_IDS[name])))
        if not entry:
            continue
        ii = entry["input_identity_stage"]
        oo = entry["output_identity_stage"]
        rows.append(
            {
                "operator": name,
                "pipe_id": PIPE_IDS[name],
                "submit_batch_size": ii.get("submit_batch_size"),
                "input_records_measured": ii.get("input_records"),
                "output_records_measured": oo.get("input_records"),
                "input_identity_ms_per_sample": ii.get("mean_ms_per_sample"),
                "input_submit_ms_per_sample": ii.get("submit_ms_per_sample"),
                "output_identity_ms_per_sample": oo.get("mean_ms_per_sample"),
                "output_fetch_ms_per_sample": oo.get("fetch_ms_per_sample"),
                "method": ii.get("method"),
            }
        )
    _write_csv(run_dir / "boundary_profile.csv", rows)
    manifest = {
        "commit": _git_commit(),
        "result_dir": str(run_dir.relative_to(ROOT)),
        "small_files": {},
        "large_files": {},
        "provenance": {},
    }
    for name in SMALL_FILES:
        path = run_dir / name
        if not path.exists():
            continue
        data = path.read_bytes()
        manifest["small_files"][name] = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
            "rows": len(data.decode(errors="replace").splitlines()) - 1,
        }
        shutil.copy2(path, docs / name)
    for name in ("stage_profile.log", "validate.log", "smoke_profile.log"):
        path = run_dir / name
        if path.exists():
            data = path.read_bytes()
            manifest["large_files"][name] = {
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
            }
    for name in ("fusion_handoff_experiment.py",):
        path = ROOT / "scripts" / name
        manifest["provenance"][name] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    for name in ("my_optimizer.py", "dataset.py", "ray_service.py"):
        path = (
            ROOT / "cedar/compose" / name
            if name == "my_optimizer.py"
            else ROOT / ("cedar/client" if name == "dataset.py" else "cedar/service") / name
        )
        manifest["provenance"][f"cedar/{path.parent.name}/{name}"] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (run_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    shutil.copy2(run_dir / "MANIFEST.json", docs / "MANIFEST.json")
    shutil.copytree(frozen_dir, docs / "frozen_profile", dirs_exist_ok=True)
    print(f"manifest: {len(manifest['small_files'])} small files -> {docs}")
    return 0


def _git_commit() -> str:
    import subprocess

    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    def common(node):
        node.add_argument("--run-dir", required=True)
        node.add_argument(
            "--inputs", default="outputs/fusion_discount_20260923/expB"
        )
        node.add_argument("--batch-size", type=int, default=BATCH_SIZE)
        node.add_argument("--local-cpu", type=int, default=12)
        node.add_argument("--remote-cpu", type=int, default=8)
        node.add_argument("--same-remote-cpu", action="store_true", default=True)
        node.add_argument("--ray-ip", default="172.23.166.105:6379")

    stage = sub.add_parser("stage-profile")
    common(stage)
    stage.add_argument("--warmup", type=int, default=20)
    stage.add_argument("--batches", type=int, default=40)
    stage.set_defaults(func=cmd_stage_profile)

    predict = sub.add_parser("predict")
    predict.add_argument("--run-dir", required=True)
    predict.add_argument("--profile", required=True)
    predict.set_defaults(func=cmd_predict)

    validate = sub.add_parser("validate")
    common(validate)
    validate.add_argument("--warmup", type=int, default=10)
    validate.add_argument("--batches", type=int, default=40)
    validate.add_argument("--rounds", type=int, default=5)
    validate.add_argument("--instrument-rounds", type=int, default=1)
    validate.add_argument("--order", nargs="+", default=["U", "P", "F"])
    validate.set_defaults(func=cmd_validate)

    summarise = sub.add_parser("summarise")
    summarise.add_argument("--run-dir", required=True)
    summarise.set_defaults(func=cmd_summarise)

    manifest = sub.add_parser("manifest")
    manifest.add_argument("--run-dir", required=True)
    manifest.add_argument("--profile", required=True)
    manifest.set_defaults(func=cmd_manifest)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
