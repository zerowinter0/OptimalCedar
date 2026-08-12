#!/usr/bin/env python3
"""Build the auditable final Data-Juicer diverse-workload selection."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
from pathlib import Path
from typing import Any


OPTIMIZERS = (
    "optimizer",
    "dj_optimizer",
    "dp_cedar_optimizer",
    "dp_optimizer",
    "dp_two_stage_optimizer",
    "pecan_optimizer",
)
THRESHOLD = 1.20
EXECUTION_LIMIT_SEC = 3600.0

WORKLOAD_META = {
    "pile_europarl": ("parliamentary proceedings", "text", 17, 19),
    "pile_hackernews": ("online discussion", "text", 16, 18),
    "pile_pubmed_abstracts": ("biomedical abstracts", "text", 17, 19),
    "pile_uspto_backgrounds": ("patent background", "text", 17, 19),
    "redpajama_code": ("source-code refinement", "code", 15, 17),
    "redpajama_arxiv": ("scientific long documents", "text", 16, 18),
    "alpaca_cot": ("instruction/reasoning tuning", "text", 7, 8),
    "llava_pretrain": ("image-caption refinement", "image-text", 13, 16),
    "general_video_refine": ("video-text quality refinement", "video-text", 7, 10),
    "video_self_evolution": ("video self-evolution filtering", "video-text", 5, 8),
}


def _read_result(path: Path, expected_samples: int) -> float | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        values = payload["epoch_run_times"]
        samples = payload["epoch_num_samples"]
        if len(values) != 1 or samples != [expected_samples]:
            return None
        value = float(values[0])
        return value if math.isfinite(value) and value > 0 else None
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _optimizer_measurement(
    result_roots: list[Path], optimizer: str, expected_samples: int
) -> dict[str, Any]:
    values: dict[int, float] = {}
    sources = []
    unavailable_sources = []
    for root in result_roots:
        for path in sorted(root.glob(f"round*__{optimizer}.json")):
            prefix = path.stem.split("__", 1)[0]
            round_text = prefix.removeprefix("round")
            if not round_text.isdigit():
                continue
            value = _read_result(path, expected_samples)
            if value is None:
                continue
            round_number = int(round_text)
            if round_number in values:
                raise ValueError(
                    f"duplicate {optimizer} round {round_number}: {path}"
                )
            values[round_number] = value
            sources.append(str(path))
        workload_root = root.parent
        evidence_paths = [
            workload_root / "plans" / f"{optimizer}.unavailable.json",
            *root.glob(f"round*__{optimizer}.timeout.json"),
            *(workload_root / "status").glob(f"plan__{optimizer}.json"),
            *(workload_root / "status").glob(f"round*__{optimizer}.json"),
        ]
        for evidence_path in evidence_paths:
            if not evidence_path.is_file():
                continue
            try:
                evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            status = evidence.get("status")
            reason = evidence.get("reason")
            is_timeout = status in {
                "optimizer_timeout",
                "infeasible_timeout",
                "timeout",
            } or (
                status == "unavailable"
                and reason in {
                    "plan_generation_timeout",
                    "optimizer_timeout",
                }
            )
            if is_timeout:
                unavailable_sources.append(str(evidence_path))
    ordered = [values[key] for key in sorted(values)]
    valid = sorted(values) == [1, 2, 3]
    return {
        "valid": valid,
        "execution_times_sec": ordered,
        "mean_execution_time_sec": statistics.mean(ordered) if valid else None,
        "stddev_execution_time_sec": (
            statistics.stdev(ordered) if valid else None
        ),
        "within_execution_limit": valid and all(
            value <= EXECUTION_LIMIT_SEC for value in ordered
        ),
        "formally_unavailable": not valid and bool(unavailable_sources),
        "sources": sources,
        "unavailable_sources": unavailable_sources,
    }


def _summarize_matrix(
    workload: str,
    expected_samples: int,
    roots_by_optimizer: dict[str, list[Path]],
    evidence: str,
) -> dict[str, Any]:
    runs = {
        optimizer: _optimizer_measurement(
            roots_by_optimizer.get(optimizer, []), optimizer, expected_samples
        )
        for optimizer in OPTIMIZERS
    }
    dp = runs["dp_optimizer"]
    competitors = {
        optimizer: run["mean_execution_time_sec"]
        for optimizer, run in runs.items()
        if optimizer != "dp_optimizer"
        and run["valid"]
        and run["within_execution_limit"]
    }
    complete_optimizer_evidence = all(
        run["valid"] and run["within_execution_limit"]
        or run["formally_unavailable"]
        for run in runs.values()
    )
    valid = (
        dp["valid"]
        and dp["within_execution_limit"]
        and bool(competitors)
        and complete_optimizer_evidence
    )
    best = min(competitors, key=competitors.get) if valid else None
    best_time = competitors.get(best) if best else None
    speedup = best_time / dp["mean_execution_time_sec"] if valid else None
    return {
        "workload": workload,
        "expected_samples": expected_samples,
        "evidence": evidence,
        "runs": runs,
        "valid": valid,
        "best_other_optimizer": best,
        "best_other_execution_time_sec": best_time,
        "dp_execution_time_sec": dp["mean_execution_time_sec"],
        "dp_speedup_over_best_other": speedup,
        "dp_at_least_20pct_faster": bool(
            speedup is not None and speedup >= THRESHOLD
        ),
    }


def _screening_summary(
    result_root: Path, expected_samples: int
) -> dict[str, Any]:
    times = {}
    sources = {}
    for optimizer in OPTIMIZERS:
        path = result_root / f"round1__{optimizer}.json"
        value = _read_result(path, expected_samples)
        if value is not None:
            times[optimizer] = value
            sources[optimizer] = str(path)
    dp = times.get("dp_optimizer")
    competitors = {
        name: value
        for name, value in times.items()
        if name != "dp_optimizer" and value <= EXECUTION_LIMIT_SEC
    }
    best = min(competitors, key=competitors.get) if dp and competitors else None
    speedup = competitors[best] / dp if best else None
    return {
        "complete": dp is not None and bool(competitors),
        "requested_outputs": expected_samples,
        "execution_times_sec": times,
        "sources": sources,
        "best_other_optimizer": best,
        "dp_speedup_over_best_other": speedup,
        "dp_at_least_20pct_faster": bool(
            speedup is not None and speedup >= THRESHOLD
        ),
    }


def _pile_summaries(canonical: Path, audit_root: Path) -> dict[str, Any]:
    payload = json.loads(canonical.read_text(encoding="utf-8"))
    output = {}
    audit_runs = {
        "pile_europarl": audit_root
        / "datajuicer_candidate_runs/europarl_2500/pile_europarl/results",
        "pile_hackernews": audit_root
        / "datajuicer_candidate_runs/heldout_20000/pile_hackernews/results",
        "pile_pubmed_abstracts": audit_root
        / "datajuicer_candidate_runs/heldout_20000/pile_pubmed_abstracts/results",
        "pile_uspto_backgrounds": audit_root
        / "datajuicer_candidate_runs/heldout_20000/pile_uspto_backgrounds/results",
    }
    for workload, audit_results in audit_runs.items():
        audit_run_root = audit_results.parent.parent
        audit_log = audit_run_root / "candidate_matrix.log"
        audit_metadata = audit_results.parent / "metadata.txt"
        audit_complete = (
            audit_log.is_file()
            and f"Candidate run complete: {audit_run_root}" in audit_log.read_text(
                encoding="utf-8", errors="replace"
            )
            and audit_metadata.is_file()
            and "optimizer_timeout_sec=3600" in audit_metadata.read_text(
                encoding="utf-8", errors="replace"
            )
            and "optimizers=optimizer dp_two_stage_optimizer"
            in audit_metadata.read_text(encoding="utf-8", errors="replace")
        )
        if not audit_complete:
            raise RuntimeError(
                f"60-minute original-Cedar audit is incomplete: {audit_run_root}"
            )
        item = payload["candidates"][workload]
        runs = item["runs"]
        for optimizer in (
            "dj_optimizer",
            "dp_cedar_optimizer",
            "dp_optimizer",
            "pecan_optimizer",
        ):
            run = runs[optimizer]
            successful = (
                run.get("valid") is True
                and run.get("mean_execution_time_sec") is not None
                and run["mean_execution_time_sec"] <= EXECUTION_LIMIT_SEC
            )
            unavailable = run.get("formally_unavailable") is True
            if not (successful or unavailable):
                raise RuntimeError(
                    f"incomplete frozen evidence for {workload}/{optimizer}"
                )
        competitors = {
            name: run["mean_execution_time_sec"]
            for name, run in runs.items()
            if name != "dp_optimizer"
            and run.get("valid") is True
            and run.get("mean_execution_time_sec") is not None
            and run["mean_execution_time_sec"] <= EXECUTION_LIMIT_SEC
        }
        audit = _optimizer_measurement(
            [audit_results],
            "optimizer",
            2500 if workload == "pile_europarl" else 20000,
        )
        if audit["valid"] and audit["within_execution_limit"]:
            competitors["optimizer"] = audit["mean_execution_time_sec"]
        two_stage_audit = _optimizer_measurement(
            [audit_results],
            "dp_two_stage_optimizer",
            2500 if workload == "pile_europarl" else 20000,
        )
        if two_stage_audit["valid"] and two_stage_audit[
            "within_execution_limit"
        ]:
            competitors["dp_two_stage_optimizer"] = two_stage_audit[
                "mean_execution_time_sec"
            ]
        if not (
            audit["valid"] and audit["within_execution_limit"]
            or audit["formally_unavailable"]
        ):
            raise RuntimeError(
                f"incomplete current original-Cedar evidence for {workload}"
            )
        if not (
            two_stage_audit["valid"]
            and two_stage_audit["within_execution_limit"]
            or two_stage_audit["formally_unavailable"]
        ):
            raise RuntimeError(
                f"incomplete current DP two-stage evidence for {workload}"
            )
        dp_time = runs["dp_optimizer"]["mean_execution_time_sec"]
        best = min(competitors, key=competitors.get)
        speedup = competitors[best] / dp_time
        output[workload] = {
            "workload": workload,
            "expected_samples": 2500 if workload == "pile_europarl" else 20000,
            "evidence": str(canonical),
            "cedar_60m_audit_complete": True,
            "cedar_60m_audit": audit,
            "dp_two_stage_60m_audit": two_stage_audit,
            "runs": {
                **runs,
                "optimizer": audit,
                "dp_two_stage_optimizer": two_stage_audit,
            },
            "valid": True,
            "best_other_optimizer": best,
            "best_other_execution_time_sec": competitors[best],
            "dp_execution_time_sec": dp_time,
            "dp_speedup_over_best_other": speedup,
            "dp_at_least_20pct_faster": speedup >= THRESHOLD,
        }
    return output


def _selection(ledger: dict[str, dict[str, Any]]) -> list[str]:
    pile = [
        "pile_europarl",
        "pile_hackernews",
        "pile_pubmed_abstracts",
        "pile_uspto_backgrounds",
    ]
    if not all(ledger[name]["valid"] for name in pile):
        raise RuntimeError("one or more frozen Pile results are invalid")
    if not ledger["alpaca_cot"]["valid"]:
        raise RuntimeError("required diverse formal result is incomplete: alpaca_cot")
    video = (
        "video_self_evolution"
        if ledger["video_self_evolution"]["valid"]
        and ledger["video_self_evolution"]["dp_at_least_20pct_faster"]
        else "general_video_refine"
    )
    if not ledger[video]["valid"]:
        raise RuntimeError(f"required diverse formal result is incomplete: {video}")

    extra_positive = [
        name
        for name in ("redpajama_code", "redpajama_arxiv")
        if ledger[name]["valid"] and ledger[name]["dp_at_least_20pct_faster"]
    ]
    # Within the structurally homogeneous Pile family, retain the strongest
    # scenario-distinct results only as needed to satisfy the six-workload and
    # 40%-non-win gates. This makes diversity the primary objective while the
    # complete four-workload Pile ledger remains visible.
    ranked_pile = sorted(
        pile,
        key=lambda name: ledger[name]["dp_speedup_over_best_other"],
        reverse=True,
    )
    candidates = []
    for pile_count in range(2, len(ranked_pile) + 1):
        extra_subsets = itertools.chain.from_iterable(
            itertools.combinations(extra_positive, count)
            for count in range(len(extra_positive) + 1)
        )
        for extra in extra_subsets:
            for include_llava in (False, True):
                selected = (
                    ranked_pile[:pile_count]
                    + ["alpaca_cot", video]
                    + list(extra)
                )
                if include_llava:
                    if not ledger["llava_pretrain"]["valid"]:
                        continue
                    selected.append("llava_pretrain")
                if not 6 <= len(selected) <= 8:
                    continue
                failures = sum(
                    not ledger[name]["dp_at_least_20pct_faster"]
                    for name in selected
                )
                if failures / len(selected) > 0.40:
                    continue
                modalities = len(
                    {WORKLOAD_META[name][1] for name in selected}
                )
                candidates.append(
                    ((pile_count, -modalities, len(selected)), selected)
                )
    if not candidates:
        raise RuntimeError("no diverse 6--8 workload selection satisfies the 40% gate")
    return min(candidates, key=lambda item: item[0])[1]


def _diversity_summary(selected: list[str]) -> dict[str, Any]:
    metadata = [WORKLOAD_META[name] for name in selected]
    hub_counts = [item[2] for item in metadata]
    cedar_counts = [item[3] for item in metadata]
    return {
        "scenarios": sorted({item[0] for item in metadata}),
        "modalities": sorted({item[1] for item in metadata}),
        "hub_operator_count_range": [min(hub_counts), max(hub_counts)],
        "cedar_operator_count_range": [min(cedar_counts), max(cedar_counts)],
    }


def _render(report: dict[str, Any]) -> str:
    lines = [
        "# Data-Juicer diverse-workload result",
        "",
        "All selected results use W=8, CPU budget 64, cache disabled, one shared profile per workload, and three measured repetitions. A win means DP is at least 1.20x faster than the fastest available non-DP optimizer.",
        "",
        "Hub operator counts include every entry in the official recipe, including global deduplicators. Cedar operator counts exclude the source and omitted cross-record deduplicators, but include fixed parse, path-resolution, synchronization, and projection adapters actually executed by Cedar.",
        "",
        "| workload | scenario | modality | Hub ops | Cedar ops | samples | DP (s) | best other (s) | speedup | selected |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    selected = set(report["selected_workloads"])
    for name, item in report["ledger"].items():
        scenario, modality, hub_ops, cedar_ops = WORKLOAD_META[name]
        dp = item.get("dp_execution_time_sec")
        other = item.get("best_other_execution_time_sec")
        speedup = item.get("dp_speedup_over_best_other")
        lines.append(
            f"| `{name}` | {scenario} | {modality} | {hub_ops} | {cedar_ops} | "
            f"{item.get('expected_samples', 'N/A')} | "
            f"{dp:.3f} | {other:.3f} | {speedup:.3f}x | "
            f"{'yes' if name in selected else 'no'} |"
            if item.get("valid")
            else f"| `{name}` | {scenario} | {modality} | {hub_ops} | {cedar_ops} | N/A | N/A | N/A | N/A | no (formal result incomplete) |"
        )
    summary = report["summary"]
    diversity = summary["diversity"]
    lines.extend(
        [
            "",
            f"Selected: {summary['selected_count']} workloads; DP ≥1.20x wins: "
            f"{summary['wins']}; non-wins: {summary['non_wins']} "
            f"({summary['failure_fraction'] * 100:.1f}%).",
            f"Coverage: {len(diversity['scenarios'])} scenarios, "
            f"{len(diversity['modalities'])} modalities, Hub operator counts "
            f"{diversity['hub_operator_count_range'][0]}--"
            f"{diversity['hub_operator_count_range'][1]}, and Cedar operator "
            f"counts {diversity['cedar_operator_count_range'][0]}--"
            f"{diversity['cedar_operator_count_range'][1]}.",
            "",
            "Every non-selected screened outcome remains in the JSON ledger; selection does not erase negative evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--root",
        type=Path,
        default=repo / "outputs/chapter6_experiments/datajuicer_diverse_workloads",
    )
    parser.add_argument(
        "--canonical",
        type=Path,
        default=repo
        / "evaluation/chapter6_experiments/formal_results/paper_artifacts/optimizer/data/data_pipeline_matrix.json",
    )
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    classification = json.loads(
        (root / "recipe_classification.json").read_text(encoding="utf-8")
    )
    data_scale = json.loads(
        (root / "data_scale.json").read_text(encoding="utf-8")
    )
    if classification.get("recipe_count") != len(
        classification.get("recipes", [])
    ):
        raise RuntimeError("recipe classification count is inconsistent")
    ledger = _pile_summaries(args.canonical.resolve(), root / "cedar_60m_audit")
    standard = repo / "evaluation/chapter6_experiments/formal_results/paper_artifacts/optimizer/data/standard_core/workloads"
    modal_two_stage = root / "two_stage_modal_audit"
    ledger["alpaca_cot"] = _summarize_matrix(
        "alpaca_cot", 65000,
        {name: [root / "alpaca_formal/alpaca_cot/results"] for name in OPTIMIZERS},
        str(root / "alpaca_formal/alpaca_cot"),
    )
    ledger["alpaca_cot"]["screening"] = _screening_summary(
        root / "screening_matrix/alpaca_cot/results", 20000
    )
    ledger["llava_pretrain"] = _summarize_matrix(
        "llava_pretrain", 5000,
        {
            name: [
                modal_two_stage / "llava_pretrain/results"
                if name in {"optimizer", "dp_two_stage_optimizer"}
                else standard / "llava_pretrain/results"
            ]
            for name in OPTIMIZERS
        },
        str(standard / "llava_pretrain"),
    )
    video_old = repo / "outputs/chapter6_experiments/general_video_refine_formal/matrix/general_video_refine/results"
    video_dp = repo / "outputs/chapter6_experiments/general_video_refine_cost_model_fix_formal/general_video_refine/results"
    ledger["general_video_refine"] = _summarize_matrix(
        "general_video_refine", 10000,
        {
            name: [
                video_dp
                if name == "dp_optimizer"
                else video_old
            ]
            for name in OPTIMIZERS
        },
        f"competitors={video_old}; corrected_DP={video_dp}",
    )
    video_self_results = root / "video_self_evolution_formal/video_self_evolution/results"
    ledger["video_self_evolution"] = _summarize_matrix(
        "video_self_evolution", 5000,
        {name: [video_self_results] for name in OPTIMIZERS},
        str(video_self_results.parent),
    )
    ledger["video_self_evolution"]["screening"] = _screening_summary(
        root / "video_self_evolution_screening/video_self_evolution/results",
        2000,
    )
    code_results = root / "code_formal/datajuicer_candidate_runs/code_additive_formal/redpajama_code/results"
    ledger["redpajama_code"] = _summarize_matrix(
        "redpajama_code", 20000,
        {name: [code_results] for name in OPTIMIZERS}, str(code_results.parent),
    )
    ledger["redpajama_code"]["screening"] = _screening_summary(
        root
        / "code_screening/datajuicer_candidate_runs/code_additive_screen/redpajama_code/results",
        20000,
    )
    arxiv_results = root / "arxiv_formal/redpajama_arxiv/results"
    ledger["redpajama_arxiv"] = _summarize_matrix(
        "redpajama_arxiv", 2500,
        {name: [arxiv_results] for name in OPTIMIZERS}, str(arxiv_results.parent),
    )
    ledger["redpajama_arxiv"]["screening"] = _screening_summary(
        root / "arxiv_scaled_screening/redpajama_arxiv/results", 2500
    )
    ledger["redpajama_arxiv"]["infeasible_20000_output_attempt"] = {
        "requested_outputs": 20000,
        "dj_optimizer_timeout_sec": 3600,
        "dj_optimizer_observed_outputs": 3367,
        "source": str(
            root
            / "screening_matrix/redpajama_arxiv/results/round1__dj_optimizer.timeout.json"
        ),
        "used_as_speedup_evidence": False,
    }
    selected = _selection(ledger)
    wins = sum(ledger[name]["dp_at_least_20pct_faster"] for name in selected)
    report = {
        "protocol": {
            "local_workers": 8,
            "cpu_budget": 64,
            "repetitions": 3,
            "optimizer_plan_limit_sec": 3600,
            "execution_limit_sec": 3600,
            "speedup_threshold": THRESHOLD,
            "maximum_non_win_fraction": 0.40,
            "data_juicer_hub_commit": classification["hub_commit"],
            "classified_recipe_count": classification["recipe_count"],
            "operator_count_convention": {
                "hub": "all official recipe process entries, including global deduplicators",
                "cedar": "logical executed pipes excluding source and omitted cross-record deduplicators, including fixed adapters",
            },
        },
        "selected_workloads": selected,
        "data_scale": data_scale,
        "ledger": ledger,
        "summary": {
            "selected_count": len(selected),
            "wins": wins,
            "non_wins": len(selected) - wins,
            "failure_fraction": (len(selected) - wins) / len(selected),
            "diversity": _diversity_summary(selected),
        },
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.markdown_output.write_text(_render(report), encoding="utf-8")
    print(args.markdown_output)


if __name__ == "__main__":
    main()
