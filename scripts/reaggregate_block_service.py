"""Re-aggregate the v1 block-service CSVs into explicit per-batch units.

The v1 harness stored, for each batch:

    elapsed_ms            end-to-end time of the batch            (ms per batch)
    compute_<member>_ms   mean over the batch's records of one
                          member's wall time                      (ms per record)
    compute_sum_ms        sum of the three per-record means       (ms per record)
    noncompute_residual_ms = elapsed_ms - compute_sum_ms          (mixed units!)

The batch-level member *sums* are exactly recoverable as
``source_records * compute_<member>_ms``, so the corrected decomposition needs
no re-run.  This script writes:

    legacy_v1/            the original files plus INVALID.md
    service_raw.csv       corrected per-batch rows (explicit units)
    service_summary.csv   per-round summary with the compute+overhead identity
    service_summary_repeats.csv  cross-round summary

Usage:
  python -u scripts/reaggregate_block_service.py --run-dir outputs/<run>
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
from pathlib import Path
from typing import Any, Dict, List

MEMBER_PREFIX = "compute_"
MEMBERS = ("B_blur", "H_flip", "J_jitter")

INVALID_NOTE = """# 这些文件的口径有误，已由上层目录的修正版取代

`service_raw.csv`（v1）中：

- `elapsed_ms` 是**每批**端到端时间；
- `compute_<member>_ms` 是**每记录**的批内均值；
- `compute_sum_ms` 是三项每记录均值之和（每记录口径）；
- `noncompute_residual_ms = elapsed_ms - compute_sum_ms` 把每批时间与每记录时间相减，**单位不一致**。

批级成员求和可精确恢复为 `source_records × compute_<member>_ms`，因此修正版不需要重跑：
修正后的分解见上层目录 `service_raw.csv` / `service_summary.csv` /
`service_summary_repeats.csv`（每批先求和、再相减，每记录只除一次记录数）。
本目录仅作留档。
"""


def _mean(values: List[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _stdev(values: List[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def convert(run_dir: Path) -> Dict[str, Any]:
    raw_path = run_dir / "service_raw.csv"
    rows = list(csv.DictReader(raw_path.open()))
    if not rows:
        raise SystemExit(f"{raw_path} is empty")
    header = list(rows[0])
    if "elapsed_ms_per_batch" in header:
        print(f"{raw_path} already uses the corrected schema; nothing to do")
        return {}

    legacy_dir = run_dir / "legacy_v1"
    legacy_dir.mkdir(exist_ok=True)
    shutil.copy2(raw_path, legacy_dir / "service_raw.csv")
    for name in ("service_summary.csv", "service_summary_repeats.csv"):
        candidate = run_dir / name
        if candidate.exists():
            shutil.copy2(candidate, legacy_dir / name)
    (legacy_dir / "INVALID.md").write_text(INVALID_NOTE)

    corrected: List[Dict[str, Any]] = []
    for row in rows:
        records = int(row["source_records"])
        elapsed = float(row["elapsed_ms"])
        out: Dict[str, Any] = {
            "config": row["config"],
            "round": int(row.get("repeat", row.get("round", 1)) or 1),
            "order_index": row.get("order_index", ""),
            "batch_id": int(row["batch_id"]),
            "source_records": records,
            "elapsed_ms_per_batch": elapsed,
        }
        member_batch = {}
        for member in MEMBERS:
            per_record = float(row[f"{MEMBER_PREFIX}{member}_ms"])
            member_batch[member] = records * per_record
            out[f"compute_{member}_ms_per_batch"] = member_batch[member]
            out[f"compute_{member}_ms_per_record"] = per_record
        compute_batch = sum(member_batch.values())
        out["compute_sum_ms_per_batch"] = compute_batch
        out["compute_sum_ms_per_record"] = compute_batch / records
        out["other_overhead_ms_per_batch"] = elapsed - compute_batch
        out["other_overhead_ms_per_record"] = (
            elapsed - compute_batch
        ) / records
        out["elapsed_ms_per_record"] = elapsed / records
        out["schema_source"] = "legacy_v1_recovered"
        corrected.append(out)

    fields = list(corrected[0].keys())
    with raw_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in corrected:
            writer.writerow(row)

    # per-round summary
    summary_fields = [
        "config", "round", "measured_batches", "measured_records",
        "source_records",
        *[f"compute_{m}_ms_per_record_mean" for m in MEMBERS],
        "compute_sum_ms_per_record_mean", "compute_sum_ms_per_record_stdev",
        "other_overhead_ms_per_record_mean",
        "other_overhead_ms_per_record_stdev",
        "elapsed_ms_per_record_mean", "elapsed_ms_per_record_stdev",
        "elapsed_ms_per_batch_mean", "elapsed_ms_per_batch_stdev",
        "elapsed_ms_per_batch_p50", "elapsed_ms_per_batch_p90",
        "compute_sum_ms_per_batch_mean", "other_overhead_ms_per_batch_mean",
        "identity_error_ms_per_batch_max", "schema_source",
    ]
    summary_rows = []
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in corrected:
        groups.setdefault((row["config"], row["round"]), []).append(row)
    with (run_dir / "service_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for (config, round_index), group in groups.items():
            elapsed = [r["elapsed_ms_per_batch"] for r in group]
            compute = [r["compute_sum_ms_per_batch"] for r in group]
            other = [r["other_overhead_ms_per_batch"] for r in group]
            payload = {
                "config": config,
                "round": round_index,
                "measured_batches": len(group),
                "measured_records": sum(r["source_records"] for r in group),
                "source_records": group[0]["source_records"],
                **{
                    f"compute_{m}_ms_per_record_mean": _mean(
                        [r[f"compute_{m}_ms_per_record"] for r in group]
                    )
                    for m in MEMBERS
                },
                "compute_sum_ms_per_record_mean": _mean(
                    [r["compute_sum_ms_per_record"] for r in group]
                ),
                "compute_sum_ms_per_record_stdev": _stdev(
                    [r["compute_sum_ms_per_record"] for r in group]
                ),
                "other_overhead_ms_per_record_mean": _mean(
                    [r["other_overhead_ms_per_record"] for r in group]
                ),
                "other_overhead_ms_per_record_stdev": _stdev(
                    [r["other_overhead_ms_per_record"] for r in group]
                ),
                "elapsed_ms_per_record_mean": _mean(
                    [r["elapsed_ms_per_record"] for r in group]
                ),
                "elapsed_ms_per_record_stdev": _stdev(
                    [r["elapsed_ms_per_record"] for r in group]
                ),
                "elapsed_ms_per_batch_mean": _mean(elapsed),
                "elapsed_ms_per_batch_stdev": _stdev(elapsed),
                "elapsed_ms_per_batch_p50": _quantile(elapsed, 0.5),
                "elapsed_ms_per_batch_p90": _quantile(elapsed, 0.9),
                "compute_sum_ms_per_batch_mean": _mean(compute),
                "other_overhead_ms_per_batch_mean": _mean(other),
                "identity_error_ms_per_batch_max": max(
                    abs(c + o - e) for c, o, e in zip(compute, other, elapsed)
                ),
                "schema_source": "legacy_v1_recovered",
            }
            writer.writerow(payload)
            summary_rows.append(payload)

    # cross-round summary
    with (run_dir / "service_summary_repeats.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["config", "rounds", "elapsed_ms_per_record_mean",
             "elapsed_ms_per_record_stdev", "compute_ms_per_record_mean",
             "compute_ms_per_record_stdev",
             "other_overhead_ms_per_record_mean",
             "other_overhead_ms_per_record_stdev", "schema_source"]
        )
        by_config: Dict[str, List[Dict[str, Any]]] = {}
        for payload in summary_rows:
            by_config.setdefault(payload["config"], []).append(payload)
        for config, payloads in sorted(by_config.items()):
            totals = [p["elapsed_ms_per_record_mean"] for p in payloads]
            computes = [p["compute_sum_ms_per_record_mean"] for p in payloads]
            others = [p["other_overhead_ms_per_record_mean"] for p in payloads]
            writer.writerow(
                [config, len(payloads), _mean(totals), _stdev(totals),
                 _mean(computes), _stdev(computes), _mean(others),
                 _stdev(others), "legacy_v1_recovered"]
            )

    result = {
        "run_dir": str(run_dir),
        "batches": len(corrected),
        "configs": sorted({r["config"] for r in corrected}),
        "identity_error_max_ms": max(
            abs(r["compute_sum_ms_per_batch"]
                + r["other_overhead_ms_per_batch"]
                - r["elapsed_ms_per_batch"])
            for r in corrected
        ),
    }
    (run_dir / "reaggregation_report.json").write_text(
        json.dumps(result, indent=2)
    )
    print(json.dumps(result, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--include-subdirs", action="store_true",
        help="also convert repeat*/ and control_*/ directories",
    )
    args = parser.parse_args()
    convert(args.run_dir)
    if args.include_subdirs:
        for sub in sorted(args.run_dir.glob("repeat*")):
            if sub.is_dir() and (sub / "service_raw.csv").exists():
                convert(sub)
        for sub in sorted(args.run_dir.glob("control_*")):
            if sub.is_dir() and (sub / "service_raw.csv").exists():
                convert(sub)
        for sub in sorted(args.run_dir.glob("instrument_overhead_*")):
            if sub.is_dir() and (sub / "service_raw.csv").exists():
                convert(sub)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
