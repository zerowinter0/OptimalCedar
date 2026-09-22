"""Assemble the data the §3.1 mechanism figure needs.

Reads the two experiment outputs and the Cedar model export and writes
``figure_data.json`` / ``figure_data.md``:

  * service time per configuration split into member compute and residual,
  * U/F retention ratios (total, compute, residual) per backend,
  * Cedar's own retention ratio for the same pair, with N/A when the block is
    clipped to zero,
  * pipeline throughput per configuration and worker count.

Usage (inside the container):
  python -u scripts/block_mechanism_figure_data.py --run-dir outputs/<run>
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def _mean(values: List[float]) -> Optional[float]:
    return statistics.fmean(values) if values else None


def build(run_dir: Path) -> Dict[str, Any]:
    raw = _read_csv(run_dir / "service_raw.csv")
    summary = _read_csv(run_dir / "service_summary.csv")
    pivot = []
    for extra in sorted(run_dir.glob("pipeline_results*.csv")):
        pivot.extend(_read_csv(extra))
    verification_path = run_dir / "service_verification.json"
    verification = (
        json.loads(verification_path.read_text())
        if verification_path.exists()
        else {}
    )
    meta_path = run_dir / "service_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    cedar_path = run_dir / "cedar_cost_breakdown.json"
    cedar = json.loads(cedar_path.read_text()) if cedar_path.exists() else {}

    configs: Dict[str, Any] = {}
    for row in summary:
        config = row["config"]
        rows = [item for item in raw if item["config"] == config]
        records_per_batch = int(rows[0]["source_records"]) if rows else 4

        def per_record(column: str) -> Optional[float]:
            values = [
                float(item[column]) / records_per_batch
                for item in rows
                if item.get(column) not in (None, "")
            ]
            return _mean(values)

        def per_record_quantile(column: str, quantile: float) -> Optional[float]:
            values = sorted(
                float(item[column]) / records_per_batch
                for item in rows
                if item.get(column) not in (None, "")
            )
            if not values:
                return None
            index = min(len(values) - 1, int(quantile * len(values)))
            return values[index]

        compute_columns = [
            key for key in (rows[0] if rows else {})
            if key.startswith("compute_") and key.endswith("_ms")
        ]
        compute_columns = [c for c in compute_columns if c != "compute_sum_ms"]
        configs[config] = {
            "batches": int(row["measured_batches"]),
            "source_records": int(row["measured_records"]),
            "actors": int(row["actors"]),
            "elapsed_ms_per_record": per_record("elapsed_ms"),
            "compute_sum_ms_per_record": per_record("compute_sum_ms"),
            "residual_ms_per_record": per_record("noncompute_residual_ms"),
            "elapsed_ms_per_record_median": per_record_quantile("elapsed_ms", 0.5),
            "elapsed_ms_per_record_p90": per_record_quantile("elapsed_ms", 0.9),
            "compute_sum_ms_per_record_median": per_record_quantile("compute_sum_ms", 0.5),
            "residual_ms_per_record_median": per_record_quantile("noncompute_residual_ms", 0.5),
            "residual_ms_per_record_p90": per_record_quantile("noncompute_residual_ms", 0.9),
            "member_compute_ms_per_record": {
                column[len("compute_"):-len("_ms")]: per_record(column)
                for column in compute_columns
            },
            "elapsed_ms_per_batch_mean": float(row["elapsed_ms_mean"]),
            "elapsed_ms_per_batch_stdev": float(row["elapsed_ms_stdev"]),
            "pilot_ms_per_batch": float(row["pilot_ms_per_batch"]),
        }

    def ratio(fast: Optional[float], slow: Optional[float]) -> Optional[float]:
        if fast is None or slow is None or slow == 0:
            return None
        return fast / slow

    # Cross-repeat statistics: the primary run plus any repeat*/ directories.
    repeat_sources = [(1, summary)]
    for extra_dir in sorted(run_dir.glob("repeat*")):
        extra = _read_csv(extra_dir / "service_summary.csv")
        if extra:
            repeat_sources.append((len(repeat_sources) + 1, extra))
    repeats: Dict[str, Any] = {}
    for config in configs:
        rows = []
        for index, table in repeat_sources:
            match = next((r for r in table if r["config"] == config), None)
            if match is None:
                continue
            records = int(match["measured_records"]) or 1
            batches = int(match["measured_batches"]) or 1
            rows.append(
                {
                    "repeat": index,
                    "elapsed_ms_per_record": float(match["elapsed_ms_mean"]) * batches / records,
                    "compute_ms_per_record": float(match["compute_sum_ms_mean"]) * batches / records,
                    "residual_ms_per_record": float(match["residual_ms_mean"]) * batches / records,
                }
            )
        if not rows:
            continue
        repeats[config] = {
            key: {
                "mean": _mean([row[key] for row in rows]),
                "stdev": (
                    statistics.stdev([row[key] for row in rows])
                    if len(rows) > 1
                    else 0.0
                ),
                "per_repeat": [row[key] for row in rows],
            }
            for key in ("elapsed_ms_per_record", "compute_ms_per_record",
                        "residual_ms_per_record")
        }
        repeats[config]["repeats"] = len(rows)

    if repeats:
        repeat_csv = run_dir / "service_summary_repeats.csv"
        with repeat_csv.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["config", "repeats", "elapsed_ms_per_record_mean",
                 "elapsed_ms_per_record_stdev", "compute_ms_per_record_mean",
                 "compute_ms_per_record_stdev", "residual_ms_per_record_mean",
                 "residual_ms_per_record_stdev"]
            )
            for config, payload in repeats.items():
                writer.writerow(
                    [
                        config,
                        payload["repeats"],
                        payload["elapsed_ms_per_record"]["mean"],
                        payload["elapsed_ms_per_record"]["stdev"],
                        payload["compute_ms_per_record"]["mean"],
                        payload["compute_ms_per_record"]["stdev"],
                        payload["residual_ms_per_record"]["mean"],
                        payload["residual_ms_per_record"]["stdev"],
                    ]
                )

    retention = {}
    for backend, unfused, fused in (("local", "L-U", "L-F"), ("ray", "R-U", "R-F")):
        if unfused not in configs or fused not in configs:
            continue
        retention[backend] = {
            "total_time_ratio_fused_over_unfused": ratio(
                configs[fused]["elapsed_ms_per_record"],
                configs[unfused]["elapsed_ms_per_record"],
            ),
            "compute_ratio_fused_over_unfused": ratio(
                configs[fused]["compute_sum_ms_per_record"],
                configs[unfused]["compute_sum_ms_per_record"],
            ),
            "residual_ratio_fused_over_unfused": ratio(
                configs[fused]["residual_ms_per_record"],
                configs[unfused]["residual_ms_per_record"],
            ),
        }

    cedar_ratios: Dict[str, Any] = {}
    model_configs = cedar.get("configs", {})
    for backend, unfused, fused in (("local", "L-U_w1", "L-F_w1"),
                                    ("ray", "R-U_w1", "R-F_w1")):
        if unfused not in model_configs or fused not in model_configs:
            continue
        cached = model_configs[fused]["block_cost_ms_per_sample"]
        total_unfused = model_configs[unfused]["block_cost_ms_per_sample"]
        entry: Dict[str, Any] = {
            "block_cost_unfused_ms_per_sample": total_unfused,
            "block_cost_fused_ms_per_sample": cached,
            "block_retention_ratio": ratio(cached, total_unfused),
            "full_plan_cost_ratio_fused_over_unfused": ratio(
                model_configs[fused]["full_plan_cost_ms_per_sample"],
                model_configs[unfused]["full_plan_cost_ms_per_sample"],
            ),
            "member_cost_sum_ms_per_sample": model_configs[unfused][
                "member_cost_sum_ms_per_sample"
            ],
            "block_cost_ms_per_sample": cached,
            "block_cost_unfused_ms_per_sample": total_unfused,
            "rho_io_fused_over_base": model_configs[unfused][
                "rho_io_fused_over_base"
            ],
        }
        if cached == 0 or total_unfused == 0:
            entry["block_retention_ratio"] = None
            entry["annotation"] = "N/A (Cedar clips the offloaded block to 0)"
        cedar_ratios[backend] = entry

    pipeline: Dict[str, Any] = {}
    for row in pivot:
        key = f"{row['config']}_w{row['workers']}"
        throughput = row.get("throughput_samples_per_sec")
        pipeline.setdefault(key, {"config": row["config"], "workers": int(row["workers"]), "repeats": []})
        pipeline[key]["repeats"].append(
            {
                "repeat": int(row["repeat"]),
                "returncode": int(row["returncode"]),
                "num_samples": int(row["num_samples"]) if row.get("num_samples") else None,
                "perf_time_sec": float(row["perf_time_sec"]) if row.get("perf_time_sec") else None,
                "throughput_samples_per_sec": float(throughput) if throughput else None,
                "setup_time_sec": float(row["setup_time_sec"]) if row.get("setup_time_sec") else None,
                "actors": row.get("actors"),
            }
        )
    for key, payload in pipeline.items():
        rates = [r["throughput_samples_per_sec"] for r in payload["repeats"] if r["throughput_samples_per_sec"]]
        payload["throughput_mean"] = _mean(rates)
        payload["throughput_stdev"] = statistics.stdev(rates) if len(rates) > 1 else 0.0
        payload["ok_repeats"] = sum(1 for r in payload["repeats"] if r["returncode"] == 0)
        payload["failed_repeats"] = sum(1 for r in payload["repeats"] if r["returncode"] != 0)
        # Ray stage actors: observed with block_actor_probe.py where available,
        # otherwise derived from the plan (W replicas x Ray stages).
        probe_path = run_dir / f"actor_probe_{key}.json"
        if probe_path.exists():
            payload["ray_actors_observed"] = json.loads(
                probe_path.read_text()
            )["peak_alive_actors"]
        else:
            ray_stages = 1 if payload["config"].endswith("F") else 3
            payload["ray_actors_derived"] = (
                payload["workers"] * ray_stages
                if payload["config"].startswith("R")
                else 0
            )

    return {
        "service": {
            "configs": configs,
            "repeats": repeats,
            "retention": retention,
            "cedar_model": cedar_ratios,
            "verification": verification,
            "meta": meta,
        },
            "pipeline": pipeline,
        "cedar_model": cedar,
    }


def render_markdown(data: Dict[str, Any]) -> str:
    service = data["service"]
    lines = ["# SimCLRv2 B/H/J 机制图数据", ""]
    lines.append("## 实验 A：串行服务时间（ms/source-record）")
    lines.append("")
    lines.append("| 配置 | 后端 | 组织 | 计算 B | 计算 H | 计算 J | 计算合计 | 交接/框架余项 | 总计 |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    labels = {
        "L-U": ("local", "三个独立阶段"),
        "L-F": ("local", "一个融合阶段"),
        "R-U": ("ray", "三个 actor 阶段"),
        "R-F": ("ray", "一个融合 actor"),
    }
    for config, label in labels.items():
        payload = service["configs"].get(config)
        if not payload:
            continue
        members = payload["member_compute_ms_per_record"]
        lines.append(
            "| %s | %s | %s | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f |" % (
                config, label[0], label[1],
                members.get("B_blur", float("nan")),
                members.get("H_flip", float("nan")),
                members.get("J_jitter", float("nan")),
                payload["compute_sum_ms_per_record"],
                payload["residual_ms_per_record"],
                payload["elapsed_ms_per_record"],
            )
        )
    lines += ["", "## U/F 保留比例", ""]
    lines.append("| 后端 | 计算保留 | 余项保留 | 总时间保留 | Cedar 模型（块成本） |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for backend, row in service["retention"].items():
        cedar_row = service["cedar_model"].get(backend, {})
        cedar_value = cedar_row.get("block_retention_ratio")
        cedar_text = (
            "%.4f" % cedar_value if cedar_value is not None
            else cedar_row.get("annotation", "n/a")
        )
        lines.append("| %s | %.4f | %.4f | %.4f | %s |" % (
            backend,
            row["compute_ratio_fused_over_unfused"],
            row["residual_ratio_fused_over_unfused"],
            row["total_time_ratio_fused_over_unfused"],
            cedar_text,
        ))
    lines += ["", "## 实验 B：完整流水线（records/s）", ""]
    lines.append("| 配置 | W | 重复 | 吞吐均值 | 标准差 | 实际 actor 数 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
    for key, payload in sorted(data["pipeline"].items()):
        if not payload.get("throughput_mean"):
            continue
        actors = payload.get("ray_actors_observed")
        actor_text = (
            "%d (probe)" % actors if actors is not None
            else "%d (derived)" % payload.get("ray_actors_derived", 0)
        )
        lines.append("| %s | %d | %d | %.2f | %.2f | %s |" % (
            payload["config"], payload["workers"], payload["ok_repeats"],
            payload["throughput_mean"], payload["throughput_stdev"],
            actor_text,
        ))
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    data = build(args.run_dir)
    (args.run_dir / "figure_data.json").write_text(
        json.dumps(data, indent=2)
    )
    (args.run_dir / "figure_data.md").write_text(render_markdown(data))
    print(render_markdown(data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
