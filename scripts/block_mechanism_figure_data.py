"""Assemble the §3.1 mechanism-figure data from the corrected measurements.

Sources (all inside one result directory):

  service_summary.csv              primary run, per-round, explicit units
  service_summary_repeats.csv      cross-round summary
  repeat*/service_summary.csv      additional full repeats
  control_interleaved/             3 rounds, rotated config order, wall+CPU
  control_samecore/                same, all remote actors on one CPU
  instrument_overhead_{full,none}/ timing on/off
  wrapper_overhead.json            standalone wrapper cost (reported apart)
  cedar_cost_breakdown.json        Cedar model intermediates + provenance
  pipeline_results*.csv            full-pipeline cells

Every per-record number is derived once from per-batch sums
(``value_per_batch / source_records``); the identity
``compute + other_overhead == elapsed`` is asserted for every batch.

Usage:
  python -u scripts/block_mechanism_figure_data.py --run-dir outputs/<run>
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


def load_rounds(run_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Per-config list of per-round summaries from every measurement directory."""
    sources = [("primary", run_dir)]
    for name in ("repeat2", "control_interleaved", "control_samecore",
                 "instrument_overhead_full", "instrument_overhead_none"):
        candidate = run_dir / name
        if (candidate / "service_summary.csv").exists():
            sources.append((name, candidate))
    by_config: Dict[str, List[Dict[str, Any]]] = {}
    for label, directory in sources:
        rows = read_csv(directory / "service_summary.csv")
        header = set(rows[0]) if rows else set()
        for row in rows:
            if "elapsed_ms_per_record_mean" not in header:
                continue
            config = row["config"]
            by_config.setdefault(config, []).append(
                {
                    "source": label,
                    "round": int(row.get("round", 1) or 1),
                    "batches": int(row["measured_batches"]),
                    "records": int(row["measured_records"]),
                    "elapsed_ms_per_record": float(row["elapsed_ms_per_record_mean"]),
                    "compute_ms_per_record": float(
                        row["compute_sum_ms_per_record_mean"]
                    ),
                    "other_ms_per_record": float(
                        row["other_overhead_ms_per_record_mean"]
                    ),
                    "cpu_ms_per_record": float(
                        row.get("cpu_sum_ms_per_record_mean") or 0.0
                    ),
                    "elapsed_ms_per_batch": float(
                        row.get("elapsed_ms_per_batch_mean") or 0.0
                    ),
                    "elapsed_ms_per_batch_p50": float(
                        row.get("elapsed_ms_per_batch_p50") or 0.0
                    ),
                    "elapsed_ms_per_batch_p90": float(
                        row.get("elapsed_ms_per_batch_p90") or 0.0
                    ),
                    "members": {
                        key[len("compute_") : -len("_ms_per_record_mean")]: float(
                            value
                        )
                        for key, value in row.items()
                        if key.startswith("compute_")
                        and key.endswith("_ms_per_record_mean")
                    },
                    "identity_error_ms_per_batch_max": float(
                        row.get("identity_error_ms_per_batch_max") or 0.0
                    ),
                }
            )
    return by_config


def summarise(rounds: List[Dict[str, Any]]) -> Dict[str, Any]:
    totals = [r["elapsed_ms_per_record"] for r in rounds]
    computes = [r["compute_ms_per_record"] for r in rounds]
    others = [r["other_ms_per_record"] for r in rounds]
    member_names = sorted({name for r in rounds for name in r["members"]})
    return {
        "rounds": len(rounds),
        "batches": sum(r["batches"] for r in rounds),
        "records": sum(r["records"] for r in rounds),
        "elapsed_ms_per_record_mean": mean(totals),
        "elapsed_ms_per_record_stdev": stdev(totals),
        "compute_ms_per_record_mean": mean(computes),
        "compute_ms_per_record_stdev": stdev(computes),
        "other_ms_per_record_mean": mean(others),
        "other_ms_per_record_stdev": stdev(others),
        "members_ms_per_record_mean": {
            name: mean([r["members"][name] for r in rounds if name in r["members"]])
            for name in member_names
        },
        "per_round": rounds,
        "identity_error_ms_per_batch_max": max(
            (r["identity_error_ms_per_batch_max"] for r in rounds), default=0.0
        ),
    }


def ratio(new: Optional[float], base: Optional[float]) -> Optional[float]:
    if new is None or base in (None, 0):
        return None
    return new / base


def build(run_dir: Path) -> Dict[str, Any]:
    rounds = load_rounds(run_dir)

    def pick(config: str, *sources: str) -> Dict[str, Any]:
        entries = [
            r for r in rounds.get(config, []) if r["source"] in sources
        ]
        return summarise(entries) if entries else {}

    primary = {
        config: pick(config, "primary", "repeat2")
        for config in ("L-U", "L-F", "R-U", "R-F")
    }
    interleaved = {
        config: pick(config, "control_interleaved")
        for config in ("L-U", "L-F", "R-U", "R-F")
    }
    samecore = {
        config: pick(config, "control_samecore")
        for config in ("L-U", "L-F", "R-U", "R-F")
    }
    instrument = {
        label: {
            config: pick(config, f"instrument_overhead_{label}")
            for config in ("L-U", "R-F")
        }
        for label in ("full", "none")
    }
    wrapper_path = run_dir / "wrapper_overhead.json"
    wrapper = json.loads(wrapper_path.read_text()) if wrapper_path.exists() else {}

    retention: Dict[str, Any] = {}
    for backend, unfused, fused in (("local", "L-U", "L-F"), ("ray", "R-U", "R-F")):
        u, f = primary.get(unfused, {}), primary.get(fused, {})
        if not u or not f:
            continue
        retention[backend] = {
            "elapsed_ratio_fused_over_unfused": ratio(
                f["elapsed_ms_per_record_mean"], u["elapsed_ms_per_record_mean"]
            ),
            "compute_ratio_fused_over_unfused": ratio(
                f["compute_ms_per_record_mean"], u["compute_ms_per_record_mean"]
            ),
            "other_ratio_fused_over_unfused": ratio(
                f["other_ms_per_record_mean"], u["other_ms_per_record_mean"]
            ),
            "interleaved_elapsed_ratio": ratio(
                interleaved.get(fused, {}).get("elapsed_ms_per_record_mean"),
                interleaved.get(unfused, {}).get("elapsed_ms_per_record_mean"),
            ),
            "samecore_elapsed_ratio": ratio(
                samecore.get(fused, {}).get("elapsed_ms_per_record_mean"),
                samecore.get(unfused, {}).get("elapsed_ms_per_record_mean"),
            ),
        }

    cedar_path = run_dir / "cedar_cost_breakdown.json"
    cedar = json.loads(cedar_path.read_text()) if cedar_path.exists() else {}
    model_configs = cedar.get("configs", {})
    cedar_ratios: Dict[str, Any] = {}
    for backend, unfused, fused in (("local", "L-U_w1", "L-F_w1"),
                                    ("ray", "R-U_w1", "R-F_w1")):
        if unfused not in model_configs or fused not in model_configs:
            continue
        u = model_configs[unfused]
        f = model_configs[fused]
        entry: Dict[str, Any] = {
            "block_cost_unfused_ms_per_sample": u["fused_block_cost_ms_per_sample"]
            if not u["plan_is_fused"] else u["fused_block_cost_ms_per_sample"],
            "block_cost_fused_ms_per_sample": f["fused_block_cost_ms_per_sample"],
            "full_plan_cost_ratio_fused_over_unfused": ratio(
                f["full_plan_cost_ms_per_sample"],
                u["full_plan_cost_ms_per_sample"],
            ),
            "rho_io_ratio": f["rho_io_ratio"],
            "provenance": {
                "unfused_plan": u["plan_provenance"],
                "fused_plan": f["plan_provenance"],
            },
        }
        base = entry["block_cost_unfused_ms_per_sample"]
        cached = entry["block_cost_fused_ms_per_sample"]
        entry["block_retention_ratio"] = ratio(cached, base)
        if not base or not cached:
            entry["block_retention_ratio"] = None
            entry["annotation"] = "N/A (Cedar clips the offloaded block to 0)"
        cedar_ratios[backend] = entry

    pipeline: Dict[str, Any] = {}
    for extra in sorted(run_dir.glob("pipeline_results*.csv")):
        for row in read_csv(extra):
            key = f"{row['config']}_w{row['workers']}"
            payload = pipeline.setdefault(
                key,
                {
                    "config": row["config"],
                    "workers": int(row["workers"]),
                    "repeats": [],
                    "source_file": extra.name,
                },
            )
            payload["repeats"].append(
                {
                    "repeat": int(row["repeat"]),
                    "returncode": int(row["returncode"]),
                    "num_samples": int(row["num_samples"]) if row.get("num_samples") else None,
                    "perf_time_sec": float(row["perf_time_sec"]) if row.get("perf_time_sec") else None,
                    "throughput_samples_per_sec": (
                        float(row["throughput_samples_per_sec"])
                        if row.get("throughput_samples_per_sec") else None
                    ),
                    "setup_time_sec": (
                        float(row["setup_time_sec"]) if row.get("setup_time_sec") else None
                    ),
                    "wall_time_sec": float(row["wall_time_sec"]) if row.get("wall_time_sec") else None,
                }
            )
    for key, payload in pipeline.items():
        rates = [
            r["throughput_samples_per_sec"] for r in payload["repeats"]
            if r["throughput_samples_per_sec"]
        ]
        payload["throughput_mean"] = mean(rates)
        payload["throughput_stdev"] = stdev(rates)
        payload["ok_repeats"] = sum(
            1 for r in payload["repeats"] if r["returncode"] == 0
        )
        payload["failed_repeats"] = sum(
            1 for r in payload["repeats"] if r["returncode"] != 0
        )
        probe = run_dir / f"actor_probe_{key}.json"
        payload["ray_actors"] = (
            json.loads(probe.read_text())["peak_alive_actors"]
            if probe.exists() else None
        )
    return {
        "units": {
            "service": "ms per source record (batch value / source_records)",
            "compute": "sum over a batch's records of wall time measured around "
                       "each member callable",
            "other_overhead": "per-batch end-to-end time minus that batch's "
                              "summed member wall time; not pure network time",
        },
        "service": {
            "primary": primary,
            "interleaved": interleaved,
            "samecore": samecore,
            "instrument": instrument,
            "retention": retention,
            "cedar_model": cedar_ratios,
            "wrapper_overhead": wrapper,
        },
        "pipeline": pipeline,
        "cedar_model": cedar,
    }


def render(data: Dict[str, Any]) -> str:
    lines = ["# SimCLRv2 B/H/J 机制图数据（修正版）", ""]
    lines += ["## 实验 A：串行服务时间（ms/source-record）", ""]
    lines.append("| 配置 | 计算 B | 计算 H | 计算 J | 计算合计 | 其他开销 | 总时间 | 轮数 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for config, payload in data["service"]["primary"].items():
        if not payload:
            continue
        members = payload["members_ms_per_record_mean"]
        lines.append(
            "| %s | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %d |" % (
                config,
                members.get("B_blur", float("nan")),
                members.get("H_flip", float("nan")),
                members.get("J_jitter", float("nan")),
                payload["compute_ms_per_record_mean"],
                payload["other_ms_per_record_mean"],
                payload["elapsed_ms_per_record_mean"],
                payload["rounds"],
            )
        )
    lines += ["", "## 控制条件（交错顺序 / 同核 / 插桩）", ""]
    lines.append("| 配置 | 主测量总时间 | 交错3轮总时间 | 同核总时间 | 主测量计算 | 主测量其他 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for config in data["service"]["primary"]:
        primary = data["service"]["primary"][config]
        inter = data["service"]["interleaved"].get(config, {})
        same = data["service"]["samecore"].get(config, {})
        if not primary:
            continue
        lines.append(
            "| %s | %.4f | %s | %s | %.4f | %.4f |" % (
                config,
                primary["elapsed_ms_per_record_mean"],
                ("%.4f" % inter["elapsed_ms_per_record_mean"]) if inter else "n/a",
                ("%.4f" % same["elapsed_ms_per_record_mean"]) if same else "n/a",
                primary["compute_ms_per_record_mean"],
                primary["other_ms_per_record_mean"],
            )
        )
    lines += ["", "## U/F 保留比例", ""]
    lines.append("| 后端 | 计算 | 其他开销 | 总时间 | Cedar 模型块成本比 | 备注 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
    for backend, payload in data["service"]["retention"].items():
        model = data["service"]["cedar_model"].get(backend, {})
        retention = model.get("block_retention_ratio")
        note = model.get("annotation", "")
        lines.append(
            "| %s | %.4f | %.4f | %.4f | %s | %s |" % (
                backend,
                payload["compute_ratio_fused_over_unfused"],
                payload["other_ratio_fused_over_unfused"],
                payload["elapsed_ratio_fused_over_unfused"],
                ("%.4f" % retention) if retention is not None else "N/A",
                note,
            )
        )
    lines += ["", "## 实验 B：完整流水线（records/s）", ""]
    lines.append("| 配置 | W | 成功轮 | 吞吐均值 | 标准差 | 实测 Ray actor |")
    lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
    for key, payload in sorted(data["pipeline"].items()):
        if not payload.get("throughput_mean"):
            continue
        actors = payload.get("ray_actors")
        lines.append(
            "| %s | %d | %d | %.2f | %.2f | %s |" % (
                payload["config"], payload["workers"], payload["ok_repeats"],
                payload["throughput_mean"], payload["throughput_stdev"] or 0.0,
                actors if actors is not None else "n/a (local plan)",
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    data = build(args.run_dir)
    (args.run_dir / "figure_data.json").write_text(json.dumps(data, indent=2))
    markdown = render(data)
    (args.run_dir / "figure_data.md").write_text(markdown)
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
