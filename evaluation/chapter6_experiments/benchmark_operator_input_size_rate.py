#!/usr/bin/env python3
"""Measure operator rate versus real per-record input size.

The benchmark deliberately reuses one existing, unmodified Data-Juicer
pipeline.  Records flow through StackExchangeFeature in its original order;
at each operator boundary a bounded stratified reservoir stores immutable
snapshots of the legal inputs that actually reached that operator.  The
selected snapshots are then replayed directly against that operator's Python
callable so the measurement excludes source I/O and prefix execution.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import pathlib
import pickle
import random
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cedar.compose.utils import topological_sort
from cedar.pipes import FilterPipe, MapperPipe
from cedar.pipes.common import get_sizeof_data
from cedar.sources import LocalLineSource
from evaluation.pipelines.stackexchange.cedar_dataset import (
    DEFAULT_DATASET_PATH,
    StackExchangeFeature,
)


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class OperatorStage:
    position: int
    pipe_id: int
    tag: str
    name: str
    kind: str
    scaling: str
    fn: Callable[[Any], Any]


@dataclass(frozen=True)
class CapturedInput:
    input_bytes: int
    source_record_index: int
    snapshot: bytes


class LogSizeReservoir:
    """Keep a bounded deterministic sample in half-octave size buckets."""

    def __init__(self, per_bucket: int, seed: int) -> None:
        if per_bucket < 1:
            raise ValueError("per_bucket must be positive")
        self.per_bucket = per_bucket
        self._rng = random.Random(seed)
        self._buckets: Dict[int, List[CapturedInput]] = {}
        self._seen: Dict[int, int] = {}

    @staticmethod
    def bucket_for(input_bytes: int) -> int:
        if input_bytes < 1:
            raise ValueError("input_bytes must be positive")
        return math.floor(2.0 * math.log2(input_bytes))

    def offer(self, value: Any, source_record_index: int) -> None:
        input_bytes = int(get_sizeof_data(value))
        if input_bytes < 1:
            return
        snapshot = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        bucket = self.bucket_for(input_bytes)
        seen = self._seen.get(bucket, 0) + 1
        self._seen[bucket] = seen
        values = self._buckets.setdefault(bucket, [])
        captured = CapturedInput(
            input_bytes=input_bytes,
            source_record_index=source_record_index,
            snapshot=snapshot,
        )
        if len(values) < self.per_bucket:
            values.append(captured)
            return
        replacement = self._rng.randrange(seen)
        if replacement < self.per_bucket:
            values[replacement] = captured

    def select(self, max_points: int) -> List[CapturedInput]:
        if max_points < 1:
            raise ValueError("max_points must be positive")
        representatives: List[CapturedInput] = []
        for bucket, values in sorted(self._buckets.items()):
            center = (bucket + 0.5) / 2.0
            representatives.append(
                min(
                    values,
                    key=lambda item: abs(math.log2(item.input_bytes) - center),
                )
            )
        if len(representatives) <= max_points:
            return representatives
        indices = np.linspace(0, len(representatives) - 1, max_points)
        selected = []
        for index in indices:
            item = representatives[int(round(float(index)))]
            if not selected or item != selected[-1]:
                selected.append(item)
        return selected

    def metadata(self) -> Dict[str, Any]:
        return {
            "bucket_width_octaves": 0.5,
            "per_bucket_capacity": self.per_bucket,
            "seen_per_bucket": dict(sorted(self._seen.items())),
            "retained_per_bucket": {
                bucket: len(values)
                for bucket, values in sorted(self._buckets.items())
            },
        }


def build_stages(dataset_path: pathlib.Path) -> List[OperatorStage]:
    """Read the exact logical order and annotations from the real feature."""
    feature = StackExchangeFeature()
    feature.apply(LocalLineSource(str(dataset_path)))
    stages: List[OperatorStage] = []
    for pipe_id in topological_sort(feature.logical_adj_list):
        pipe = feature.logical_pipes[pipe_id]
        if not isinstance(pipe, (MapperPipe, FilterPipe)):
            continue
        if not pipe.tag:
            raise RuntimeError(f"Benchmark operator {pipe_id} has no tag")
        stages.append(
            OperatorStage(
                position=len(stages) + 1,
                pipe_id=pipe_id,
                tag=pipe.tag,
                name=pipe.name,
                kind="filter" if isinstance(pipe, FilterPipe) else "mapper",
                scaling=pipe.compute_scaling.value,
                fn=pipe.fn,
            )
        )
    if not stages:
        raise RuntimeError("StackExchangeFeature contains no operators")
    return stages


def collect_legal_inputs(
    stages: Sequence[OperatorStage],
    dataset_path: pathlib.Path,
    max_source_records: int,
    per_bucket: int,
    seed: int,
) -> tuple[Dict[str, LogSizeReservoir], Dict[str, Dict[str, int]]]:
    reservoirs = {
        stage.tag: LogSizeReservoir(per_bucket, seed + stage.position)
        for stage in stages
    }
    counts = {
        stage.tag: {"reached": 0, "passed": 0, "errors": 0}
        for stage in stages
    }
    with dataset_path.open("r", encoding="utf-8") as source:
        for source_index, line in enumerate(source):
            if source_index >= max_source_records:
                break
            value: Any = line.rstrip("\r\n")
            for stage in stages:
                counts[stage.tag]["reached"] += 1
                reservoirs[stage.tag].offer(value, source_index)
                try:
                    output = stage.fn(value)
                except Exception:
                    counts[stage.tag]["errors"] += 1
                    break
                if stage.kind == "filter":
                    if not output:
                        break
                else:
                    value = output
                counts[stage.tag]["passed"] += 1
    return reservoirs, counts


def _run_batch(stage: OperatorStage, snapshot: bytes, calls: int) -> float:
    inputs = [pickle.loads(snapshot) for _ in range(calls)]
    was_enabled = gc.isenabled()
    gc.disable()
    start = time.perf_counter_ns()
    sink = None
    for value in inputs:
        sink = stage.fn(value)
    elapsed_ns = time.perf_counter_ns() - start
    if was_enabled:
        gc.enable()
    # Retain a visible reference through the timer boundary. Python does not
    # eliminate calls, but this also keeps lazy-looking return values honest.
    if sink is NotImplemented:
        raise RuntimeError("Unexpected operator result")
    return elapsed_ns / 1e9


def benchmark_input(
    stage: OperatorStage,
    captured: CapturedInput,
    repeats: int,
    target_batch_sec: float,
    max_calls: int,
    max_batch_bytes: int,
) -> Dict[str, Any]:
    # Lazy models and regex/tokenizer caches are initialized outside measured
    # repeats. Each warmup receives a fresh legal snapshot.
    for _ in range(3):
        stage.fn(pickle.loads(captured.snapshot))

    calibration_calls = 3
    elapsed = _run_batch(stage, captured.snapshot, calibration_calls)
    seconds_per_call = max(elapsed / calibration_calls, 1e-9)
    calls = max(1, min(max_calls, math.ceil(target_batch_sec / seconds_per_call)))
    snapshot_bytes = max(1, len(captured.snapshot))
    calls = min(calls, max(1, max_batch_bytes // snapshot_bytes))

    rates = []
    durations = []
    for _ in range(repeats):
        duration = _run_batch(stage, captured.snapshot, calls)
        durations.append(duration)
        rates.append(calls / duration)
    rate_median = float(statistics.median(rates))
    return {
        "schema_version": SCHEMA_VERSION,
        "position": stage.position,
        "pipe_id": stage.pipe_id,
        "tag": stage.tag,
        "name": stage.name,
        "kind": stage.kind,
        "scaling": stage.scaling,
        "source_record_index": captured.source_record_index,
        "input_bytes": captured.input_bytes,
        "snapshot_bytes": len(captured.snapshot),
        "input_sha256": hashlib.sha256(captured.snapshot).hexdigest(),
        "calls_per_repeat": calls,
        "repeats": repeats,
        "rate_records_per_sec": rate_median,
        "rate_q1_records_per_sec": float(np.percentile(rates, 25)),
        "rate_q3_records_per_sec": float(np.percentile(rates, 75)),
        "latency_ns_per_record": 1e9 / rate_median,
        "throughput_mib_per_sec": (
            rate_median * captured.input_bytes / (1024.0 * 1024.0)
        ),
        "repeat_rates_records_per_sec": rates,
        "repeat_durations_sec": durations,
    }


def add_elasticities(
    rows: Sequence[Mapping[str, Any]],
    stage_summaries: Dict[str, Dict[str, Any]],
) -> None:
    by_tag: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        by_tag.setdefault(str(row["tag"]), []).append(row)
    for tag, values in by_tag.items():
        unique = {
            (float(value["input_bytes"]), float(value["latency_ns_per_record"]))
            for value in values
        }
        if len(unique) < 3:
            stage_summaries[tag]["latency_elasticity"] = None
            stage_summaries[tag]["fit_r_squared"] = None
            continue
        x = np.log([pair[0] for pair in sorted(unique)])
        y = np.log([pair[1] for pair in sorted(unique)])
        slope, intercept = np.polyfit(x, y, 1)
        predicted = slope * x + intercept
        residual = float(np.sum((y - predicted) ** 2))
        total = float(np.sum((y - np.mean(y)) ** 2))
        stage_summaries[tag]["latency_elasticity"] = float(slope)
        stage_summaries[tag]["fit_r_squared"] = (
            float(1.0 - residual / total) if total > 0 else 1.0
        )


def write_csv(path: pathlib.Path, rows: Sequence[Mapping[str, Any]]) -> None:
    scalar_keys = [
        key
        for key, value in rows[0].items()
        if not isinstance(value, (list, dict))
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=scalar_keys, lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in scalar_keys})


def plot_results(rows: Sequence[Mapping[str, Any]], output_dir: pathlib.Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    tags = sorted(
        {str(row["tag"]) for row in rows},
        key=lambda tag: min(
            int(row["position"]) for row in rows if row["tag"] == tag
        ),
    )
    colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(tags))))
    for color, tag in zip(colors, tags):
        values = sorted(
            (row for row in rows if row["tag"] == tag),
            key=lambda row: float(row["input_bytes"]),
        )
        x = np.array([float(row["input_bytes"]) for row in values])
        y = np.array([float(row["rate_records_per_sec"]) for row in values])
        q1 = np.array([float(row["rate_q1_records_per_sec"]) for row in values])
        q3 = np.array([float(row["rate_q3_records_per_sec"]) for row in values])
        ax.plot(x, y, marker="o", markersize=2.6, linewidth=1.05, color=color)
        ax.fill_between(x, q1, q3, color=color, alpha=0.10, linewidth=0)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_title("Operator execution rate versus legal input size")
    ax.set_xlabel("Input bytes per record")
    ax.set_ylabel("Single-operator execution rate (records/s)")
    ax.grid(True, which="both", alpha=0.22, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output_dir / "operator_rate_vs_input_bytes.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "operator_rate_vs_input_bytes.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=pathlib.Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--max-source-records", type=int, default=5000)
    parser.add_argument("--per-bucket", type=int, default=4)
    parser.add_argument("--max-points-per-operator", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--target-batch-sec", type=float, default=0.05)
    parser.add_argument("--max-calls", type=int, default=2048)
    parser.add_argument("--max-batch-mib", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260817)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for name in (
        "max_source_records",
        "per_bucket",
        "max_points_per_operator",
        "repeats",
        "max_calls",
        "max_batch_mib",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.target_batch_sec <= 0:
        raise ValueError("--target-batch-sec must be positive")
    dataset_path = args.dataset.resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    available_cpus = sorted(os.sched_getaffinity(0))
    if not available_cpus:
        raise RuntimeError("No CPU is available to the benchmark process")
    os.sched_setaffinity(0, {available_cpus[0]})

    stages = build_stages(dataset_path)
    reservoirs, counts = collect_legal_inputs(
        stages,
        dataset_path,
        args.max_source_records,
        args.per_bucket,
        args.seed,
    )
    rows: List[Dict[str, Any]] = []
    stage_summaries: Dict[str, Dict[str, Any]] = {}
    for stage in stages:
        selected = reservoirs[stage.tag].select(args.max_points_per_operator)
        if len(selected) < 3:
            raise RuntimeError(
                f"Operator {stage.tag} has only {len(selected)} distinct "
                "size strata; increase --max-source-records"
            )
        stage_summaries[stage.tag] = {
            "position": stage.position,
            "pipe_id": stage.pipe_id,
            "name": stage.name,
            "kind": stage.kind,
            "scaling": stage.scaling,
            "collection": counts[stage.tag],
            "reservoir": reservoirs[stage.tag].metadata(),
            "selected_points": len(selected),
            "min_input_bytes": min(item.input_bytes for item in selected),
            "max_input_bytes": max(item.input_bytes for item in selected),
        }
        print(
            f"[{stage.position:02d}/{len(stages)}] {stage.tag}: "
            f"benchmarking {len(selected)} legal size points",
            flush=True,
        )
        for captured in selected:
            rows.append(
                benchmark_input(
                    stage,
                    captured,
                    args.repeats,
                    args.target_batch_sec,
                    args.max_calls,
                    args.max_batch_mib * 1024 * 1024,
                )
            )

    add_elasticities(rows, stage_summaries)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "operator_input_size_execution_rate",
        "pipeline": "StackExchangeFeature",
        "pipeline_source": "evaluation/pipelines/stackexchange/cedar_dataset.py",
        "data_juicer_recipe": "refined_recipes/pretrain/redpajama-pile-stackexchange-refine.yaml",
        "dataset": os.path.relpath(dataset_path, pathlib.Path.cwd()),
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "size_definition": "cedar.pipes.get_sizeof_data(input)",
        "rate_definition": "operator invocations / measured wall-clock seconds",
        "timing_scope": "operator callable only; source, prefix, snapshot deserialization excluded",
        "input_provenance": "real records after the original preceding operators",
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "parameters": {
            key: value
            for key, value in vars(args).items()
            if key not in {"dataset", "output_dir"}
        },
        "operators": [
            {key: value for key, value in asdict(stage).items() if key != "fn"}
            for stage in stages
        ],
        "operator_summaries": stage_summaries,
    }
    with (args.output_dir / "raw_results.json").open("w", encoding="utf-8") as stream:
        json.dump(rows, stream, indent=2)
        stream.write("\n")
    with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")
    write_csv(args.output_dir / "raw_results.csv", rows)
    plot_results(rows, args.output_dir)
    (args.output_dir / "COMPLETE").touch()
    print(f"Complete: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
