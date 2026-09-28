"""Controlled fusion-rule experiment on the SimCLRv2 Jitter-Grayscale-Blur block.

Declared order (unchanged): Reader -> to_float -> Crop -> Flip -> [J -> G -> B]
-> Normalize -> Batch, with pipe ids J=4, G=3, B=2.

Organisations (all Ray, W=1, width=1, one batch in flight):
  U  three independent stages
  P  Jitter+Grayscale fused, Blur independent
  F  Jitter+Grayscale+Blur fused  (the block Cedar fuses as FusedPipe{2,3,4}? see
     the profile; this script always uses the declared order J -> G -> B)

Two cost models, both sharing the same U baseline:
  discount  rho_q * sum_{i in q} (C_i + H_i),  rho_q = I_q / sum I_i
  split     sum_{i in q} C_i + H_q            (PICO-style decomposition)

Phases (run in order; U/P/F validation never feeds the fit):
  capture    real records entering Jitter (reader->to_float->crop->flip)
  profile    member compute C_i, boundary bytes, fixed+byte boundary fit -> frozen
  predict    discount and split predictions for U/P/F
  validate   >=5 interleaved serial-service rounds (instrumentation off) plus a
             paired instrumented run and an output-equivalence check
  summarise  delivery tables
  manifest   hashes + docs snapshot
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import ray  # noqa: E402
import block_service_harness as bsh  # noqa: E402
import fusion_cost_split_harness as fcs  # noqa: E402
from block_mechanism_common import build_feature  # noqa: E402
from cedar.compose.my_optimizer import MyOptimizer  # noqa: E402
from cedar.pipes.ray_variant import (  # noqa: E402
    configure_remote_ray_experiment,
    get_ray_actor_options,
)

PIPE_JITTER, PIPE_GRAYSCALE, PIPE_BLUR = 4, 3, 2
BLOCK = (("jitter", PIPE_JITTER), ("grayscale", PIPE_GRAYSCALE), ("blur", PIPE_BLUR))
OP_SEED = {"jitter": 37, "grayscale": 61, "blur": 11}
ORGANISATIONS = {
    "U": (("jitter",), ("grayscale",), ("blur",)),
    "P": (("jitter", "grayscale"), ("blur",)),
    "F": (("jitter", "grayscale", "blur"),),
}
DATASET = ROOT / "evaluation/datasets/imagenette2/imagenette2/train"
OUT = ROOT / "outputs/pico_jgb_fusion_20260928"
DOCS = ROOT / "docs/pico_jgb_fusion_20260928"
BATCH_SIZE = 4


def _install_block() -> None:
    """Point the shared harness at the J-G-B block."""
    fcs.BLOCK = BLOCK
    fcs.ORGANISATIONS = ORGANISATIONS
    fcs.OP_SEED = OP_SEED
    fcs.DEFAULT_INPUTS = OUT


def _block_callables(batch_size: int):
    feature = build_feature(batch_size=batch_size)
    return {
        name: feature.logical_pipes[p_id].get_fused_callable()
        for name, p_id in BLOCK
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return None


def _git_commit() -> str:
    import subprocess

    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()


# ------------------------------------------------------------------ capture --


def cmd_capture(args) -> int:
    from PIL import Image
    from torchvision import transforms
    from torchvision.io import ImageReadMode, read_image

    out = Path(args.out)
    (out / "inputs").mkdir(parents=True, exist_ok=True)
    paths = sorted(DATASET.rglob("*.JPEG"))[: args.records]
    crop = transforms.RandomResizedCrop((244, 244))
    flip = transforms.RandomHorizontalFlip()
    records = []
    for index, path in enumerate(paths):
        image = read_image(str(path), mode=ImageReadMode.RGB)
        tensor = image.to(torch.float32)          # the recipe's to_float (cast)
        torch.manual_seed(20260928 + index * 1_000_003 + 53)
        tensor = crop(tensor)
        torch.manual_seed(20260928 + index * 1_000_003 + 23)
        tensor = flip(tensor)
        records.append(tensor.contiguous())
    torch.save(
        {
            "records": records,
            "source_paths": [str(p) for p in paths],
            "shape": list(records[0].shape),
            "dtype": str(records[0].dtype),
            "seed_rule": "20260928 + index*1000003 + {53 crop, 23 flip}",
        },
        out / "inputs" / "block_inputs.pt",
    )
    meta = {
        "workload": "simclrv2 J-G-B block input (records entering Jitter)",
        "records": len(records),
        "shape": list(records[0].shape),
        "dtype": str(records[0].dtype),
        "declared_order": "Reader -> to_float -> Crop -> Flip -> [J -> G -> B] -> Normalize -> Batch",
        "capture_chain": "read_image(RGB) -> to_float -> RandomResizedCrop((244,244)) -> RandomHorizontalFlip()",
        "commit": _git_commit(),
    }
    (out / "capture_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"captured {len(records)} records -> {out/'inputs/block_inputs.pt'}")
    return 0


# ------------------------------------------------------------------ profile --


def _boundary_actor_cls():
    @ray.remote(num_cpus=0)
    class PingPongActor:
        def __init__(self):
            from cedar.utils.threading import limit_native_threadpools

            self._limiter = limit_native_threadpools(1)

        def echo(self, value):
            return value

        def set_affinity(self, cpu: int):
            try:
                os.sched_setaffinity(0, {int(cpu)})
            except (AttributeError, OSError):
                return False
            return True

        def location(self):
            return {"ip": ray.util.get_node_ip_address()}

    return PingPongActor


def cmd_profile(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records = list(torch.load(out / "inputs" / "block_inputs.pt")["records"])
    callables = _block_callables(BATCH_SIZE)

    bsh._limit_threads()
    bsh._pin_cpu(args.local_cpu)
    configure_remote_ray_experiment()
    ray.init(address=args.ray_ip, ignore_reinit_error=True, logging_level="ERROR")
    actor_cls = bsh._ray_actor_classes()[0]
    options = get_ray_actor_options(0.0)

    # --- member compute on each member's real input (remote, single thread) --
    member_rows: List[Dict[str, Any]] = []
    stage_inputs: Dict[str, List[Any]] = {}
    current = list(records)
    for name, _p_id in BLOCK:
        stage_inputs[name] = list(current)
        fn = callables[name]
        actor = actor_cls.options(**options).remote(
            name, name, fn, 20260928, 2**31 - 1, OP_SEED[name]
        )
        try:
            ray.get(actor.set_affinity.remote(args.remote_cpu))
            actor_inputs = [current[i % len(current)] for i in range(len(current))]
            # warmup then measure per-record wall time through the real actor
            for item in actor_inputs[:8]:
                ray.get(actor.process.remote([item]))
            durations: List[float] = []
            bytes_in: List[int] = []
            bytes_out: List[int] = []
            outputs: List[Any] = []
            for item in actor_inputs:
                bytes_in.append(int(item.numel() * item.element_size()))
                started = time.perf_counter()
                result = ray.get(actor.process.remote([item]))
                durations.append((time.perf_counter() - started) * 1000.0)
                outputs.append(result[0])
                bytes_out.append(int(result[0].numel() * result[0].element_size()))
            events = ray.get(actor.take_events.remote())
            ray.get(actor.reset_events.remote())
            compute_ms = [wall_ns / 1e6 for _k, _op, wall_ns, _cpu, _meta in events]
            if not compute_ms:
                raise RuntimeError(f"no member-compute events for {name}")
            member_rows.append(
                {
                    "operator": name,
                    "pipe_id": dict(BLOCK)[name],
                    "records": len(compute_ms),
                    "compute_ms_per_record": statistics.fmean(compute_ms),
                    "compute_stdev_ms_per_record": statistics.pstdev(compute_ms),
                    "round_trip_ms_per_record": statistics.fmean(durations),
                    "native_bytes_in": statistics.fmean(bytes_in),
                    "native_bytes_out": statistics.fmean(bytes_out),
                }
            )
            current = outputs
        finally:
            try:
                ray.kill(actor)
            except Exception:  # noqa: BLE001
                pass

    # --- boundary ladder: serial ping-pong round trips on the real byte range
    ping = actor_cls  # keep the reference for type checkers
    probe_cls = _boundary_actor_cls()
    sizes = sorted(
        {
            int(row["native_bytes_in"] + row["native_bytes_out"])
            for row in member_rows
        }
        | {
            int(member_rows[0]["native_bytes_in"] + member_rows[-1]["native_bytes_out"]),
            int(member_rows[0]["native_bytes_in"] + member_rows[1]["native_bytes_out"]),
        }
    )
    ladder: List[Dict[str, float]] = []
    actor = probe_cls.options(**options).remote()
    try:
        ray.get(actor.set_affinity.remote(args.remote_cpu))
        location = ray.get(actor.location.remote())
        for size in sizes:
            half = max(1, size // 2)
            payload = np.zeros(half, dtype=np.uint8)
            for _ in range(5):
                ray.get(actor.echo.remote(payload))
            durations = []
            for _ in range(30):
                started = time.perf_counter()
                result = ray.get(actor.echo.remote(payload))
                durations.append((time.perf_counter() - started) * 1000.0)
                if result is None:
                    raise RuntimeError("ping-pong actor returned nothing")
            ladder.append(
                {
                    "total_bytes_per_batch": float(size),
                    "round_trip_ms_per_batch": statistics.fmean(durations),
                    "round_trip_stdev_ms": statistics.pstdev(durations),
                }
            )
    finally:
        try:
            ray.kill(actor)
        except Exception:  # noqa: BLE001
            pass
    x = np.array([row["total_bytes_per_batch"] for row in ladder])
    y = np.array([row["round_trip_ms_per_batch"] for row in ladder])
    design = np.column_stack([x, np.ones_like(x)])
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    slope, intercept = float(coefficients[0]), float(max(coefficients[1], 0.0))
    predicted = slope * x + intercept
    r2 = 1.0 - float(np.sum((y - predicted) ** 2)) / max(
        float(np.sum((y - y.mean()) ** 2)), 1e-12
    )
    boundary = {
        "fixed_ms_per_batch": intercept,
        "throughput_bytes_per_sec": (1000.0 / slope) if slope > 0 else float("inf"),
        "slope_ms_per_byte": slope,
        "r_squared": r2,
        "ladder": ladder,
        "method": "serial_ping_pong_real_byte_range",
        "actor_location": location,
        "submit_batch_size": BATCH_SIZE,
    }
    frozen = {
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "commit": _git_commit(),
        "batch_size": BATCH_SIZE,
        "records_per_batch": BATCH_SIZE,
        "backend": "RAY",
        "workers": 1,
        "stage_width": 1,
        "member_compute": member_rows,
        "boundary": boundary,
        "units": "ms per source record unless noted",
        "byte_basis": "native tensor bytes (numel * element_size); serialization not used",
        "representations": {
            "jitter_in": {"dtype": "float32", "channels": 3, "shape": [3, 244, 244]},
            "grayscale_out": {"dtype": "float32", "channels": 1, "shape": [1, 244, 244]},
        },
    }
    (out / "profile_frozen.json").write_text(json.dumps(frozen, indent=1))
    _write_csv(out / "boundary_ladder.csv", ladder)
    _write_csv(out / "operator_timing_summary.csv", member_rows)
    print(
        f"profile frozen: fixed={intercept:.3f} ms/batch, "
        f"tp={boundary['throughput_bytes_per_sec']/1e6:.1f} MB/s, r2={r2:.4f}"
    )
    return 0


def _load_frozen(out: Path) -> Dict[str, Any]:
    return json.loads((out / "profile_frozen.json").read_text())


def _bytes_per_member(frozen) -> Dict[str, float]:
    return {
        row["operator"]: float(row["native_bytes_in"] + row["native_bytes_out"])
        for row in frozen["member_compute"]
    }


def _h_ms(frozen, in_bytes: float, out_bytes: float) -> float:
    boundary = frozen["boundary"]
    per_batch = boundary["fixed_ms_per_batch"] + (
        (in_bytes + out_bytes)
        * frozen["records_per_batch"]
        / boundary["throughput_bytes_per_sec"]
        * 1000.0
    )
    return per_batch / frozen["records_per_batch"]


def cmd_predict(args) -> int:
    out = Path(args.out)
    frozen = _load_frozen(out)
    compute = {row["operator"]: row["compute_ms_per_record"] for row in frozen["member_compute"]}
    member_bytes = _bytes_per_member(frozen)
    order = [name for name, _ in BLOCK]
    # external in/out bytes of a group follow the declared chain
    in_of = {}
    out_of = {}
    for row in frozen["member_compute"]:
        in_of[row["operator"]] = float(row["native_bytes_in"])
        out_of[row["operator"]] = float(row["native_bytes_out"])
    rows = []
    rho = {}
    for organisation, groups in ORGANISATIONS.items():
        total_compute = sum(compute[name] for name in order)
        if organisation == "U":
            handoff = sum(
                _h_ms(frozen, in_of[name], out_of[name]) for name in order
            )
            rows.append(
                {
                    "organisation": "U",
                    "method": "both",
                    "rho": 1.0,
                    "compute_ms_per_record": total_compute,
                    "handoff_ms_per_record": handoff,
                    "total_ms_per_record": total_compute + handoff,
                    "terms": json.dumps({"groups": [[n] for n in order]}),
                }
            )
            continue
        split_handoff = 0.0
        discount_total = 0.0
        breakdown = []
        for group in groups:
            first, last = group[0], group[-1]
            fused_bytes = in_of[first] + out_of[last]
            base_bytes = sum(member_bytes[name] for name in group)
            rho_q = fused_bytes / base_bytes
            h_q = _h_ms(frozen, in_of[first], out_of[last])
            split_handoff += h_q
            group_compute = sum(compute[name] for name in group)
            group_handoff = sum(
                _h_ms(frozen, in_of[name], out_of[name]) for name in group
            )
            discount_total += rho_q * (group_compute + group_handoff)
            breakdown.append(
                {
                    "group": list(group),
                    "fused_bytes": fused_bytes,
                    "member_bytes": base_bytes,
                    "rho": rho_q,
                    "member_compute": group_compute,
                    "group_handoff": group_handoff,
                    "fused_handoff": h_q,
                }
            )
            rho["+".join(group)] = rho_q
        scaled_compute = 0.0
        scaled_handoff = 0.0
        for item in breakdown:
            scaled_compute += item["rho"] * item["member_compute"]
            scaled_handoff += item["rho"] * item["group_handoff"]
        rows.append(
            {
                "organisation": organisation,
                "method": "discount",
                "rho": json.dumps(
                    {item["group"][0] + "+" + item["group"][-1]: item["rho"] for item in breakdown}
                ),
                "compute_ms_per_record": scaled_compute,
                "handoff_ms_per_record": scaled_handoff,
                "total_ms_per_record": discount_total,
                "terms": json.dumps(breakdown),
            }
        )
        rows.append(
            {
                "organisation": organisation,
                "method": "split",
                "rho": "",
                "compute_ms_per_record": total_compute,
                "handoff_ms_per_record": split_handoff,
                "total_ms_per_record": total_compute + split_handoff,
                "terms": json.dumps(breakdown),
            }
        )
    fields = [
        "organisation", "method", "rho", "compute_ms_per_record",
        "handoff_ms_per_record", "total_ms_per_record", "terms",
    ]
    with (out / "predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    (out / "rho_breakdown.json").write_text(
        json.dumps(
            {
                "member_bytes": member_bytes,
                "member_in_bytes": in_of,
                "member_out_bytes": out_of,
                "rho": rho,
                "note": "I_i = native in+out of member i; I_q = fused group external in+out",
            },
            indent=1,
        )
    )
    print(f"predictions rows={len(rows)}")
    return 0


# ----------------------------------------------------------------- validate --


def _submit_batches(frozen, organisation: str) -> List[int]:
    in_of = {r["operator"]: float(r["native_bytes_in"]) for r in frozen["member_compute"]}
    out_of = {r["operator"]: float(r["native_bytes_out"]) for r in frozen["member_compute"]}
    sizes = []
    for group in ORGANISATIONS[organisation]:
        sizes.append(
            int(
                MyOptimizer._dp_ray_submit_batch_size(
                    in_of[group[0]], out_of[group[-1]]
                )
            )
        )
    return sizes


def cmd_validate(args) -> int:
    _install_block()
    out = Path(args.out)
    frozen = _load_frozen(out)
    ray, actor_classes, options = fcs._setup(args)
    pool = list(torch.load(out / "inputs" / "block_inputs.pt")["records"])
    callables = _block_callables(BATCH_SIZE)
    runners: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []
    correctness: Dict[str, Any] = {}
    try:
        for organisation in ORGANISATIONS:
            runners[organisation] = fcs.StageRunner(
                organisation, callables, BATCH_SIZE, options, actor_classes,
                args.remote_cpu, True,
                submit_batch_by_stage=_submit_batches(frozen, organisation),
            )
        for instrument in ("off", "on"):
            for runner in runners.values():
                runner.set_instrument(instrument == "on")
            for runner in runners.values():
                for index in range(args.warmup):
                    runner.run_batch(fcs._batch_records(pool, index, BATCH_SIZE))
            rounds = args.rounds if instrument == "off" else args.instrument_rounds
            for repeat in range(rounds):
                order = ["U", "P", "F"]
                if repeat % 2 == 1:
                    order = list(reversed(order))
                shift = repeat % 3
                order = order[shift:] + order[:shift]
                for organisation in order:
                    runner = runners[organisation]
                    for index in range(args.batches):
                        records = fcs._batch_records(pool, index, BATCH_SIZE)
                        runner.reset_events()
                        result = runner.run_batch(records)
                        stage_compute: List[float] = []
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
                                "round": repeat,
                                "order_index": order.index(organisation),
                                "organisation": organisation,
                                "batch_id": index,
                                "source_records": BATCH_SIZE,
                                "service_ms": result["whole_time"] * 1000.0,
                                "service_ms_per_record": (
                                    result["whole_time"] * 1000.0 / BATCH_SIZE
                                ),
                                "stage_T_ms_per_record": json.dumps(
                                    [
                                        value * 1000.0 / BATCH_SIZE
                                        for value in result["stage_times"]
                                    ]
                                ),
                                "stage_C_ms_per_record": json.dumps(
                                    [value / BATCH_SIZE for value in stage_compute]
                                ),
                                "stages": json.dumps([list(g) for g in runner.stages]),
                            }
                        )
                print(f"instrument={instrument} round={repeat} rows={len(rows)}", flush=True)
        # output equivalence on a fixed subset (>=32 records)
        subset = min(args.correctness_records, len(pool))
        outputs = {}
        for organisation, runner in runners.items():
            outs = []
            for index in range(0, subset, BATCH_SIZE):
                records = [
                    pool[(index + offset) % len(pool)]
                    for offset in range(min(BATCH_SIZE, subset - index))
                ]
                result = runner.run_batch(records)
                outs.extend(result["outputs"])
            outputs[organisation] = outs
        reference = outputs["U"]
        for organisation in ("P", "F"):
            diffs = [
                float((a - b).abs().max())
                for a, b in zip(reference, outputs[organisation])
            ]
            correctness[organisation] = {
                "records": len(diffs),
                "max_abs_diff": max(diffs) if diffs else None,
                "allclose_1e-6": all(d <= 1e-6 for d in diffs),
            }
        correctness["records_checked"] = subset
        correctness["comparison"] = "U as reference; elementwise max abs difference"
        correctness["random_state"] = (
            "per (record key, operator) seeds, identical across U/P/F; "
            "bitwise equality expected for deterministic members"
        )
    finally:
        for runner in runners.values():
            runner.shutdown()
    _write_csv(out / "measurements.csv", rows)
    (out / "correctness.json").write_text(json.dumps(correctness, indent=1))
    print(f"validation rows={len(rows)} correctness={correctness.get('P')}")
    return 0


def cmd_summarise(args) -> int:
    out = Path(args.out)
    rows = list(csv.DictReader((out / "measurements.csv").open()))
    predictions = list(csv.DictReader((out / "predictions.csv").open()))
    summary: List[Dict[str, Any]] = []
    for instrument in ("off", "on"):
        for organisation in ("U", "P", "F"):
            subset = [
                float(row["service_ms_per_record"])
                for row in rows
                if row["instrument"] == instrument
                and row["organisation"] == organisation
            ]
            if not subset:
                continue
            per_round: Dict[str, List[float]] = {}
            for row in rows:
                if (
                    row["instrument"] == instrument
                    and row["organisation"] == organisation
                ):
                    per_round.setdefault(row["round"], []).append(
                        float(row["service_ms_per_record"])
                    )
            round_means = [statistics.fmean(v) for v in per_round.values()]
            summary.append(
                {
                    "instrument": instrument,
                    "organisation": organisation,
                    "batches": len(subset),
                    "rounds": len(round_means),
                    "mean_ms_per_record": statistics.fmean(subset),
                    "median_ms_per_record": statistics.median(subset),
                    "p10_ms_per_record": float(np.quantile(subset, 0.10)),
                    "p90_ms_per_record": float(np.quantile(subset, 0.90)),
                    "round_means": json.dumps(round_means),
                    "round_stdev_ms": (
                        statistics.pstdev(round_means) if len(round_means) > 1 else 0.0
                    ),
                }
            )
    _write_csv(out / "summary.csv", summary)
    actual = {
        row["organisation"]: float(row["mean_ms_per_record"])
        for row in summary
        if row["instrument"] == "off"
    }
    stdev = {
        row["organisation"]: float(row["round_stdev_ms"])
        for row in summary
        if row["instrument"] == "off"
    }
    comparison = []
    for row in predictions:
        organisation = row["organisation"]
        predicted = float(row["total_ms_per_record"])
        measured = actual[organisation]
        comparison.append(
            {
                "organisation": organisation,
                "method": row["method"],
                "rho": row["rho"],
                "predicted_compute_ms": row["compute_ms_per_record"],
                "predicted_handoff_ms": row["handoff_ms_per_record"],
                "predicted_total_ms": predicted,
                "measured_ms": measured,
                "abs_error_ms": predicted - measured,
                "rel_error": (predicted - measured) / measured,
                "measured_round_stdev_ms": stdev[organisation],
                "error_over_stdev": (
                    (predicted - measured) / stdev[organisation]
                    if stdev[organisation]
                    else float("nan")
                ),
            }
        )
    _write_csv(out / "comparison.csv", comparison)
    instrument = []
    for organisation in ("U", "P", "F"):
        off = next(
            r for r in summary
            if r["instrument"] == "off" and r["organisation"] == organisation
        )
        on = next(
            (r for r in summary
             if r["instrument"] == "on" and r["organisation"] == organisation),
            None,
        )
        if on is None:
            continue
        instrument.append(
            {
                "organisation": organisation,
                "instrument_off_ms": off["mean_ms_per_record"],
                "instrument_on_ms": on["mean_ms_per_record"],
                "instrument_effect_pct": (
                    100.0
                    * (float(on["mean_ms_per_record"]) - float(off["mean_ms_per_record"]))
                    / float(off["mean_ms_per_record"])
                ),
            }
        )
    _write_csv(out / "instrumentation_comparison.csv", instrument)
    figure = {
        "measured_ms_per_record": actual,
        "measured_round_stdev_ms": stdev,
        "predictions": comparison,
        "summary": summary,
        "window_note": (
            "measured bars = uninstrumented serial service (whole block, one batch "
            "in flight); instrumented run only used for the member-compute split"
        ),
    }
    (out / "figure_data.json").write_text(json.dumps(figure, indent=1, default=float))

    def lookup(org: str, method: str):
        return next(
            (row for row in comparison if row["organisation"] == org and row["method"] == method),
            None,
        )

    lines = [
        "# J–G–B 融合规则受控实验（SimCLRv2，2026-09-28）",
        "",
        "声明顺序不变：Reader → to_float → Crop → Flip → **[Jitter → Grayscale → Blur]** → Normalize → Batch。",
        "全部阶段使用远端 Ray、W=1、每阶段 width=1、批大小 4、单批在飞。",
        "",
        "## 独立剖析（冻结于 `profile_frozen.json`）",
        "",
        "| 成员 | C_i (ms/记录) | 往返 (ms/记录) | in→out 原生字节 |",
        "| --- | ---: | ---: | --- |",
    ]
    frozen = _load_frozen(out)
    for row in frozen["member_compute"]:
        lines.append(
            f"| {row['operator']} | {row['compute_ms_per_record']:.3f} | "
            f"{row.get('round_trip_ms_per_record', float('nan')):.3f} | "
            f"{int(row['native_bytes_in'])} → {int(row['native_bytes_out'])} |"
        )
    boundary = frozen["boundary"]
    lines += [
        "",
        f"边界模型（序列 ping-pong，覆盖实际 in+out 范围）：固定 {boundary['fixed_ms_per_batch']:.3f} ms/批、"
        f"{boundary['throughput_bytes_per_sec']/1e6:.1f} MB/s、R²={boundary['r_squared']:.4f}；"
        f"字节口径 = **原生张量字节**（非序列化、非线上）。",
        "",
        "## 预测（冻结后）vs 5 轮无插桩实测",
        "",
        "| 组织 | 实测 (ms/记录) | 轮间 stdev | 整体折扣 | 误差 | PICO 分项 | 误差 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for organisation in ("U", "P", "F"):
        method = "both" if organisation == "U" else "discount"
        discount = lookup(organisation, method)
        split = lookup(organisation, "split")
        if organisation == "U":
            lines.append(
                f"| U | {actual['U']:.2f} | {stdev['U']:.2f} | {discount['predicted_total_ms']:.2f} "
                f"| {100*discount['rel_error']:+.1f}% | 同左 | {100*discount['rel_error']:+.1f}% |"
            )
            continue
        lines.append(
            f"| {organisation} | {actual[organisation]:.2f} | {stdev[organisation]:.2f} | "
            f"{discount['predicted_total_ms']:.2f} | {100*discount['rel_error']:+.1f}% | "
            f"{split['predicted_total_ms']:.2f} | {100*split['rel_error']:+.1f}% |"
        )
    lines += [
        "",
        "## 结论边界",
        "",
        "- 实测 = 无成员插桩的完整串行服务时间（块输入提交 → 最终输出可用），U/P/F 同一计时边界。",
        "- 插桩对照见 `instrumentation_comparison.csv`；两套口径不混用。",
        "- 输出一致性见 `correctness.json`（U 为参考，逐元素最大绝对差）。",
        "- 组合计价的部署差异：本实验用 `profile_frozen.json` 的独立参数；部署版 PICO 读的是 profile 的",
        "  `object_boundary`/`staged_handoff` 字段（见 `docs/experiments.md` §4.18），二者不宣称等价。",
    ]
    (out / "README.md").write_text("\n".join(lines))
    print(f"comparison rows={len(comparison)}")
    return 0


SMALL = (
    "README.md", "protocol.json", "profile_frozen.json", "rho_breakdown.json",
    "predictions.csv", "measurements.csv", "instrumentation_comparison.csv",
    "operator_timing_summary.csv", "correctness.json", "figure_data.json",
    "comparison.csv", "summary.csv",
)


def cmd_manifest(args) -> int:
    import shutil

    out = Path(args.out)
    docs = DOCS
    docs.mkdir(parents=True, exist_ok=True)
    protocol = {
        "commit": _git_commit(),
        "workload": "SimCLRv2 J-G-B block (jitter -> grayscale -> blur)",
        "declared_order": "Reader -> to_float -> Crop -> Flip -> J -> G -> B -> Normalize -> Batch",
        "backend": "RAY (remote host pinned)", "workers": 1, "stage_width": 1,
        "batch_size": BATCH_SIZE, "records_per_batch": BATCH_SIZE,
        "host": "172.23.166.105", "ray_ip": "172.23.166.105:6379",
        "threads": 1, "timing_boundary": "client submit of block input -> last output available",
        "byte_basis": "native tensor bytes (numel * element_size)",
        "units": "ms per source record",
        "input": str((out / "inputs" / "block_inputs.pt").relative_to(ROOT)),
        "input_meta": json.loads((out / "capture_meta.json").read_text()),
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=1))
    manifest = {"result_dir": str(out.relative_to(ROOT)), "small_files": {}, "provenance": {}}
    for name in SMALL:
        path = out / name
        if not path.exists():
            continue
        data = path.read_bytes()
        manifest["small_files"][name] = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
            "rows": len(data.decode(errors="replace").splitlines()) - 1,
        }
        shutil.copy2(path, docs / name)
    payload = out / "inputs" / "block_inputs.pt"
    if payload.exists():
        data = payload.read_bytes()
        manifest["large_files"] = {
            "inputs/block_inputs.pt": {
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
            }
        }
    for name in ("fusion_jgb_experiment.py",):
        path = ROOT / "scripts" / name
        manifest["provenance"][name] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    shutil.copy2(out / "MANIFEST.json", docs / "MANIFEST.json")
    print(f"manifest: {len(manifest['small_files'])} small files -> {docs}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    capture = sub.add_parser("capture")
    capture.add_argument("--out", default=str(OUT))
    capture.add_argument("--records", type=int, default=200)
    capture.set_defaults(func=cmd_capture)
    profile = sub.add_parser("profile")
    profile.add_argument("--out", default=str(OUT))
    profile.add_argument("--local-cpu", type=int, default=12)
    profile.add_argument("--remote-cpu", type=int, default=8)
    profile.add_argument("--ray-ip", default="172.23.166.105:6379")
    profile.set_defaults(func=cmd_profile)
    predict = sub.add_parser("predict")
    predict.add_argument("--out", default=str(OUT))
    predict.set_defaults(func=cmd_predict)
    validate = sub.add_parser("validate")
    validate.add_argument("--out", default=str(OUT))
    validate.add_argument("--rounds", type=int, default=5)
    validate.add_argument("--batches", type=int, default=100)
    validate.add_argument("--warmup", type=int, default=10)
    validate.add_argument("--instrument-rounds", type=int, default=1)
    validate.add_argument("--correctness-records", type=int, default=32)
    validate.add_argument("--local-cpu", type=int, default=12)
    validate.add_argument("--remote-cpu", type=int, default=8)
    validate.add_argument("--same-remote-cpu", action="store_true", default=True)
    validate.add_argument("--ray-ip", default="172.23.166.105:6379")
    validate.set_defaults(func=cmd_validate)
    summarise = sub.add_parser("summarise")
    summarise.add_argument("--out", default=str(OUT))
    summarise.set_defaults(func=cmd_summarise)
    manifest = sub.add_parser("manifest")
    manifest.add_argument("--out", default=str(OUT))
    manifest.set_defaults(func=cmd_manifest)
    args = parser.parse_args()
    raise SystemExit(args.func(args))
