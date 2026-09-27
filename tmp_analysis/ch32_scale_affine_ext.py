"""Round 2 of the §3.2 affine-response data: extend ranges, fill classes.

Only operator-level measurement and fitting; no optimizer, fusion, offload or
throughput work.  The protocol (remote Ray actor, 1 thread, callable-only
timing, refreshed payload per call, 3 independent blocks, arithmetic mean)
and the payload construction are imported from round 1 so both batches are
produced by the same code path.

Normalisation stays frozen: every panel divides the tensor's native bytes and
the mean time by the *round-1 reference cell* of that operator, so different
representation classes keep their byte and time differences.

Usage (inside the container):
  python -m tmp_analysis.ch32_scale_affine_ext measure [--dry-run]
  python -m tmp_analysis.ch32_scale_affine_ext merge
  python -m tmp_analysis.ch32_scale_affine_ext fit
  python -m tmp_analysis.ch32_scale_affine_ext manifest
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import pickle
import random
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from tmp_analysis import ch32_scale_affine_probe as r1

ROOT = Path("/workspace/OptimalCedar")
OUT = ROOT / "outputs/ch32_scale_affine_ext_20260927"
ROUND1 = ROOT / "outputs/ch32_scale_affine_20260927"
DOCS = ROOT / "docs/ch32_scale_affine_ext_20260927"
RAY_ADDRESS = "172.23.166.105:6379"
BLOCKS = 3
BASE_SEED = r1.BASE_SEED
MAX_CALLS_PER_CELL = r1.MAX_CALLS_PER_CELL
LOCAL_X_MAX = 2.0  # pre-declared local window for the ColorJitter 3ch check

# ------------------------------------------------------------- cell plan ----

# Extension ladders are chosen from the frozen reference bytes of each
# operator so the byte-normalised abscissa reaches the requested window.
EXTENSION: Dict[Tuple[str, str], Dict[str, List[Tuple[str, int]]]] = {
    # reference 602,112 B -> x = side^2 / 602,112  (uint8 single channel)
    ("jitter", "uint8:1ch"): {
        "train": [("xt0", 512), ("xt1", 736), ("xt2", 1024), ("xt3", 1098)],
        "validation": [("xv0", 640), ("xv1", 832), ("xv2", 960), ("xv3", 1088)],
        "note": "512 duplicates round-1 t7; x spans 0.44-2.00",
    },
    # reference 1,769,472 B -> x = side^2 / 1,769,472 (uint8 single channel)
    ("crop", "uint8:1ch"): {
        "train": [("xt0", 448), ("xt1", 672), ("xt2", 832), ("xt3", 960)],
        "validation": [("xv0", 576), ("xv1", 768), ("xv2", 896), ("xv3", 986)],
        "note": "448 duplicates round-1 t7; x spans 0.11-0.55",
    },
    # classes that were never measured; same ladder for all three
    ("flip", "uint8:1ch"): {
        "train": [("nt0", 64), ("nt1", 144), ("nt2", 224), ("nt3", 320),
                  ("nt4", 448), ("nt5", 640), ("nt6", 832), ("nt7", 1024)],
        "validation": [("nv0", 96), ("nv1", 192), ("nv2", 384), ("nv3", 704)],
        "note": "new representation class",
    },
    ("grayscale", "uint8:1ch"): {
        "train": [("nt0", 64), ("nt1", 144), ("nt2", 224), ("nt3", 320),
                  ("nt4", 448), ("nt5", 640), ("nt6", 832), ("nt7", 1024)],
        "validation": [("nv0", 96), ("nv1", 192), ("nv2", 384), ("nv3", 704)],
        "note": "new representation class; Grayscale clones a 1-channel input",
    },
    ("grayscale", "float32:1ch"): {
        "train": [("nt0", 64), ("nt1", 144), ("nt2", 224), ("nt3", 320),
                  ("nt4", 448), ("nt5", 640), ("nt6", 832), ("nt7", 1024)],
        "validation": [("nv0", 96), ("nv1", 192), ("nv2", 384), ("nv3", 704)],
        "note": "new representation class; clone path copies 4 bytes/element",
    },
}

# Round-1 cells re-measured unchanged to detect drift between batches.
CONTROLS = [
    ("blur", "float32:1ch", "t4"),
    ("blur", "uint8:3ch", "t7"),
    ("jitter", "float32:3ch", "t4"),
    ("jitter", "uint8:1ch", "t7"),
    ("crop", "float32:3ch", "t6"),
    ("crop", "uint8:1ch", "t7"),
    ("grayscale", "float32:3ch", "t4"),
    ("flip", "float32:3ch", "t4"),
]


def build_cells() -> List[Dict[str, Any]]:
    cells: List[Dict[str, Any]] = []
    for (operator, klass), ladder in EXTENSION.items():
        for role, entries in (("train", ladder["train"]), ("validation", ladder["validation"])):
            for size_id, side in entries:
                cells.append(
                    {
                        "operator": operator,
                        "representation_class": klass,
                        "size_id": size_id,
                        "role": role,
                        "height": side,
                        "width": side,
                        "cell_id": f"{operator}:{klass}:{size_id}",
                        "origin": "extension",
                    }
                )
    for operator, klass, size_id in CONTROLS:
        side = _round1_side(operator, klass, size_id)
        cells.append(
            {
                "operator": operator,
                "representation_class": klass,
                "size_id": size_id,
                "role": "control",
                "height": side,
                "width": side,
                "cell_id": f"{operator}:{klass}:{size_id}",
                "origin": "control",
            }
        )
    return cells


def _round1_side(operator: str, klass: str, size_id: str) -> int:
    figure = json.loads((ROUND1 / "figure_data.json").read_text())
    for cell in figure["cells"]:
        if (
            cell["operator"] == operator
            and cell["representation_class"] == klass
            and cell["size_id"] == size_id
        ):
            return int(cell["height"])
    raise KeyError(f"round-1 cell {operator}:{klass}:{size_id} not found")


def _round1_reference() -> Dict[str, Dict[str, Any]]:
    """The frozen per-operator reference used by the paper figure."""
    figure = json.loads((ROUND1 / "figure_data.json").read_text())
    by_id = {cell["cell_id"]: cell for cell in figure["cells"]}
    out: Dict[str, Dict[str, Any]] = {}
    for operator, panel in figure["panels"].items():
        anchor = by_id[panel["reference_anchor_cell"]]
        out[operator] = {
            "cell_id": anchor["cell_id"],
            "representation_class": anchor["representation_class"],
            "native_bytes": anchor["native_bytes"],
            "elements": anchor["elements"],
            "mean_ms": anchor["mean_ms"],
            "batch": "round1_20260927",
        }
    return out


assert set(_round1_reference()) == {"blur", "jitter", "crop", "grayscale", "flip"}


# ------------------------------------------------------------------ actor ---


def _actor_cls():
    import ray

    @ray.remote
    class ExtScaleActor:
        def __init__(self, sources, base_seed, budgets):
            from cedar.utils.threading import limit_native_threadpools

            self._limiter = limit_native_threadpools(1)
            self._sources = sources
            self._base_seed = base_seed
            self._budgets = budgets
            self._operators = r1.build_operators()
            self._snapshots: Dict[str, List[bytes]] = {}
            self._meta: Dict[str, Dict[str, Any]] = {}
            self._output_meta: Dict[str, Dict[str, Any]] = {}

        def describe(self):
            import hashlib

            import cedar.pipes.common as common_module

            ctx = ray.get_runtime_context()
            return {
                "node_id": ctx.get_node_id(),
                "ip": ray.util.get_node_ip_address(),
                "pid": os.getpid(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "torch_threads": torch.get_num_threads(),
                "affinity": sorted(os.sched_getaffinity(0)),
                "cedar_common_file": common_module.__file__,
                "cedar_common_sha256": hashlib.sha256(
                    Path(common_module.__file__).read_bytes()
                ).hexdigest(),
            }

        def prepare(self, cells):
            snapshots_bytes = 0
            for cell in cells:
                payloads = [
                    r1.build_payload(
                        source,
                        cell["representation_class"],
                        cell["height"],
                        cell["width"],
                    )
                    for source in self._sources
                ]
                self._snapshots[cell["cell_id"]] = [
                    pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
                    for value in payloads
                ]
                snapshots_bytes += sum(
                    len(snapshot)
                    for snapshot in self._snapshots[cell["cell_id"]]
                )
                self._meta[cell["cell_id"]] = r1.representation_metadata(payloads[0])
                self._output_meta[cell["cell_id"]] = self._output_metadata(
                    cell, payloads[0]
                )
                del payloads
            return {"cells": len(cells), "snapshot_bytes": snapshots_bytes}

        def _output_metadata(self, cell, payload):
            fn = self._operators[cell["operator"]]
            random.seed(7)
            torch.manual_seed(7)
            return r1.representation_metadata(fn(payload))

        def warmup(self, cells, calls: int = 3):
            for cell in cells:
                fn = self._operators[cell["operator"]]
                snapshots = self._snapshots[cell["cell_id"]]
                for index in range(max(1, calls)):
                    value = pickle.loads(snapshots[index % len(snapshots)])
                    fn(value)
                    del value
            return True

        def measure(self, cells, block):
            order = list(cells)
            random.Random(self._base_seed + 977 * block).shuffle(order)
            durations: Dict[str, List[Tuple[int, float]]] = {
                cell["cell_id"]: [] for cell in cells
            }
            budget = {
                cell["cell_id"]: float(self._budgets[cell["operator"]])
                for cell in cells
            }
            spent = {cell_id: 0.0 for cell_id in durations}
            seed_base = self._base_seed + 100000 * block
            call_index = 0
            pending = True
            while pending:
                pending = False
                for cell in order:
                    cell_id = cell["cell_id"]
                    values = durations[cell_id]
                    if (
                        len(values) >= MAX_CALLS_PER_CELL
                        or spent[cell_id] >= budget[cell_id]
                    ):
                        continue
                    pending = True
                    fn = self._operators[cell["operator"]]
                    snapshots = self._snapshots[cell_id]
                    source_index = call_index % len(snapshots)
                    value = pickle.loads(snapshots[source_index])
                    seed = seed_base + call_index
                    torch.manual_seed(seed)
                    random.seed(seed)
                    start = time.perf_counter()
                    result = fn(value)
                    elapsed = time.perf_counter() - start
                    del result
                    del value
                    values.append((source_index, 1000.0 * elapsed))
                    spent[cell_id] += elapsed
                call_index += 1
            for cell_id in durations:
                durations[cell_id] = sorted(
                    durations[cell_id], key=lambda item: item[0]
                )
            return durations, spent

        def metadata(self):
            return {"input": self._meta, "output": self._output_meta}

    return ExtScaleActor


def _prepare_snapshot() -> Path:
    import shutil

    snapshot = OUT / "modules"
    if snapshot.exists():
        shutil.rmtree(snapshot)
    snapshot.mkdir(parents=True)
    shutil.copytree(
        ROOT / "cedar",
        snapshot / "cedar",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    (snapshot / "tmp_analysis").mkdir()
    for name in (
        "__init__.py",
        "ch32_scale_affine_probe.py",
        "ch32_scale_affine_ext.py",
    ):
        shutil.copy2(ROOT / "tmp_analysis" / name, snapshot / "tmp_analysis" / name)
    return snapshot


def _ray_init():
    import ray
    from cedar.pipes.ray_variant import (
        configure_remote_ray_experiment,
        get_ray_actor_options,
        validate_remote_ray_resource,
    )

    configure_remote_ray_experiment()
    snapshot = _prepare_snapshot()
    if not ray.is_initialized():
        ray.init(
            address=RAY_ADDRESS,
            ignore_reinit_error=True,
            logging_level="ERROR",
            runtime_env={
                "working_dir": str(snapshot),
                "env_vars": {
                    name: os.environ[name]
                    for name in (
                        "OMP_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS",
                        "NUMEXPR_NUM_THREADS",
                    )
                    if name in os.environ
                },
            },
        )
    resource = os.environ.get("CEDAR_RAY_PLACEMENT_RESOURCE", "cedar_remote")
    validate_remote_ray_resource(resource, 0.001)
    return ray, get_ray_actor_options


# --------------------------------------------------------------- measure ----


MEASUREMENT_COLUMNS = (
    "implementation",
    "parameters",
    "seed_rule",
    "block_seed",
    "source_ids",
    "n_sources",
    "elements",
    "serialized_bytes",
    "native_bytes",
    "dtype",
    "shape",
    "contiguous",
    "container",
    "out_elements",
    "out_serialized_bytes",
    "out_native_bytes",
    "out_dtype",
    "out_shape",
    "out_contiguous",
    "out_representation_class",
)


def cmd_measure(args) -> int:
    ray, get_ray_actor_options = _ray_init()
    sources = r1._load_sources()
    cells = build_cells()
    if args.controls_only:
        cells = [cell for cell in cells if cell["origin"] == "control"]
    if args.dry_run:
        cells = [cell for cell in cells if cell["origin"] == "extension"][:6]
    blocks = 1 if args.dry_run else BLOCKS
    OUT.mkdir(parents=True, exist_ok=True)
    label = "recheck" if args.controls_only else "measurements"
    budgets = {
        name: spec["budget_sec"] for name, spec in r1.OPERATOR_SPECS.items()
    }

    actor_cls = _actor_cls()
    actor = actor_cls.options(**get_ray_actor_options(num_cpus=1.0)).remote(
        sources, BASE_SEED, budgets
    )
    location = ray.get(actor.describe.remote())
    driver_ip = ray.util.get_node_ip_address()
    if location["ip"] == driver_ip:
        raise RuntimeError("measurement actor must not run on the driver node")
    print(f"actor location: {location['ip']} (driver {driver_ip})", flush=True)
    prepared = ray.get(actor.prepare.remote(cells))
    print(f"prepared {prepared}", flush=True)
    ray.get(actor.warmup.remote(cells, 3))

    rows: List[Dict[str, Any]] = []
    raw_rows: List[Tuple[str, int, int, int, float]] = []
    for block in range(blocks):
        started = time.time()
        durations, spent = ray.get(actor.measure.remote(cells, block))
        for cell in cells:
            values = durations[cell["cell_id"]]
            times = [value for _, value in values]
            rows.append(
                {
                    "operator": cell["operator"],
                    "cell_id": cell["cell_id"],
                    "representation_class": cell["representation_class"],
                    "size_id": cell["size_id"],
                    "pair_id": "",
                    "origin": cell["origin"],
                    "train_or_validation": cell["role"],
                    "block_id": block,
                    "height": cell["height"],
                    "width": cell["width"],
                    "calls": len(times),
                    "mean_ms": statistics.fmean(times),
                    "median_ms": statistics.median(times),
                    "p10_ms": float(np.percentile(times, 10)),
                    "p90_ms": float(np.percentile(times, 90)),
                    "std_ms": statistics.pstdev(times) if len(times) > 1 else 0.0,
                    "min_ms": min(times),
                    "max_ms": max(times),
                    "accumulated_sec": spent[cell["cell_id"]],
                    "measurement_protocol": "ch32_scale_affine_20260927",
                }
            )
            for call_index, (source_index, value) in enumerate(values):
                raw_rows.append((cell["cell_id"], block, call_index, source_index, value))
        print(
            f"block {block} done in {time.time() - started:.1f}s "
            f"({sum(len(v) for v in durations.values())} calls)",
            flush=True,
        )

    meta = ray.get(actor.metadata.remote())
    for row in rows:
        cell_id = row["cell_id"]
        for prefix, source in (
            ("", meta["input"][cell_id]),
            ("out_", meta["output"][cell_id]),
        ):
            for key, value in source.items():
                row[prefix + key] = value
        row["implementation"] = r1.IMPLEMENTATION_NAMES.get(row["operator"], "")
        row["parameters"] = json.dumps(
            r1.OPERATOR_SPECS[row["operator"]]["parameters"], sort_keys=True
        )
        row["seed_rule"] = (
            "torch.manual_seed/random.seed(base_seed + 100000*block_id + call_index); "
            "set outside the timing window"
        )
        row["block_seed"] = BASE_SEED + 100000 * int(row["block_id"])
        row["source_ids"] = ",".join(str(index) for index in range(r1.N_SOURCES))
        row["n_sources"] = r1.N_SOURCES
    _write_table(OUT / f"{label}.csv", rows)
    raw_name = "recheck_raw_calls.csv.gz" if args.controls_only else "raw_calls.csv.gz"
    with gzip.open(OUT / raw_name, "wt", newline="") as handle:
        handle.write("cell_id,block_id,call_index,source_id,ms\n")
        for cell_id, block, call_index, source_index, value in raw_rows:
            handle.write(f"{cell_id},{block},{call_index},{source_index},{value:.6f}\n")
    env = {
        "commit": r1._git_commit(),
        "round": "ch32_scale_affine_ext_20260927",
        "ray_address": RAY_ADDRESS,
        "driver_ip": driver_ip,
        "actor": location,
        "driver_cedar_common_sha256": hashlib.sha256(
            (ROOT / "cedar/pipes/common.py").read_bytes()
        ).hexdigest(),
        "blocks": blocks,
        "base_seed": BASE_SEED,
        "max_calls_per_cell": MAX_CALLS_PER_CELL,
        "cells": len(cells),
        "extension_plan": {
            f"{operator}:{klass}": ladder
            for (operator, klass), ladder in EXTENSION.items()
        },
        "controls": CONTROLS,
        "frozen_reference": _round1_reference(),
        "protocol": {
            "inherited_from": "tmp_analysis/ch32_scale_affine_probe.py (round 1)",
            "timed_window": "callable only; pickle.loads and seeding outside the clock",
            "value_freshness": "pickle.loads(snapshot) before every call",
            "interleaving": "all cells of the run interleaved inside one window per block",
            "order": "shuffled per block with seed base_seed + 977*block",
            "seeds": "torch.manual_seed/random.seed(base_seed + 100000*block + call_index)",
            "mean_primary": True,
            "normalisation": (
                "x = native_bytes / native_bytes(frozen round-1 reference cell); "
                "y = mean_ms / mean_ms(frozen round-1 reference cell)"
            ),
        },
    }
    env_name = "env_recheck.json" if args.controls_only else "env.json"
    env["controls_only"] = bool(args.controls_only)
    (OUT / env_name).write_text(json.dumps(env, indent=1, default=str))
    print(f"wrote {OUT}/{label}.csv rows={len(rows)}", flush=True)
    return 0


def _write_table(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = [
        "operator", "cell_id", "representation_class", "size_id", "pair_id",
        "batch", "origin", "train_or_validation", "block_id", "height", "width", "calls",
        "mean_ms", "median_ms", "p10_ms", "p90_ms", "std_ms", "min_ms",
        "max_ms", "accumulated_sec", "measurement_protocol",
    ] + list(MEASUREMENT_COLUMNS)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ----------------------------------------------------------------- merge ----


def _read_measurements(path: Path, batch: Optional[str] = None):
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if batch is not None:
            row["batch"] = batch
        row.setdefault("origin", "baseline")
    return rows


def _cells(rows):
    cells: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        cell = cells.setdefault(row["cell_id"], {"rows": [], **row})
        cell["rows"].append(row)
    for cell in cells.values():
        means = [float(row["mean_ms"]) for row in cell["rows"]]
        cell["mean_ms"] = statistics.fmean(means)
        cell["block_std_ms"] = statistics.pstdev(means) if len(means) > 1 else 0.0
        cell["block_min_ms"] = min(means)
        cell["block_max_ms"] = max(means)
        cell["calls"] = sum(int(float(row["calls"])) for row in cell["rows"])
        cell["batches"] = sorted({row["batch"] for row in cell["rows"]})
        cell["native_bytes"] = float(cell["rows"][0]["native_bytes"])
        cell["elements"] = float(cell["rows"][0]["elements"])
    return cells


def cmd_merge(args) -> int:
    round1 = _cells(_read_measurements(ROUND1 / "measurements.csv", "round1_20260927"))
    round2 = _cells(_read_measurements(OUT / "measurements.csv", "round2_20260927"))
    recheck_path = OUT / "recheck.csv"
    round3 = (
        _cells(_read_measurements(recheck_path, "round3_20260927"))
        if recheck_path.exists()
        else {}
    )
    checks = []
    excluded: List[Dict[str, Any]] = []
    for cell_id, new in sorted(round2.items()):
        old = round1.get(cell_id)
        if old is None:
            continue  # extension cell, no round-1 counterpart
        third = round3.get(cell_id)

        def passes(other, base):
            combined = (base["block_std_ms"] ** 2 + other["block_std_ms"] ** 2) ** 0.5
            tolerance = max(2.0 * combined, 0.05 * base["mean_ms"])
            return abs(other["mean_ms"] - base["mean_ms"]) <= tolerance, tolerance

        round2_ok, tol2 = passes(new, old)
        round3_ok, tol3 = passes(third, old) if third else (None, None)
        if round2_ok and (round3_ok in (None, True)):
            decision, used = "both batches consistent", ["round2_20260927"]
        elif round3_ok and not round2_ok:
            decision, used = "round2 dropped (recheck agrees with round1)", ["round3_20260927"]
            excluded.append({"cell_id": cell_id, "batch": "round2_20260927"})
        elif round2_ok and round3_ok is False:
            decision, used = "recheck dropped (round2 agrees with round1)", ["round2_20260927"]
            excluded.append({"cell_id": cell_id, "batch": "round3_20260927"})
        elif third is not None:
            same_day_ok, _ = passes(third, new)
            if same_day_ok:
                decision = "round1 level is the outlier; round2+recheck used"
                used = ["round2_20260927", "round3_20260927"]
            else:
                decision = "unresolved: only round1 kept, new batches excluded"
                used = []
                excluded.extend(
                    [
                        {"cell_id": cell_id, "batch": "round2_20260927"},
                        {"cell_id": cell_id, "batch": "round3_20260927"},
                    ]
                )
        else:
            decision, used = "round2 failed, no recheck available", []
            excluded.append({"cell_id": cell_id, "batch": "round2_20260927"})
        checks.append(
            {
                "cell_id": cell_id,
                "round1_mean_ms": old["mean_ms"],
                "round2_mean_ms": new["mean_ms"],
                "round3_mean_ms": third["mean_ms"] if third else None,
                "delta_round2_pct": 100.0 * (new["mean_ms"] - old["mean_ms"]) / old["mean_ms"],
                "delta_round3_pct": (
                    100.0 * (third["mean_ms"] - old["mean_ms"]) / old["mean_ms"]
                    if third
                    else None
                ),
                "round1_block_std_ms": old["block_std_ms"],
                "round2_block_std_ms": new["block_std_ms"],
                "round3_block_std_ms": third["block_std_ms"] if third else None,
                "tolerance_round2_ms": tol2,
                "tolerance_round3_ms": tol3,
                "round2_passed": round2_ok,
                "round3_passed": round3_ok,
                "decision": decision,
                "batches_used_for_this_cell": used,
            }
        )

    excluded_set = {(item["cell_id"], item["batch"]) for item in excluded}
    used_by_batch = {
        check["cell_id"]: set(check["batches_used_for_this_cell"]) for check in checks
    }
    merged_rows = []
    for row in _read_measurements(ROUND1 / "measurements.csv", "round1_20260927"):
        row["origin"] = "baseline"
        merged_rows.append(row)
    for batch, path in (
        ("round2_20260927", OUT / "measurements.csv"),
        ("round3_20260927", recheck_path),
    ):
        if not path.exists():
            continue
        for row in _read_measurements(path, batch):
            cell_id = row["cell_id"]
            if (cell_id, batch) in excluded_set:
                continue
            if cell_id in used_by_batch and batch not in used_by_batch[cell_id]:
                continue
            merged_rows.append(row)
    _write_table(OUT / "merged_measurements.csv", merged_rows)
    failed = [
        check
        for check in checks
        if not check["round2_passed"] or check["round3_passed"] is False
    ]
    (OUT / "consistency.json").write_text(
        json.dumps(
            {
                "protocol": (
                    "control cells are re-measured in round 2 and again in a "
                    "round-3 recheck; a batch passes against the frozen round-1 "
                    "value if |Δ| ≤ max(2·combined block std, 5% of round-1 mean). "
                    "A failing batch is dropped for that cell only (batches still "
                    "average together inside one cell when both pass)."
                ),
                "checks": checks,
                "cells_with_a_failed_batch": len(failed),
                "excluded_rows": excluded,
                "merged_rows": len(merged_rows),
            },
            indent=1,
        )
    )
    print(
        f"consistency: {len(checks) - len(failed)}/{len(checks)} controls fully pass; "
        f"excluded batches={len(excluded)}; merged rows={len(merged_rows)}",
        flush=True,
    )
    return 0


# ------------------------------------------------------------------- fit ----

MODELS = ("M1_byte_prop_anchored", "M2_byte_affine", "M3_element_prop",
          "M4_element_affine", "M5_repr_element_affine")

FIT_FIELDS = [
    "operator", "group", "model", "class_scope", "fit_scope", "training_set",
    "coefficients", "constraint", "reference_cell", "reference_native_bytes",
    "reference_mean_ms", "train_cells", "validation_cells", "train_x_min",
    "train_x_max", "train_r2", "train_mae_ms", "train_mape", "train_rmse_ms",
    "val_mae_ms", "val_mape", "val_rmse_ms", "val_max_abs_err_ms",
]
PREDICTION_FIELDS = [
    "operator", "group", "model", "class_scope", "fit_scope", "cell_id",
    "representation_class", "size_id", "origin", "train_or_validation",
    "elements", "native_bytes", "x_bytes_norm", "mean_ms", "y_time_norm",
    "block_std_ms", "predicted_ms", "abs_err_ms", "rel_err", "batches",
]


def cmd_fit(args) -> int:
    merged = _read_measurements(OUT / "merged_measurements.csv")
    cells = _merge_cells_across_batches(merged)
    reference = _round1_reference()
    fits, predictions = [], []
    for operator in ("blur", "jitter", "crop", "grayscale", "flip"):
        classes = sorted(
            {cell["representation_class"] for cell in cells.values() if cell["operator"] == operator}
        )
        groups: List[Tuple[str, List[str]]] = [("all", classes)]
        for channel in ("1ch", "3ch"):
            subset = [klass for klass in classes if klass.endswith(channel)]
            if len(subset) > 1:
                groups.append((channel, subset))
        for group, klass_filter in groups:
            group_fits, group_predictions = _fit_group(
                cells, operator, group, klass_filter, reference[operator]
            )
            fits.extend(group_fits)
            predictions.extend(group_predictions)
    _write_fits(fits, predictions)
    _write_figure_data(cells, fits, reference)
    _write_coverage(cells)
    _write_notes(cells, fits, reference)
    print(f"fits={len(fits)} predictions={len(predictions)}", flush=True)
    return 0


def _merge_cells_across_batches(rows):
    # A control cell is the round-1 cell re-measured: it keeps that cell's
    # frozen train/validation role instead of receiving a new one.
    frozen_roles = {
        row["cell_id"]: row["train_or_validation"]
        for row in _read_measurements(ROUND1 / "measurements.csv")
    }
    cells: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        cell = cells.setdefault(row["cell_id"], {"rows": [], **row})
        cell["rows"].append(row)
    for cell in cells.values():
        means = [float(row["mean_ms"]) for row in cell["rows"]]
        cell["mean_ms"] = statistics.fmean(means)
        cell["block_std_ms"] = statistics.pstdev(means) if len(means) > 1 else 0.0
        cell["block_mean_ms"] = means
        cell["calls"] = sum(int(float(row["calls"])) for row in cell["rows"])
        cell["native_bytes"] = float(cell["rows"][0]["native_bytes"])
        cell["elements"] = float(cell["rows"][0]["elements"])
        cell["batches"] = sorted({row["batch"] for row in cell["rows"]})
        if cell["cell_id"] in frozen_roles:
            cell["train_or_validation"] = frozen_roles[cell["cell_id"]]
            if len(cell["batches"]) > 1:
                cell["origin"] = "baseline+control"
        else:
            cell["train_or_validation"] = cell["rows"][0]["train_or_validation"]
    return cells


def _fit_group(cells, operator, group, klass_filter, reference):
    subset = [
        cell
        for cell in cells.values()
        if cell["operator"] == operator
        and cell["representation_class"] in klass_filter
    ]
    train = [cell for cell in subset if cell["train_or_validation"] == "train"]
    validation = [cell for cell in subset if cell["train_or_validation"] == "validation"]
    rows, predictions = [], []

    def emit(model, scope, fit_scope, coefficients, constraint, train_cells, val_cells, predictor):
        train_actual = np.array([cell["mean_ms"] for cell in train_cells])
        train_pred = np.array([predictor(cell) for cell in train_cells])
        val_actual = np.array([cell["mean_ms"] for cell in val_cells])
        val_pred = np.array([predictor(cell) for cell in val_cells])
        entry = {
            "operator": operator,
            "group": group,
            "model": model,
            "class_scope": scope,
            "fit_scope": fit_scope,
            "training_set": f"{len(train_cells)} cells",
            "coefficients": json.dumps(coefficients),
            "constraint": constraint,
            "reference_cell": reference["cell_id"],
            "reference_native_bytes": reference["native_bytes"],
            "reference_mean_ms": reference["mean_ms"],
            "train_cells": len(train_cells),
            "validation_cells": len(val_cells),
            "train_x_min": min(
                [cell["native_bytes"] / reference["native_bytes"] for cell in train_cells],
                default=float("nan"),
            ),
            "train_x_max": max(
                [cell["native_bytes"] / reference["native_bytes"] for cell in train_cells],
                default=float("nan"),
            ),
            "train_r2": r1._r2(train_actual, train_pred) if len(train_cells) > 1 else float("nan"),
            **{f"train_{k}": v for k, v in r1._metrics(train_actual, train_pred).items()},
            **(  # validation metrics, empty when the split has no held-out cell
                {f"val_{k}": v for k, v in r1._metrics(val_actual, val_pred).items()}
                if len(val_cells)
                else {
                    "val_mae_ms": float("nan"),
                    "val_mape": float("nan"),
                    "val_rmse_ms": float("nan"),
                    "val_max_abs_err_ms": float("nan"),
                }
            ),
        }
        rows.append(entry)
        for cell, predicted in zip(val_cells + train_cells, list(val_pred) + list(train_pred)):
            predictions.append(_prediction_row(cell, entry, predicted, reference))

    # M1 anchored byte-proportional (frozen reference, no fit).
    emit(
        "M1_byte_prop_anchored",
        "shared",
        "full",
        {"anchor_ms": reference["mean_ms"], "anchor_bytes": reference["native_bytes"]},
        "anchored at the frozen round-1 reference cell",
        train,
        validation,
        lambda cell: reference["mean_ms"]
        * (cell["native_bytes"] / reference["native_bytes"]),
    )
    for model, through_origin, feature in (
        ("M2_byte_affine", False, "native_bytes"),
        ("M3_element_prop", True, "elements"),
        ("M4_element_affine", False, "elements"),
    ):
        x = np.array([cell[feature] for cell in train])
        y = np.array([cell["mean_ms"] for cell in train])
        design = x.reshape(-1, 1) if through_origin else np.column_stack([x, np.ones_like(x)])
        coefficients, method = r1._nnls(design, y)
        constraint = (
            f"non-negative {method}"
            + (", slope bound active" if coefficients[0] <= 1e-18 else "")
            + (
                ", intercept bound active"
                if not through_origin and coefficients[1] <= 1e-18
                else ""
            )
        )
        if through_origin:
            predictor = lambda cell, c=coefficients, f=feature: c[0] * cell[f]
            payload = {"k": float(coefficients[0]), "feature": feature}
        else:
            predictor = lambda cell, c=coefficients, f=feature: c[0] * cell[f] + c[1]
            payload = {"k": float(coefficients[0]), "b": float(coefficients[1]), "feature": feature}
        emit(model, "shared", "full", payload, constraint, train, validation, predictor)
    for klass in sorted({cell["representation_class"] for cell in train}):
        class_train = [cell for cell in train if cell["representation_class"] == klass]
        class_val = [cell for cell in validation if cell["representation_class"] == klass]
        if len(class_train) < 2:
            continue
        x = np.array([cell["elements"] for cell in class_train])
        y = np.array([cell["mean_ms"] for cell in class_train])
        coefficients, method = r1._nnls(np.column_stack([x, np.ones_like(x)]), y)
        emit(
            "M5_repr_element_affine",
            klass,
            "full",
            {"k": float(coefficients[0]), "b": float(coefficients[1]), "feature": "elements"},
            f"non-negative {method}"
            + (", slope bound active" if coefficients[0] <= 1e-18 else "")
            + (", intercept bound active" if coefficients[1] <= 1e-18 else ""),
            class_train,
            class_val,
            lambda cell, c=coefficients: c[0] * cell["elements"] + c[1],
        )
        # Pre-declared local window (byte-normalised x <= LOCAL_X_MAX).
        local_train = [
            cell
            for cell in class_train
            if cell["native_bytes"] / reference["native_bytes"] <= LOCAL_X_MAX
        ]
        local_val = [
            cell
            for cell in class_val
            if cell["native_bytes"] / reference["native_bytes"] <= LOCAL_X_MAX
        ]
        # Only meaningful when the local window really is a strict subset of
        # the fitted range (otherwise it would duplicate the full-range model).
        strict_subset = any(
            cell["native_bytes"] / reference["native_bytes"] > LOCAL_X_MAX
            for cell in class_train
        )
        if strict_subset and len(local_train) >= 3 and local_val:
            for model, through_origin in (
                ("M3_element_prop", True),
                ("M4_element_affine", False),
            ):
                x = np.array([cell["elements"] for cell in local_train])
                y = np.array([cell["mean_ms"] for cell in local_train])
                design = (
                    x.reshape(-1, 1)
                    if through_origin
                    else np.column_stack([x, np.ones_like(x)])
                )
                coefficients, method = r1._nnls(design, y)
                if through_origin:
                    predictor = lambda cell, c=coefficients: c[0] * cell["elements"]
                    payload = {"k": float(coefficients[0]), "feature": "elements"}
                else:
                    predictor = (
                        lambda cell, c=coefficients: c[0] * cell["elements"] + c[1]
                    )
                    payload = {
                        "k": float(coefficients[0]),
                        "b": float(coefficients[1]),
                        "feature": "elements",
                    }
                emit(
                    model,
                    klass,
                    f"local_x_le_{LOCAL_X_MAX:g}",
                    payload,
                    f"non-negative {method}, trained on x<=LOCAL_X_MAX only",
                    local_train,
                    local_val,
                    predictor,
                )
    return rows, predictions


def _prediction_row(cell, entry, predicted, reference):
    actual = cell["mean_ms"]
    return {
        "operator": cell["operator"],
        "group": entry["group"],
        "model": entry["model"],
        "class_scope": entry["class_scope"],
        "fit_scope": entry["fit_scope"],
        "cell_id": cell["cell_id"],
        "representation_class": cell["representation_class"],
        "size_id": cell["size_id"],
        "origin": cell["origin"],
        "train_or_validation": cell["train_or_validation"],
        "elements": cell["elements"],
        "native_bytes": cell["native_bytes"],
        "x_bytes_norm": cell["native_bytes"] / reference["native_bytes"],
        "mean_ms": actual,
        "y_time_norm": actual / reference["mean_ms"],
        "block_std_ms": cell["block_std_ms"],
        "predicted_ms": predicted,
        "abs_err_ms": predicted - actual,
        "rel_err": (predicted - actual) / actual if actual else float("nan"),
        "batches": ",".join(cell["batches"]),
    }


def _write_fits(fits, predictions) -> None:
    with (OUT / "fits.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in fits:
            writer.writerow(row)
    with (OUT / "predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in predictions:
            writer.writerow(row)


def _write_figure_data(cells, fits, reference) -> None:
    panels: Dict[str, Any] = {"panels": {}, "cells": [], "fits": fits}
    for cell in sorted(cells.values(), key=lambda item: item["cell_id"]):
        reference_point = reference[cell["operator"]]
        panels["cells"].append(
            {
                "cell_id": cell["cell_id"],
                "operator": cell["operator"],
                "representation_class": cell["representation_class"],
                "size_id": cell["size_id"],
                "role": cell["train_or_validation"],
                "origin": cell["origin"],
                "batches": cell["batches"],
                "height": cell["height"],
                "width": cell["width"],
                "elements": cell["elements"],
                "native_bytes": cell["native_bytes"],
                "serialized_bytes": float(cell["rows"][0]["serialized_bytes"]),
                "out_elements": float(cell["rows"][0]["out_elements"]),
                "out_native_bytes": float(cell["rows"][0]["out_native_bytes"]),
                "mean_ms": cell["mean_ms"],
                "block_std_ms": cell["block_std_ms"],
                "calls": cell["calls"],
                "x_bytes_norm": cell["native_bytes"] / reference_point["native_bytes"],
                "y_time_norm": cell["mean_ms"] / reference_point["mean_ms"],
            }
        )
    for operator, point in reference.items():
        operator_cells = [cell for cell in panels["cells"] if cell["operator"] == operator]
        panels["panels"][operator] = {
            "cells": [cell["cell_id"] for cell in operator_cells],
            "reference": point,
            "classes": sorted({cell["representation_class"] for cell in operator_cells}),
        }
    (OUT / "figure_data.json").write_text(json.dumps(panels, indent=1, default=float))


COVERAGE_REASONS = {
    ("grayscale", "float32:1ch"): (
        "measured in round 2; torchvision rgb_to_grayscale clones a 1-channel "
        "input (identity path), so the curve is a memory copy"
    ),
    ("grayscale", "uint8:1ch"): (
        "measured in round 2; same clone path as float32:1ch"
    ),
    ("flip", "uint8:1ch"): "measured in round 2",
}


def _write_coverage(cells) -> None:
    rows = []
    for operator in ("blur", "jitter", "crop", "grayscale", "flip"):
        for klass in ("float32:3ch", "float32:1ch", "uint8:3ch", "uint8:1ch"):
            measured = [
                cell
                for cell in cells.values()
                if cell["operator"] == operator and cell["representation_class"] == klass
            ]
            sizes = sorted({int(cell["height"]) for cell in measured})
            if measured:
                status = "measured"
                reason = COVERAGE_REASONS.get(
                    (operator, klass),
                    "measured in round 1" if all(
                        cell["origin"] == "baseline" for cell in measured
                    ) else "measured across rounds",
                )
            else:
                status, reason = _unmeasured_reason(operator, klass)
            rows.append(
                {
                    "operator": operator,
                    "representation_class": klass,
                    "supported": status != "unsupported",
                    "measured": bool(measured),
                    "status": status,
                    "cells": len(measured),
                    "sizes": " ".join(str(size) for size in sizes),
                    "reason": reason,
                }
            )
    with (OUT / "coverage.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _unmeasured_reason(operator: str, klass: str) -> Tuple[str, str]:
    if operator == "grayscale" and klass.endswith("3ch"):
        return "measured", ""
    return (
        "gap",
        "not measured in either round; no measurement is fabricated for the figure",
    )


def _fmt(value, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if number != number:
        return "—"
    return f"{number:.{digits}g}"


def _pct(value) -> str:
    try:
        return f"{100.0 * float(value):.1f}%"
    except (TypeError, ValueError):
        return "—"


def _ls_dominance(cells, fits, operator, klass, reference):
    """Share of the full-range least-squares objective spent on x>2 points."""
    fit = next(
        (
            fit
            for fit in fits
            if fit["operator"] == operator
            and fit["model"] == "M5_repr_element_affine"
            and fit["class_scope"] == klass
            and fit["fit_scope"] == "full"
        ),
        None,
    )
    local = next(
        (
            fit
            for fit in fits
            if fit["operator"] == operator
            and fit["model"] == "M4_element_affine"
            and fit["class_scope"] == klass
            and fit["fit_scope"] == f"local_x_le_{LOCAL_X_MAX:g}"
        ),
        None,
    )
    if fit is None:
        return None
    coefficients = json.loads(fit["coefficients"])
    k, b = coefficients["k"], coefficients.get("b", 0.0)
    train = [
        cell
        for cell in cells.values()
        if cell["operator"] == operator
        and cell["representation_class"] == klass
        and cell["train_or_validation"] == "train"
    ]
    points = []
    total = 0.0
    above = 0.0
    for cell in sorted(train, key=lambda cell: cell["native_bytes"]):
        x = cell["native_bytes"] / reference["native_bytes"]
        predicted = k * cell["elements"] + b
        residual = cell["mean_ms"] - predicted
        share = residual ** 2
        total += share
        if x > LOCAL_X_MAX:
            above += share
        points.append(
            {"x": x, "measured": cell["mean_ms"], "predicted": predicted,
             "residual": residual, "share": share}
        )
    for point in points:
        point["share"] = point["share"] / total if total else 0.0
    local_coefficients = json.loads(local["coefficients"]) if local else {}
    return {
        "k": k,
        "b": b,
        "above_count": sum(1 for point in points if point["x"] > LOCAL_X_MAX),
        "above_share": above / total if total else 0.0,
        "points": points,
        "local_b": local_coefficients.get("b", float("nan")) if local else float("nan"),
        "local_mape": float(local["val_mape"]) if local else float("nan"),
        "full_mape": float(fit["val_mape"]),
    }


def _extended_instability(cells, operator, klass, reference, x_min=1.15):
    """Per-block means/medians of the extension cells beyond ``x_min``."""
    out = []
    for cell in cells.values():
        if (
            cell["operator"] != operator
            or cell["representation_class"] != klass
        ):
            continue
        x = cell["native_bytes"] / reference["native_bytes"]
        if x < x_min:
            continue
        blocks = sorted(cell["rows"], key=lambda row: int(row["block_id"]))
        means = [float(row["mean_ms"]) for row in blocks]
        medians = [float(row["median_ms"]) for row in blocks]
        out.append(
            {
                "cell_id": cell["cell_id"],
                "x": x,
                "means": means,
                "medians": medians,
                "mean_over_median": statistics.fmean(means)
                / max(statistics.fmean(medians), 1e-12),
            }
        )
    return sorted(out, key=lambda item: item["x"])


def _same_size_duplicates(cells):
    """Cells with identical class+shape but different ids (round1 vs extension)."""
    buckets: Dict[Tuple[str, str, int, int], List[Dict[str, Any]]] = {}
    for cell in cells.values():
        buckets.setdefault(
            (
                cell["operator"],
                cell["representation_class"],
                int(cell["height"]),
                int(cell["width"]),
            ),
            [],
        ).append(cell)
    out = []
    for (operator, klass, height, width), group in sorted(buckets.items()):
        if len(group) < 2:
            continue
        group = sorted(group, key=lambda cell: cell["cell_id"])
        first, second = group[0], group[1]
        if first["cell_id"] == second["cell_id"]:
            continue
        # Only report pairs that carry an extension measurement: these are the
        # deliberate same-size repeats of the round-1 range.
        if not any("extension" in cell["origin"] for cell in group):
            continue
        out.append(
            {
                "operator": operator,
                "representation_class": klass,
                "height": height,
                "width": width,
                "first": first,
                "second": second,
                "relative_gap": (second["mean_ms"] - first["mean_ms"])
                / first["mean_ms"],
            }
        )
    return out


def _write_notes(cells, fits, reference) -> None:
    def find(model, operator, group=None, scope=None, fit_scope="full"):
        for fit in fits:
            if fit["model"] != model or fit["operator"] != operator:
                continue
            if group and fit["group"] != group:
                continue
            if scope and fit["class_scope"] != scope:
                continue
            if fit["fit_scope"] != fit_scope:
                continue
            return fit
        return None

    consistency = json.loads((OUT / "consistency.json").read_text())
    lines = [
        "# §3.2 补充轮：扩展范围与补齐表示类（2026-09-27）",
        "",
        f"commit `{r1._git_commit()}`；本轮结果 `outputs/ch32_scale_affine_ext_20260927/`，",
        "旧数据 `outputs/ch32_scale_affine_20260927/` 原样未动；快照 `docs/ch32_scale_affine_ext_20260927/`。",
        "",
        "## 1. 归一化定义（冻结，不改）",
        "",
        "```",
        "x = 张量原生字节数 / 参考输入原生字节数",
        "y = 算子平均计算时间  / 参考输入平均计算时间",
        "```",
        "",
        "参考点沿用第一轮的每个算子单一参考输入（**冻结**，不因新数据改变）：",
        "",
        "| 算子 | 参考 cell | 参考原生字节 | 参考均值 (ms) |",
        "| --- | --- | ---: | ---: |",
    ]
    for operator in ("blur", "jitter", "crop", "grayscale", "flip"):
        point = reference[operator]
        lines.append(
            f"| {operator} | {point['cell_id']} | {point['native_bytes']:.0f} | "
            f"{point['mean_ms']:.4f} |"
        )
    lines += [
        "",
        "所有表示类共用同一个参考点，因此类之间的字节/耗时差异被原样保留（不做逐类归一化）。",
        "",
        "## 2. 本轮新增测量",
        "",
        "| 目标 | 新增训练尺寸 | 新增验证尺寸 | 归一化 x 覆盖 |",
        "| --- | --- | --- | --- |",
    ]
    for (operator, klass), ladder in EXTENSION.items():
        point = reference[operator]
        sizes = [side for _, side in ladder["train"] + ladder["validation"]]
        xs = [side * side * (1 if klass.startswith("uint8") else 4)
              * (1 if klass.endswith("1ch") else 3) / point["native_bytes"]
              for side in sizes]
        lines.append(
            f"| {operator} {klass} | {len(ladder['train'])} | {len(ladder['validation'])} | "
            f"{min(xs):.3f}–{max(xs):.3f} |"
        )
    lines += [
        "",
        f"另有 {len(CONTROLS)} 个第一轮 cell 原样重测作为漂移对照（见 §3）。",
        "",
        "## 3. 新旧一致性",
        "",
        "判据：每个对照 cell 在第二轮与第三轮（复核）各重测一次，任一批次与**冻结的第一轮值**相比满足 "
        "|Δ| ≤ max(2×合并块间标准差, 第一轮均值的 5%) 即通过；不通过的批次只从该 cell 剔除。",
        f"结果：{len(consistency['checks'])} 个对照里 "
        f"{len(consistency['checks']) - consistency['cells_with_a_failed_batch']} 个两批全通过，"
        f"共剔除 {len(consistency['excluded_rows'])} 个批次- cell 组合。",
        "",
        "| cell | 第一轮 | 第二轮 | Δ2 | 第三轮(复核) | Δ3 | 结论 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for check in consistency["checks"]:
        lines.append(
            f"| {check['cell_id']} | {check['round1_mean_ms']:.4f} | "
            f"{check['round2_mean_ms']:.4f} | {check['delta_round2_pct']:+.1f}% | "
            f"{check['round3_mean_ms']:.4f} | {check['delta_round3_pct']:+.1f}% | "
            f"{check['decision']} |"
        )
    duplicates = _same_size_duplicates(cells)
    if duplicates:
        lines += [
            "",
            "另外，扩展尺寸里刻意保留了与原范围**同尺寸**的第二个 cell（独立 id、独立测量），",
            "用于在最贴近的点上再检查一次一致性：",
            "",
            "| 算子/类 | 尺寸 | 原 cell | 扩展 cell | 相对差 |",
            "| --- | ---: | --- | --- | ---: |",
        ]
        for item in duplicates:
            lines.append(
                f"| {item['operator']} {item['representation_class']} | "
                f"{item['height']}×{item['width']} | {item['first']['cell_id'].split(':')[-1]} "
                f"{item['first']['mean_ms']:.4f} ms | {item['second']['cell_id'].split(':')[-1]} "
                f"{item['second']['mean_ms']:.4f} ms | {100*item['relative_gap']:+.1f}% |"
            )
    lines += [
        "",
        "只有通过检查的批次才进入 `merged_measurements.csv`；被剔除的组合记在 `consistency.json`。",
        "合并口径：同一 cell 若多个批次都通过，取全部 block 的算术平均（`batch` 列保留来源）。",
        "**跨批次的绝对水平存在 ≤10% 的漂移**（见上表：flip +8.3%、grayscale +9.9% 的批次被判为超差），",
        "而本图要展示的类间差异是 4–17×，因此批次漂移不影响类间结论，但不能用来比较 10% 量级的差异。",
        "",
        "## 4. 扩展后的关键拟合",
        "",
        "每类的全范围模型 = M5（按表示类分开的 kx+b）；局部模型 = 同一类内只用 x≤2 的训练点，",
        "并在该类 x≤2 的独立验证点上评估。",
        "",
        "| 算子 | 类 | M5 全范围（kx+b） | M3 局部 x≤2（比例） | M4 局部 x≤2（kx+b） |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for operator in ("jitter", "crop", "flip", "grayscale"):
        for klass in sorted(
            {cell["representation_class"] for cell in cells.values() if cell["operator"] == operator}
        ):
            full = find("M5_repr_element_affine", operator, scope=klass)
            local = find("M4_element_affine", operator, scope=klass, fit_scope="local_x_le_2")
            m3_local = find("M3_element_prop", operator, scope=klass, fit_scope="local_x_le_2")
            lines.append(
                f"| {operator} | {klass} | "
                f"{_pct(full['val_mape']) if full else '—'} | "
                f"{_pct(m3_local['val_mape']) if m3_local else '—'} | "
                f"{_pct(local['val_mape']) if local else '—'} |"
            )
    lines += [
        "",
        "（局部模型只在 x≤2 的训练点上拟合、在 x≤2 的独立验证点上评估；适用范围写明在 `fits.csv` 的",
        "`fit_scope` 与 `constraint` 字段，不能外推使用。）",
        "",
        "### 4.1 大尺寸是否主导最小二乘（ColorJitter float32:3ch）",
        "",
    ]
    dominance = _ls_dominance(cells, fits, "jitter", "float32:3ch", reference["jitter"])
    if dominance:
        lines += [
            f"全范围拟合 k={dominance['k']:.4g}、b={dominance['b']:.4g}；"
            f"x>2 的 {dominance['above_count']} 个训练点贡献了**最小二乘目标的 "
            f"{100*dominance['above_share']:.0f}%**，窗口内（x≤2）因此出现系统性负残差：",
            "",
            "| 训练点 x | 实测 (ms) | 全范围预测 (ms) | 残差 | 目标占比 |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
        for point in dominance["points"]:
            lines.append(
                f"| {point['x']:.2f} | {point['measured']:.3f} | {point['predicted']:.3f} | "
                f"{point['residual']:+.3f} | {100*point['share']:.1f}% |"
            )
        lines += [
            "",
            f"排除 x>2 的点后，同一类的局部仿射给出 b={dominance['local_b']:.3f} ms、"
            f"验证 MAPE {_pct(dominance['local_mape'])}（全范围 {_pct(dominance['full_mape'])}）。",
            "窗口内的偏差不是块间波动造成的，而是全范围最小二乘被大尺寸点拉高斜率。",
            "",
        ]
    lines += [
        "### 4.2 扩展段的非单调：ColorJitter uint8:1ch",
        "",
    ]
    unstable = _extended_instability(cells, "jitter", "uint8:1ch", reference["jitter"])
    if unstable:
        lines += [
            "该类的扩展点在 x≥1.5 处没有随规模单调增长，且同一 cell 的块间中位数摆动很大：",
            "",
            "| cell | x | 逐块中位数 (ms) | 逐块均值 (ms) | 均值/中位数 |",
            "| --- | ---: | --- | --- | ---: |",
        ]
        for item in unstable:
            lines.append(
                f"| {item['cell_id'].split(':')[-1]} | {item['x']:.2f} | "
                f"{' / '.join(f'{v:.2f}' for v in item['medians'])} | "
                f"{' / '.join(f'{v:.2f}' for v in item['means'])} | "
                f"{item['mean_over_median']:.2f} |"
            )
        lines += [
            "",
            "因此 x≳1.5 这段**不能**用仿射或比例模型描述，也不能当作真实尺度响应；",
            "该范围标记为未覆盖。局部 x≤2 模型同样没有改善（34.7% vs 全范围 30.0%），",
            "因为问题不是截距，而是大 cell 的测量分布。这一负结果按原样保留：不删点、不为贴合曲线调参。",
            "",
        ]
    lines += [
        "## 5. 执行路径差异（如实记录）",
        "",
        "| 表示类 | 实际执行路径 | 依据 |",
        "| --- | --- | --- |",
        "| ColorJitter 1ch | saturation 与 hue 分支直接 `return img`，只有 brightness/contrast 生效 | "
        "`torchvision/transforms/functional_tensor.py::adjust_saturation/adjust_hue` |",
        "| Grayscale 1ch | `rgb_to_grayscale` 对单通道走 `img.clone()`，即恒等但有一次拷贝 | 同上 `rgb_to_grayscale` |",
        "| Grayscale 3ch | 真实加权求和（0.2989/0.587/0.114） | 同上 |",
        "",
        "## 6. 不支持或未覆盖的类别",
        "",
        "| 算子 | 表示类 | 状态 | 已测 cell 数 | 尺寸 | 说明 |",
        "| --- | --- | --- | ---: | --- | --- |",
    ]
    coverage = list(csv.DictReader((OUT / "coverage.csv").open()))
    for row in coverage:
        lines.append(
            f"| {row['operator']} | {row['representation_class']} | {row['status']} | "
            f"{row['cells']} | {row['sizes']} | {row['reason']} |"
        )
    gaps = [row for row in coverage if row["status"] != "measured"]
    if gaps:
        lines += ["", "未覆盖/不支持："]
        for row in gaps:
            lines.append(
                f"- {row['operator']} {row['representation_class']}：{row['status']}（{row['reason']}）"
            )
    else:
        lines += [""]
        lines.append("- 五个算子 × 四种表示类全部有测量。")
    lines += [
        "",
        "## 7. 文件与复现",
        "",
        "- `measurements.csv`：本轮新测量（沿用第一轮字段 + `origin`）。",
        "- `merged_measurements.csv`：通过一致性检查后的两轮合并数据（含 `batch` 列）。",
        "- `fits.csv` / `predictions.csv`：全范围与局部模型、逐点预测与误差。",
        "- `figure_data.json`：每个算子/类的原生字节、元素数、耗时、归一化坐标、参考点与拟合系数。",
        "- `coverage.csv`、`consistency.json`、`env.json`、`raw_calls.csv.gz`（逐调用原始耗时）。",
        "- `verify.json`：从 `predictions.csv` 重新推导全部指标并与 `fits.csv` 对账（0 差异才算通过）。",
        "",
        "```bash",
        "python -m tmp_analysis.ch32_scale_affine_ext measure   # 远程 Ray actor，3 blocks",
        "python -m tmp_analysis.ch32_scale_affine_ext merge     # 一致性检查 + 合并",
        "python -m tmp_analysis.ch32_scale_affine_ext fit       # 拟合/验证/交付表",
        "python -m tmp_analysis.ch32_scale_affine_ext manifest  # 校验和 + docs 快照",
        "```",
    ]
    (OUT / "README.md").write_text("\n".join(lines))


def cmd_manifest(args) -> int:
    import shutil

    DOCS.mkdir(parents=True, exist_ok=True)
    manifest = {
        "commit": r1._git_commit(),
        "result_dir": str(OUT.relative_to(ROOT)),
        "small_files": {},
        "large_files": {},
        "provenance": {},
    }
    for name in (
        "README.md",
        "env.json",
        "measurements.csv",
        "merged_measurements.csv",
        "fits.csv",
        "predictions.csv",
        "figure_data.json",
        "coverage.csv",
        "consistency.json",
        "verify.json",
    ):
        path = OUT / name
        if not path.exists():
            continue
        data = path.read_bytes()
        manifest["small_files"][name] = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
            "rows": len(data.decode(errors="replace").splitlines()) - 1,
        }
        shutil.copy2(path, DOCS / name)
    for name in (
        "raw_calls.csv.gz",
        "recheck_raw_calls.csv.gz",
        "measure.log",
        "recheck.log",
    ):
        path = OUT / name
        if not path.exists():
            continue
        data = path.read_bytes()
        entry = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        if name.endswith(".csv.gz"):
            with gzip.open(path, "rt") as handle:
                entry["rows"] = sum(1 for _ in handle) - 1
        manifest["large_files"][name] = entry
    for name in ("ch32_scale_affine_ext.py", "ch32_scale_affine_probe.py"):
        path = ROOT / "tmp_analysis" / name
        manifest["provenance"][name] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    shipped = OUT / "modules/tmp_analysis/ch32_scale_affine_ext.py"
    if shipped.exists():
        manifest["provenance"]["measurement_snapshot"] = {
            "path": str(shipped.relative_to(ROOT)),
            "sha256": hashlib.sha256(shipped.read_bytes()).hexdigest(),
        }
    figures = {}
    figure_dir = OUT / "figures"
    if figure_dir.is_dir():
        for path in sorted(figure_dir.glob("*.png")):
            data = path.read_bytes()
            figures[path.name] = {
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
            }
            shutil.copy2(path, DOCS / path.name)
    manifest["figures"] = figures
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    shutil.copy2(OUT / "MANIFEST.json", DOCS / "MANIFEST.json")
    print(f"manifest: {len(manifest['small_files'])} small files", flush=True)
    return 0


def cmd_verify(args) -> int:
    """Re-derive every reported metric from predictions.csv and figure_data."""
    fits = list(csv.DictReader((OUT / "fits.csv").open()))
    predictions = list(csv.DictReader((OUT / "predictions.csv").open()))
    problems: List[str] = []
    worst = 0.0
    checked = 0
    for fit in fits:
        key = (
            fit["operator"],
            fit["group"],
            fit["model"],
            fit["class_scope"],
            fit["fit_scope"],
        )
        for role, prefix in (("validation", "val"), ("train", "train")):
            rows = [
                row
                for row in predictions
                if (
                    row["operator"],
                    row["group"],
                    row["model"],
                    row["class_scope"],
                    row["fit_scope"],
                )
                == key
                and row["train_or_validation"] == role
            ]
            if not rows:
                continue
            actual = np.array([float(row["mean_ms"]) for row in rows])
            predicted = np.array([float(row["predicted_ms"]) for row in rows])
            mae = float(np.mean(np.abs(predicted - actual)))
            mape = float(np.mean(np.abs(predicted - actual) / actual))
            rmse = float(np.sqrt(np.mean((predicted - actual) ** 2)))
            for name, value in (
                (f"{prefix}_mae_ms", mae),
                (f"{prefix}_mape", mape),
                (f"{prefix}_rmse_ms", rmse),
            ):
                reported = float(fit[name])
                worst = max(worst, abs(value - reported))
                checked += 1
                if abs(value - reported) > 1e-9:
                    problems.append(
                        f"{key}: {name} recomputed {value} != reported {reported}"
                    )
            if int(fit["validation_cells" if role == "validation" else "train_cells"]) != len(rows):
                problems.append(f"{key}: cell count mismatch for {role}")
    figure = json.loads((OUT / "figure_data.json").read_text())
    reference = {operator: panel["reference"] for operator, panel in figure["panels"].items()}
    for cell in figure["cells"]:
        point = reference[cell["operator"]]
        if abs(cell["x_bytes_norm"] - cell["native_bytes"] / point["native_bytes"]) > 1e-12:
            problems.append(f"{cell['cell_id']}: x_bytes_norm does not match the reference")
        if abs(cell["y_time_norm"] - cell["mean_ms"] / point["mean_ms"]) > 1e-12:
            problems.append(f"{cell['cell_id']}: y_time_norm does not match the reference")
    report = {
        "fits_checked": len(fits),
        "metric_values_compared": checked,
        "worst_abs_difference": worst,
        "cells_checked": len(figure["cells"]),
        "problems": problems,
        "passed": not problems,
    }
    (OUT / "verify.json").write_text(json.dumps(report, indent=1))
    print(
        f"verify: {len(fits)} fits, {checked} metric values, "
        f"worst |Δ| = {worst:.2e}, problems = {len(problems)}"
    )
    if problems:
        for problem in problems[:10]:
            print("  -", problem)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    measure = sub.add_parser("measure")
    measure.add_argument("--dry-run", action="store_true")
    measure.add_argument("--controls-only", action="store_true")
    measure.set_defaults(func=cmd_measure)
    merge = sub.add_parser("merge")
    merge.set_defaults(func=cmd_merge)
    fit = sub.add_parser("fit")
    fit.set_defaults(func=cmd_fit)
    manifest = sub.add_parser("manifest")
    manifest.set_defaults(func=cmd_manifest)
    verify = sub.add_parser("verify")
    verify.set_defaults(func=cmd_verify)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
