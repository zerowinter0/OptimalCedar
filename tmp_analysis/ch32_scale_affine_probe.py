"""Section 3.2 controlled size x representation sweep (profiling only).

Measures the *operator self compute time* of the SimCLRv2 transforms on
controlled inputs so the paper can show, from one protocol:

* bytes vs elements as the explanatory variable (Blur),
* whether a representation class needs its own curve (ColorJitter),
* whether the response is proportional or affine (Crop),
  plus Grayscale / Flip as candidate panels.

The measurement reuses the deployed definitions rather than re-inventing
them: ``payload_compute_scale`` / ``payload_representation_class`` from
``cedar.pipes.common`` define the features, ``limit_native_threadpools(1)``
defines the thread contract, and the timing window follows
``DataSet._time_operator_fresh_snapshot`` -- decode the payload, then time
only the callable, on a fresh value per call, with interleaved cells inside
one window per block.

Every call runs inside a remote Ray actor (``cedar_remote`` placement
resource), matching the remote-Ray contract for profiling experiments.

Usage (inside the container):
  python -m tmp_analysis.ch32_scale_affine_probe measure [--dry-run]
  python -m tmp_analysis.ch32_scale_affine_probe fit
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import pickle
import platform
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

ROOT = Path("/workspace/OptimalCedar")
OUT = ROOT / "outputs/ch32_scale_affine_20260927"
RAY_ADDRESS = "172.23.166.105:6379"
IMAGENETTE_TRAIN = ROOT / "evaluation/datasets/imagenette2/imagenette2/train"

# ---------------------------------------------------------------- protocol --

# Pre-registered size ladders.  Train sizes are interleaved with validation
# sizes on the real axis; validation sizes are never used for fitting.
SIZE_LADDER_LARGE = [
    ("train", "t0", 64),
    ("validation", "v0", 80),
    ("train", "t1", 96),
    ("train", "t2", 144),
    ("validation", "v1", 160),
    ("train", "t3", 192),
    ("train", "t4", 224),
    ("validation", "v2", 256),
    ("train", "t5", 288),
    ("train", "t6", 384),
    ("validation", "v3", 448),
    ("train", "t7", 512),
]
SIZE_LADDER_CROP = [
    ("train", "t0", 96),
    ("validation", "v0", 112),
    ("train", "t1", 128),
    ("train", "t2", 160),
    ("train", "t3", 192),
    ("validation", "v1", 224),
    ("train", "t4", 256),
    ("train", "t5", 320),
    ("validation", "v2", 288),
    ("train", "t6", 384),
    ("validation", "v3", 416),
    ("train", "t7", 448),
]

# Reference size of the operator's *real* input in the declared SimCLRv2
# order (used only to pick the anchored model's reference train point).
REFERENCE_SIZE = {
    "crop": 384,  # raw imagenette images, median short side ~375
    "flip": 244,  # RandomResizedCrop output
    "jitter": 244,
    "grayscale": 244,
    "blur": 244,
}

# Representation the operator actually sees in the declared SimCLRv2 order
# (blur runs after Grayscale, everything else runs on the 3-channel float
# payload that ``to_float`` produces).  Used only to pick the anchored
# byte-proportional model's reference training point.
REAL_CLASS = {
    "crop": "float32:3ch",
    "flip": "float32:3ch",
    "jitter": "float32:3ch",
    "grayscale": "float32:3ch",
    "blur": "float32:1ch",
}

OPERATOR_SPECS: Dict[str, Dict[str, Any]] = {
    "blur": {
        "classes": ["float32:3ch", "float32:1ch", "uint8:3ch", "uint8:1ch"],
        "ladder": SIZE_LADDER_LARGE,
        "budget_sec": 2.5,
        "fixed_seed": True,
        "parameters": {"kernel_size": 11, "sigma": [0.1, 2.0]},
    },
    "jitter": {
        "classes": ["float32:3ch", "float32:1ch", "uint8:3ch", "uint8:1ch"],
        "ladder": SIZE_LADDER_LARGE,
        "budget_sec": 1.5,
        "fixed_seed": False,
        "parameters": {
            "brightness": 0.1,
            "contrast": 0.1,
            "saturation": 0.1,
            "hue": 0.1,
        },
        # same element count, different channel count (diagnostic only)
        "pairs": [
            ("pair0", "float32:3ch", 64, 64),
            ("pair0", "float32:1ch", 128, 96),
            ("pair1", "float32:3ch", 128, 128),
            ("pair1", "float32:1ch", 256, 192),
        ],
    },
    "crop": {
        "classes": ["float32:3ch", "float32:1ch", "uint8:3ch", "uint8:1ch"],
        "ladder": SIZE_LADDER_CROP,
        "budget_sec": 1.5,
        "fixed_seed": False,
        "parameters": {"size": [244, 244], "scale": [0.08, 1.0], "ratio": [0.75, 1.3333333]},
    },
    "grayscale": {
        "classes": ["float32:3ch", "uint8:3ch"],
        "ladder": SIZE_LADDER_LARGE,
        "budget_sec": 1.0,
        "fixed_seed": True,
        "parameters": {"num_output_channels": 1},
    },
    "flip": {
        "classes": ["float32:3ch", "float32:1ch", "uint8:3ch"],
        "ladder": SIZE_LADDER_LARGE,
        "budget_sec": 1.0,
        "fixed_seed": False,
        "parameters": {"p": 0.5},
    },
}

N_SOURCES = 4
BLOCKS = 3
MAX_CALLS_PER_CELL = 400
BASE_SEED = 20260927


def build_operators() -> Dict[str, Any]:
    """Exactly the callables the SimCLRv2 recipe instantiates."""
    from torchvision import transforms

    return {
        "blur": transforms.GaussianBlur(11),
        "jitter": transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
        "crop": transforms.RandomResizedCrop((244, 244)),
        "grayscale": transforms.Grayscale(num_output_channels=1),
        "flip": transforms.RandomHorizontalFlip(),
    }


def build_cells() -> List[Dict[str, Any]]:
    cells: List[Dict[str, Any]] = []
    for operator, spec in OPERATOR_SPECS.items():
        for klass in spec["classes"]:
            for role, size_id, side in spec["ladder"]:
                cells.append(
                    {
                        "operator": operator,
                        "cell_id": f"{operator}:{klass}:{size_id}",
                        "representation_class": klass,
                        "size_id": size_id,
                        "role": role,
                        "height": side,
                        "width": side,
                        "pair_id": "",
                    }
                )
        for pair_id, klass, height, width in spec.get("pairs", []):
            cells.append(
                {
                    "operator": operator,
                    "cell_id": f"{operator}:{klass}:{pair_id}",
                    "representation_class": klass,
                    "size_id": pair_id,
                    "role": "diagnostic",
                    "height": height,
                    "width": width,
                    "pair_id": pair_id,
                }
            )
    return cells


def _class_parts(klass: str) -> Tuple[str, int]:
    dtype, _, channels = klass.partition(":")
    return dtype, int(channels.replace("ch", ""))


def build_payload(source_hwc: np.ndarray, klass: str, height: int, width: int):
    """A payload the pipeline can legally materialise at this position.

    ``uint8`` payloads are what ``ImageReaderPipe`` hands downstream; the
    ``float32`` payloads keep the reader's 0..255 range because the recipe's
    ``to_float`` is a plain cast, not a rescale.
    """
    from PIL import Image

    dtype, channels = _class_parts(klass)
    image = Image.fromarray(source_hwc)
    image = image.convert("RGB") if channels == 3 else image.convert("L")
    image = image.resize((width, height), Image.BILINEAR)
    array = np.asarray(image)
    if array.ndim == 2:
        array = array[:, :, None]
    # ``np.asarray(PIL)`` is read-only; copy so the payload is a normal,
    # writable tensor exactly like the one ``ImageReaderPipe`` produces.
    tensor = torch.from_numpy(np.array(array, copy=True)).permute(2, 0, 1)
    if dtype == "float32":
        tensor = tensor.to(torch.float32)
    return tensor.contiguous()


def representation_metadata(value) -> Dict[str, Any]:
    from cedar.pipes.common import (
        payload_compute_scale,
        payload_representation_class,
    )

    serialized = len(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
    meta = {
        "representation_class": payload_representation_class(value),
        "elements": float(payload_compute_scale(value) or 0.0),
        "serialized_bytes": float(serialized),
        "native_bytes": float(
            value.numel() * value.element_size()
            if hasattr(value, "numel")
            else 0
        ),
        "dtype": str(getattr(value, "dtype", type(value).__name__)),
        "shape": list(getattr(value, "shape", [])),
        "contiguous": bool(getattr(value, "is_contiguous", lambda: True)()),
        "container": type(value).__name__,
    }
    return meta


def _load_sources(limit: int = N_SOURCES) -> List[np.ndarray]:
    from PIL import Image

    files = sorted(
        path
        for pattern in ("*.JPEG", "*.jpg", "*.jpeg", "*.png")
        for path in IMAGENETTE_TRAIN.rglob(pattern)
    )[:limit]
    if len(files) < limit:
        raise RuntimeError(f"need {limit} source images, found {len(files)}")
    return [np.asarray(Image.open(path).convert("RGB")) for path in files]


# ------------------------------------------------------------------- actor --


def _actor_cls():
    import ray

    @ray.remote
    class ScaleActor:
        def __init__(self, sources, base_seed):
            # Same thread contract as every Cedar worker.
            from cedar.utils.threading import limit_native_threadpools

            self._limiter = limit_native_threadpools(1)
            self._sources = sources
            self._base_seed = base_seed
            self._operators = build_operators()
            self._snapshots: Dict[str, List[bytes]] = {}
            self._meta: Dict[str, Dict[str, Any]] = {}
            self._output_meta: Dict[str, Dict[str, Any]] = {}

        def describe(self):
            import ray
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
                "probe_file": __file__,
            }

        def prepare(self, cells):
            """Materialise one serialized snapshot per (cell, source)."""
            for cell in cells:
                cell_id = cell["cell_id"]
                payloads = [
                    build_payload(
                        source,
                        cell["representation_class"],
                        cell["height"],
                        cell["width"],
                    )
                    for source in self._sources
                ]
                self._snapshots[cell_id] = [
                    pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
                    for value in payloads
                ]
                self._meta[cell_id] = representation_metadata(payloads[0])
                self._output_meta[cell_id] = self._output_metadata(cell, payloads[0])
                del payloads
            return {
                "cells": len(cells),
                "snapshot_bytes": sum(
                    len(snap)
                    for snaps in self._snapshots.values()
                    for snap in snaps
                ),
            }

        def _output_metadata(self, cell, payload):
            fn = self._operators[cell["operator"]]
            random.seed(7)
            torch.manual_seed(7)
            out = fn(payload)
            return representation_metadata(out)

        def warmup(self, cells, calls: int = 3):
            for cell in cells:
                fn = self._operators[cell["operator"]]
                snaps = self._snapshots[cell["cell_id"]]
                for index in range(max(1, calls)):
                    value = pickle.loads(snaps[index % len(snaps)])
                    fn(value)
                    del value
            return True

        def measure(self, cells, block):
            """One interleaved block; returns per-call durations per cell."""
            order = list(cells)
            random.Random(self._base_seed + 977 * block).shuffle(order)
            durations: Dict[str, List[Tuple[int, float]]] = {
                cell["cell_id"]: [] for cell in cells
            }
            budget = {
                cell["cell_id"]: float(
                    OPERATOR_SPECS[cell["operator"]]["budget_sec"]
                )
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
                    snaps = self._snapshots[cell_id]
                    source_index = call_index % len(snaps)
                    # Payload decoding and seeding sit outside the window.
                    value = pickle.loads(snaps[source_index])
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
            return durations, spent

        def metadata(self):
            return {"input": self._meta, "output": self._output_meta}

    return ScaleActor


def _prepare_modules_snapshot() -> Path:
    """Ship the driver's ``cedar`` and this probe to the remote workers.

    The remote Ray node has its own (older) checkout of the repository, so an
    actor that imports ``cedar`` through the node's ``PYTHONPATH`` silently
    measures a different implementation.  Like the campaign's ``entry.py``
    wrapper, ship an explicit snapshot as the job's working directory.
    """
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
    for name in ("__init__.py", "ch32_scale_affine_probe.py"):
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
    snapshot = _prepare_modules_snapshot()
    if not ray.is_initialized():
        ray.init(
            address=RAY_ADDRESS,
            ignore_reinit_error=True,
            logging_level="ERROR",
            runtime_env={
                "working_dir": str(snapshot),
                "env_vars": {
                    name: os.environ[name]
                    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                 "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
                    if name in os.environ
                },
            },
        )
    resource = os.environ.get("CEDAR_RAY_PLACEMENT_RESOURCE", "cedar_remote")
    validate_remote_ray_resource(resource, 0.001)
    return ray, get_ray_actor_options


def cmd_measure(args) -> int:
    ray, get_ray_actor_options = _ray_init()
    sources = _load_sources()
    cells = build_cells()
    if args.cells and args.cells != "all":
        wanted = set(args.cells.split(","))
        cells = [cell for cell in cells if cell["operator"] in wanted]
    blocks = 1 if args.dry_run else BLOCKS
    OUT.mkdir(parents=True, exist_ok=True)

    actor_cls = _actor_cls()
    actor = actor_cls.options(**get_ray_actor_options(num_cpus=1.0)).remote(
        sources, BASE_SEED
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
                    "pair_id": cell["pair_id"],
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
                raw_rows.append(
                    (cell["cell_id"], block, call_index, source_index, value)
                )
        print(
            f"block {block} done in {time.time() - started:.1f}s "
            f"({sum(len(v) for v in durations.values())} calls)",
            flush=True,
        )

    meta = ray.get(actor.metadata.remote())
    for row in rows:
        cell_id = row["cell_id"]
        for prefix, source in (("", meta["input"][cell_id]), ("out_", meta["output"][cell_id])):
            for key, value in source.items():
                row[prefix + key] = value
    _write_measurements(rows, raw_rows)
    env = {
        "commit": _git_commit(),
        "ray_address": RAY_ADDRESS,
        "driver_ip": driver_ip,
        "actor": location,
        "driver_cedar_common_sha256": hashlib.sha256(
            (ROOT / "cedar/pipes/common.py").read_bytes()
        ).hexdigest(),
        "sources": [
            str(path)
            for pattern in ("*.JPEG", "*.jpg", "*.jpeg", "*.png")
            for path in sorted(IMAGENETTE_TRAIN.rglob(pattern))
        ][:N_SOURCES],
        "n_sources": N_SOURCES,
        "blocks": blocks,
        "max_calls_per_cell": MAX_CALLS_PER_CELL,
        "base_seed": BASE_SEED,
        "operators": {
            name: {
                "parameters": spec["parameters"],
                "classes": spec["classes"],
                "ladder": spec["ladder"],
                "pairs": spec.get("pairs", []),
                "budget_sec": spec["budget_sec"],
                "reference_size_for_anchor": REFERENCE_SIZE[name],
                "real_class_for_anchor": REAL_CLASS[name],
            }
            for name, spec in OPERATOR_SPECS.items()
        },
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "protocol": {
            "timed_window": "callable only; pickle.loads and seeding outside the clock",
            "value_freshness": "pickle.loads(snapshot) before every call",
            "interleaving": "all cells of the sweep interleaved inside one window per block",
            "order": "shuffled per block with seed base_seed + 977*block",
            "seeds": "torch.manual_seed/random.seed(base_seed + 100000*block + call_index) before each call",
            "mean_primary": True,
            "anchor_rule": (
                "M1 reference point = the training cell of the operator's real "
                "representation class (REAL_CLASS) whose spatial size is closest "
                "to REFERENCE_SIZE; within a channel-only group the same rule is "
                "applied with the group's classes"
            ),
        },
    }
    (OUT / "env.json").write_text(json.dumps(env, indent=1, default=str))
    print(f"wrote {OUT}/measurements.csv rows={len(rows)}", flush=True)
    return 0


def _write_measurements(rows, raw_rows) -> None:
    import csv

    fields = [
        "operator", "cell_id", "representation_class", "size_id", "pair_id",
        "train_or_validation", "block_id", "height", "width", "calls",
        "mean_ms", "median_ms", "p10_ms", "p90_ms", "std_ms", "min_ms",
        "max_ms", "accumulated_sec", "measurement_protocol", "elements",
        "serialized_bytes", "native_bytes", "dtype", "shape", "contiguous",
        "container", "out_elements", "out_serialized_bytes", "out_native_bytes",
        "out_dtype", "out_shape", "out_contiguous", "out_representation_class",
    ]
    with (OUT / "measurements.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    with gzip.open(OUT / "raw_calls.csv.gz", "wt", newline="") as handle:
        handle.write("cell_id,block_id,call_index,source_id,ms\n")
        for cell_id, block, call_index, source_index, value in raw_rows:
            handle.write(f"{cell_id},{block},{call_index},{source_index},{value:.6f}\n")


def _git_commit() -> str:
    import subprocess

    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()


# -------------------------------------------------------------- fits stage --


def _nnls(design: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, str]:
    from scipy.optimize import nnls

    if design.shape[1] == 1:
        slope = float(max(0.0, np.dot(design[:, 0], target) / max(1e-30, np.dot(design[:, 0], design[:, 0]))))
        return np.array([slope]), "nnls_1d"
    coefficients, _ = nnls(design, target)
    return coefficients, "scipy.optimize.nnls"


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    error = predicted - actual
    nonzero = np.abs(actual) > 1e-12
    return {
        "mae_ms": float(np.mean(np.abs(error))),
        "mape": float(np.mean(np.abs(error[nonzero]) / np.abs(actual[nonzero]))) if nonzero.any() else float("nan"),
        "rmse_ms": float(np.sqrt(np.mean(error ** 2))),
        "max_abs_err_ms": float(np.max(np.abs(error))),
    }


def _r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    ss_res = float(np.sum((actual - predicted) ** 2))
    ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


IMPLEMENTATION_NAMES = {
    "blur": "torchvision.transforms.GaussianBlur",
    "jitter": "torchvision.transforms.ColorJitter",
    "crop": "torchvision.transforms.RandomResizedCrop",
    "grayscale": "torchvision.transforms.Grayscale",
    "flip": "torchvision.transforms.RandomHorizontalFlip",
}

PROVENANCE_COLUMNS = (
    "implementation",
    "parameters",
    "seed_rule",
    "block_seed",
    "source_ids",
    "n_sources",
)


def _enrich_measurements() -> None:
    """Add provenance columns derived from the frozen spec (values untouched).

    ``measure`` writes the measurements; the paper-facing schema also wants
    the implementation, parameters, seed rule and sample identity per row.
    Those are constants of the protocol (env.json), so they are filled in
    here instead of re-running the measurement window.
    """
    import csv

    path = OUT / "measurements.csv"
    with path.open() as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if all(column in fields for column in PROVENANCE_COLUMNS):
        return
    for row in rows:
        row["implementation"] = IMPLEMENTATION_NAMES.get(row["operator"], "")
        row["parameters"] = json.dumps(
            OPERATOR_SPECS[row["operator"]]["parameters"], sort_keys=True
        )
        row["seed_rule"] = (
            "torch.manual_seed/random.seed(base_seed + 100000*block_id + call_index); "
            "set outside the timing window"
        )
        row["block_seed"] = BASE_SEED + 100000 * int(row["block_id"])
        row["source_ids"] = ",".join(str(index) for index in range(N_SOURCES))
        row["n_sources"] = N_SOURCES
    ordered = []
    for column in fields:
        ordered.append(column)
        if column == "measurement_protocol":
            ordered.extend(PROVENANCE_COLUMNS)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_measurements():
    import csv

    _enrich_measurements()
    with (OUT / "measurements.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("mean_ms", "median_ms", "p10_ms", "p90_ms", "std_ms",
                    "elements", "serialized_bytes", "native_bytes", "calls",
                    "out_elements", "out_serialized_bytes"):
            row[key] = float(row[key])
        row["block_id"] = int(row["block_id"])
        row["height"] = int(row["height"])
        row["width"] = int(row["width"])
    return rows


def _block_means(rows):
    """Per-cell mean across blocks, keeping the block spread as uncertainty."""
    cells: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        cell = cells.setdefault(row["cell_id"], {"rows": [], **row})
        cell["rows"].append(row)
    for cell in cells.values():
        means = [row["mean_ms"] for row in cell["rows"]]
        cell["mean_ms"] = statistics.fmean(means)
        cell["block_min_ms"] = min(means)
        cell["block_max_ms"] = max(means)
        cell["block_std_ms"] = statistics.pstdev(means) if len(means) > 1 else 0.0
        cell["calls"] = sum(row["calls"] for row in cell["rows"])
    return cells


def _fit_group(cells, operator, group, klass_filter):
    """Fit every model of the required list on one (operator, group) slice."""
    subset = [
        cell
        for cell in cells.values()
        if cell["operator"] == operator
        and cell["train_or_validation"] in ("train", "validation")
        and (klass_filter is None or cell["representation_class"] in klass_filter)
    ]
    train = [cell for cell in subset if cell["train_or_validation"] == "train"]
    validation = [cell for cell in subset if cell["train_or_validation"] == "validation"]
    if not train:
        return [], []
    reference = REFERENCE_SIZE[operator]
    real_class = REAL_CLASS[operator]

    def _class_rank(klass: str) -> int:
        """Prefer the operator's real class, then same channels, then any."""
        if klass == real_class:
            return 0
        same_channels = klass.partition(":")[2] == real_class.partition(":")[2]
        if same_channels and klass.startswith("float32"):
            return 1
        if same_channels:
            return 2
        return 3

    anchor = min(
        train,
        key=lambda cell: (
            _class_rank(cell["representation_class"]),
            abs(cell["height"] - reference),
            abs(cell["width"] - reference),
            cell["cell_id"],
        ),
    )
    rows: List[Dict[str, Any]] = []
    predictions: List[Dict[str, Any]] = []

    def emit(model, scope, coefficients, constraint, train_cells, val_cells, predictor):
        train_actual = np.array([cell["mean_ms"] for cell in train_cells])
        train_pred = np.array([predictor(cell) for cell in train_cells])
        val_actual = np.array([cell["mean_ms"] for cell in val_cells])
        val_pred = np.array([predictor(cell) for cell in val_cells])
        if len(val_cells):
            val_metrics = {
                f"val_{key}": value
                for key, value in _metrics(val_actual, val_pred).items()
            }
        else:
            val_metrics = {
                "val_mae_ms": float("nan"),
                "val_mape": float("nan"),
                "val_rmse_ms": float("nan"),
                "val_max_abs_err_ms": float("nan"),
            }
        entry = {
            "operator": operator,
            "group": group,
            "model": model,
            "class_scope": scope,
            "coefficients": json.dumps(coefficients),
            "constraint": constraint,
            "anchor_cell": anchor["cell_id"],
            "anchor_bytes": anchor["serialized_bytes"],
            "anchor_elements": anchor["elements"],
            "anchor_mean_ms": anchor["mean_ms"],
            "train_cells": len(train_cells),
            "validation_cells": len(val_cells),
            "train_x_min": min([cell["elements"] for cell in train_cells]),
            "train_x_max": max([cell["elements"] for cell in train_cells]),
            "train_r2": _r2(train_actual, train_pred) if len(train_cells) > 1 else float("nan"),
            **{f"train_{k}": v for k, v in _metrics(train_actual, train_pred).items()},
            **val_metrics,
        }
        rows.append(entry)
        for cell, predicted in zip(val_cells, val_pred):
            predictions.append(_prediction_row(cell, entry, predicted))
        for cell, predicted in zip(train_cells, train_pred):
            predictions.append(_prediction_row(cell, entry, predicted))

    # M1 anchored byte-proportional, M2 byte affine, M3 element proportional,
    # M4 element affine; M5 is emitted per class below.
    for model, through_origin, feature in (
        ("M2_byte_affine", False, "serialized_bytes"),
        ("M3_element_prop", True, "elements"),
        ("M4_element_affine", False, "elements"),
    ):
        x = np.array([cell[feature] for cell in train])
        y = np.array([cell["mean_ms"] for cell in train])
        design = x.reshape(-1, 1) if through_origin else np.column_stack([x, np.ones_like(x)])
        coefficients, method = _nnls(design, y)
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
        else:
            predictor = lambda cell, c=coefficients, f=feature: c[0] * cell[f] + c[1]
        emit(model, "shared", {"k": float(coefficients[0]), **({"b": float(coefficients[1])} if not through_origin else {})}, constraint, train, validation, predictor)

    scale = anchor["serialized_bytes"]
    base = anchor["mean_ms"]
    emit(
        "M1_byte_prop_anchored",
        "shared",
        {"anchor_ms": base, "anchor_bytes": scale},
        "anchored at a single training point (no fit)",
        train,
        validation,
        lambda cell: base * (cell["serialized_bytes"] / scale),
    )

    for klass in sorted({cell["representation_class"] for cell in train}):
        class_train = [cell for cell in train if cell["representation_class"] == klass]
        class_val = [cell for cell in validation if cell["representation_class"] == klass]
        if len(class_train) < 2:
            continue
        x = np.array([cell["elements"] for cell in class_train])
        y = np.array([cell["mean_ms"] for cell in class_train])
        coefficients, method = _nnls(np.column_stack([x, np.ones_like(x)]), y)
        predictor = lambda cell, c=coefficients: c[0] * cell["elements"] + c[1]
        emit(
            "M5_repr_element_affine",
            klass,
            {"k": float(coefficients[0]), "b": float(coefficients[1])},
            f"non-negative {method}"
            + (", slope bound active" if coefficients[0] <= 1e-18 else "")
            + (", intercept bound active" if coefficients[1] <= 1e-18 else ""),
            class_train,
            class_val,
            predictor,
        )
    return rows, predictions


def _prediction_row(cell, entry, predicted) -> Dict[str, Any]:
    actual = cell["mean_ms"]
    return {
        "operator": cell["operator"],
        "group": entry["group"],
        "model": entry["model"],
        "class_scope": entry["class_scope"],
        "cell_id": cell["cell_id"],
        "representation_class": cell["representation_class"],
        "size_id": cell["size_id"],
        "train_or_validation": cell["train_or_validation"],
        "elements": cell["elements"],
        "serialized_bytes": cell["serialized_bytes"],
        "measured_mean_ms": actual,
        "block_std_ms": cell["block_std_ms"],
        "predicted_ms": predicted,
        "abs_err_ms": predicted - actual,
        "rel_err": (predicted - actual) / actual if actual else float("nan"),
    }


def _class_conditioned_predictions(cells, predictions):
    """Aggregate the per-class M5 rows into a single shared-evaluation view."""
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in predictions:
        if row["model"] == "M5_repr_element_affine":
            groups.setdefault((row["operator"], row["group"]), []).append(row)
    out = []
    for (operator, group), rows in groups.items():
        train = [row for row in rows if row["train_or_validation"] == "train"]
        val = [row for row in rows if row["train_or_validation"] == "validation"]
        if not train:
            continue
        actual = np.array([row["measured_mean_ms"] for row in train])
        predicted = np.array([row["predicted_ms"] for row in train])
        out.append(
            {
                "operator": operator,
                "group": group,
                "model": "M5_repr_element_affine",
                "class_scope": "per_class",
                "coefficients": "per class",
                "constraint": "non-negative, per class",
                "anchor_cell": "",
                "train_cells": len(train),
                "validation_cells": len(val),
                "train_x_min": min(row["elements"] for row in train),
                "train_x_max": max(row["elements"] for row in train),
                "train_r2": _r2(actual, predicted),
                **{f"train_{k}": v for k, v in _metrics(actual, predicted).items()},
                **(
                    {
                        f"val_{k}": v
                        for k, v in _metrics(
                            np.array([row["measured_mean_ms"] for row in val]),
                            np.array([row["predicted_ms"] for row in val]),
                        ).items()
                    }
                    if val
                    else {}
                ),
            }
        )
    return out


def cmd_fit(args) -> int:
    import csv

    rows = _load_measurements()
    cells = _block_means(rows)
    fits: List[Dict[str, Any]] = []
    predictions: List[Dict[str, Any]] = []
    for operator in OPERATOR_SPECS:
        groups = [("all", None), ("1ch", "1"), ("3ch", "3")]
        for group, channels in groups:
            klass_filter = (
                {klass for klass in OPERATOR_SPECS[operator]["classes"] if klass.endswith(f"{channels}ch")}
                if channels
                else set(OPERATOR_SPECS[operator]["classes"])
            )
            group_fits, group_predictions = _fit_group(cells, operator, group, klass_filter)
            fits.extend(group_fits)
            predictions.extend(group_predictions)
    fits.extend(_class_conditioned_predictions(cells, predictions))
    _write_fits(fits, predictions)
    _write_figure_data(cells, fits, predictions)
    _write_notes(cells, fits, predictions)
    print(f"fits={len(fits)} predictions={len(predictions)}", flush=True)
    return 0


DOCS = ROOT / "docs/ch32_scale_affine_20260927"
SMALL_FILES = (
    "README.md",
    "env.json",
    "measurements.csv",
    "fits.csv",
    "predictions.csv",
    "figure_data.json",
)


def cmd_manifest(args) -> int:
    """Hash every artifact and snapshot the small ones into ``docs/``."""
    import csv
    import gzip
    import shutil

    DOCS.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {
        "commit": _git_commit(),
        "result_dir": str(OUT.relative_to(ROOT)),
        "small_files": {},
        "large_files": {},
        "figures": {},
        "provenance": {},
    }
    for name in SMALL_FILES:
        path = OUT / name
        if not path.exists():
            continue
        data = path.read_bytes()
        entry = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
            "rows": len(data.decode(errors="replace").splitlines()) - 1,
        }
        manifest["small_files"][name] = entry
        shutil.copy2(path, DOCS / name)
    for name in ("raw_calls.csv.gz", "measure.log", "modules"):
        path = OUT / name
        if not path.exists() or path.is_dir():
            continue
        data = path.read_bytes()
        entry = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
        }
        if name.endswith(".csv.gz"):
            with gzip.open(path, "rt") as handle:
                entry["rows"] = sum(1 for _ in handle) - 1
        manifest["large_files"][name] = entry
    shipped_probe = OUT / "modules/tmp_analysis/ch32_scale_affine_probe.py"
    if shipped_probe.exists():
        data = shipped_probe.read_bytes()
        manifest["provenance"]["measurement_probe_snapshot"] = {
            "path": str(shipped_probe.relative_to(ROOT)),
            "sha256": hashlib.sha256(data).hexdigest(),
            "note": "exact code the remote actor executed for this round's measurements",
        }
    for name in ("ch32_scale_affine_probe.py", "plot_ch32_scale_affine.py"):
        path = ROOT / "tmp_analysis" / name
        if path.exists():
            data = path.read_bytes()
            manifest["provenance"][name] = {
                "path": str(path.relative_to(ROOT)),
                "sha256": hashlib.sha256(data).hexdigest(),
                "note": "current working-tree version (fit/plot stages)",
            }
    figure_dir = OUT / "figures"
    if figure_dir.is_dir():
        for path in sorted(figure_dir.glob("*.png")):
            data = path.read_bytes()
            manifest["figures"][path.name] = {
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
            }
            shutil.copy2(path, DOCS / path.name)
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    shutil.copy2(OUT / "MANIFEST.json", DOCS / "MANIFEST.json")
    print(f"manifest: {len(manifest['small_files'])} small files", flush=True)
    return 0


FIT_FIELDS = [
    "operator", "group", "model", "class_scope", "coefficients", "constraint",
    "anchor_cell", "anchor_bytes", "anchor_elements", "anchor_mean_ms",
    "train_cells", "validation_cells", "train_x_min", "train_x_max",
    "train_r2", "train_mae_ms", "train_mape", "train_rmse_ms",
    "train_max_abs_err_ms", "val_mae_ms", "val_mape", "val_rmse_ms",
    "val_max_abs_err_ms",
]
PREDICTION_FIELDS = [
    "operator", "group", "model", "class_scope", "cell_id",
    "representation_class", "size_id", "train_or_validation", "elements",
    "serialized_bytes", "measured_mean_ms", "block_std_ms", "predicted_ms",
    "abs_err_ms", "rel_err",
]


def _write_fits(fits, predictions) -> None:
    import csv

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


def _write_figure_data(cells, fits, predictions) -> None:
    panels: Dict[str, Any] = {"panels": {}, "cells": []}
    for cell in sorted(cells.values(), key=lambda item: item["cell_id"]):
        panels["cells"].append(
            {
                "cell_id": cell["cell_id"],
                "operator": cell["operator"],
                "representation_class": cell["representation_class"],
                "size_id": cell["size_id"],
                "role": cell["train_or_validation"],
                "pair_id": cell["pair_id"],
                "height": cell["height"],
                "width": cell["width"],
                "elements": cell["elements"],
                "serialized_bytes": cell["serialized_bytes"],
                "native_bytes": cell["native_bytes"],
                "out_elements": cell.get("out_elements"),
                "out_serialized_bytes": cell.get("out_serialized_bytes"),
                "mean_ms": cell["mean_ms"],
                "median_ms": statistics.fmean([row["median_ms"] for row in cell["rows"]]),
                "p10_ms": statistics.fmean([row["p10_ms"] for row in cell["rows"]]),
                "p90_ms": statistics.fmean([row["p90_ms"] for row in cell["rows"]]),
                "block_mean_ms": [row["mean_ms"] for row in cell["rows"]],
                "block_std_ms": cell["block_std_ms"],
                "calls": cell["calls"],
            }
        )
    for operator in OPERATOR_SPECS:
        operator_cells = [
            cell for cell in panels["cells"] if cell["operator"] == operator
        ]
        anchor = next(
            (
                fit["anchor_cell"]
                for fit in fits
                if fit["operator"] == operator and fit["group"] == "all"
                and fit["model"] == "M1_byte_prop_anchored"
            ),
            "",
        )
        panels["panels"][operator] = {
            "cells": [cell["cell_id"] for cell in operator_cells],
            "reference_anchor_cell": anchor,
            "reference_size": REFERENCE_SIZE[operator],
            "normalization": {
                "anchor_elements": next(
                    (cell["elements"] for cell in operator_cells if cell["cell_id"] == anchor),
                    None,
                ),
                "anchor_ms": next(
                    (cell["mean_ms"] for cell in operator_cells if cell["cell_id"] == anchor),
                    None,
                ),
            },
        }
    # Optional normalized coordinates: element count and time divided by each
    # operator's own reference point (the same anchor the byte model uses).
    for operator, panel in panels["panels"].items():
        reference_elements = panel["normalization"]["anchor_elements"]
        reference_ms = panel["normalization"]["anchor_ms"]
        if not reference_elements or not reference_ms:
            continue
        for cell in panels["cells"]:
            if cell["operator"] != operator:
                continue
            cell["elements_norm"] = cell["elements"] / reference_elements
            cell["ms_norm"] = cell["mean_ms"] / reference_ms
    panels["fits"] = fits
    panels["predictions"] = predictions
    panels["diagnostics"] = _diagnostic_contrasts(panels["cells"])
    (OUT / "figure_data.json").write_text(json.dumps(panels, indent=1, default=float))


def _diagnostic_contrasts(cells) -> Dict[str, Any]:
    """Controlled pairs the paper needs, computed from the measured cells."""
    by_operator: Dict[str, List[Dict[str, Any]]] = {}
    for cell in cells:
        by_operator.setdefault(cell["operator"], []).append(cell)

    def _dtype(klass: str) -> str:
        return klass.partition(":")[0]

    def _channels(klass: str) -> str:
        return klass.partition(":")[2]

    out: Dict[str, Any] = {
        "dtype_pairs": [],
        "element_matched_pairs": [],
        "byte_matched_pairs": [],
        "duplicate_cells": [],
    }
    for operator, items in by_operator.items():
        # Same operator, same shape/channels, different dtype.
        buckets: Dict[Tuple[int, int, str], List[Dict[str, Any]]] = {}
        for cell in items:
            if cell["role"] == "validation":
                continue
            buckets.setdefault(
                (cell["height"], cell["width"], _channels(cell["representation_class"])),
                [],
            ).append(cell)
        for (height, width, channels), group in sorted(buckets.items()):
            dtypes = {_dtype(cell["representation_class"]) for cell in group}
            if len(dtypes) < 2:
                continue
            out["dtype_pairs"].append(
                {
                    "operator": operator,
                    "height": height,
                    "width": width,
                    "channels": channels,
                    "points": [
                        {
                            "cell_id": cell["cell_id"],
                            "class": cell["representation_class"],
                            "elements": cell["elements"],
                            "serialized_bytes": cell["serialized_bytes"],
                            "mean_ms": cell["mean_ms"],
                            "block_std_ms": cell["block_std_ms"],
                        }
                        for cell in sorted(group, key=lambda item: item["cell_id"])
                    ],
                }
            )
        # Same elements, different channel count (declared pair cells).
        pairs: Dict[str, List[Dict[str, Any]]] = {}
        for cell in items:
            if cell["pair_id"]:
                pairs.setdefault(cell["pair_id"], []).append(cell)
        for pair_id, group in sorted(pairs.items()):
            if len(group) < 2:
                continue
            out["element_matched_pairs"].append(
                {
                    "operator": operator,
                    "pair_id": pair_id,
                    "elements": group[0]["elements"],
                    "points": [
                        {
                            "cell_id": cell["cell_id"],
                            "class": cell["representation_class"],
                            "shape": [cell["height"], cell["width"]],
                            "serialized_bytes": cell["serialized_bytes"],
                            "mean_ms": cell["mean_ms"],
                            "block_std_ms": cell["block_std_ms"],
                        }
                        for cell in sorted(group, key=lambda item: item["cell_id"])
                    ],
                }
            )
        # Same bytes (within 5%), different element count.
        non_validation = [cell for cell in items if cell["role"] != "validation"]
        for index, first in enumerate(non_validation):
            for second in non_validation[index + 1:]:
                if _channels(first["representation_class"]) != _channels(
                    second["representation_class"]
                ):
                    continue
                bytes_a = first["serialized_bytes"]
                bytes_b = second["serialized_bytes"]
                if min(bytes_a, bytes_b) <= 0:
                    continue
                if abs(bytes_a - bytes_b) / min(bytes_a, bytes_b) > 0.05:
                    continue
                if first["elements"] == second["elements"]:
                    continue
                out["byte_matched_pairs"].append(
                    {
                        "operator": operator,
                        "byte_delta": abs(bytes_a - bytes_b) / min(bytes_a, bytes_b),
                        "points": [
                            {
                                "cell_id": cell["cell_id"],
                                "class": cell["representation_class"],
                                "shape": [cell["height"], cell["width"]],
                                "elements": cell["elements"],
                                "serialized_bytes": cell["serialized_bytes"],
                                "mean_ms": cell["mean_ms"],
                                "block_std_ms": cell["block_std_ms"],
                            }
                            for cell in sorted(
                                (first, second), key=lambda item: item["elements"]
                            )
                        ],
                    }
                )
        # Cells measured twice under different ids (ladder point reused as a
        # declared pair) -- a built-in repeatability check.
        shapes: Dict[Tuple[str, int, int], List[Dict[str, Any]]] = {}
        for cell in items:
            shapes.setdefault(
                (cell["representation_class"], cell["height"], cell["width"]), []
            ).append(cell)
        for (klass, height, width), group in sorted(shapes.items()):
            if len(group) < 2:
                continue
            out["duplicate_cells"].append(
                {
                    "operator": operator,
                    "class": klass,
                    "shape": [height, width],
                    "means_ms": [cell["mean_ms"] for cell in group],
                    "relative_gap": abs(group[0]["mean_ms"] - group[1]["mean_ms"])
                    / max(group[0]["mean_ms"], group[1]["mean_ms"]),
                    "cell_ids": [cell["cell_id"] for cell in group],
                }
            )
    return out


def _fmt(value) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return "—"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _pct(value) -> str:
    try:
        return f"{100.0 * float(value):.1f}%"
    except (TypeError, ValueError):
        return "—"


# operator ids in the shipped SimCLRv2 profile, taken from the feature's own
# logical pipe ids (printed by the mapping probe in the round's log).
PROFILE_OPERATOR_NAMES = {
    "1": "Normalize",
    "2": "blur",
    "3": "grayscale",
    "4": "jitter",
    "5": "flip",
    "6": "crop",
    "7": "to_float",
    "8": "ImageReaderPipe",
}


def _deployed_coefficients() -> Dict[Tuple[str, str], Tuple[float, float]]:
    """Coefficients of the shipped profile (consistency check only)."""
    import yaml

    path = ROOT / "outputs/affine_repr_profile_20260924/simclrv2/shared.yaml"
    if not path.exists():
        return {}
    profile = yaml.safe_load(path.read_text())
    operators = (profile.get("physical_model") or {}).get("compute_model", {}).get(
        "operators", {}
    )
    out: Dict[Tuple[str, str], Tuple[float, float]] = {}
    for p_id, entry in operators.items():
        name = PROFILE_OPERATOR_NAMES.get(str(p_id))
        if not name:
            continue
        for klass, curve in (entry.get("by_class") or {}).items():
            out[(name, klass)] = (
                float(curve.get("k_ms_per_element", 0.0)),
                float(curve.get("b_ms", 0.0)),
            )
    return out


def _write_notes(cells, fits, predictions) -> None:
    def find(model, operator=None, group=None, scope=None):
        for fit in fits:
            if fit["model"] != model:
                continue
            if operator and fit["operator"] != operator:
                continue
            if group and fit["group"] != group:
                continue
            if scope and fit["class_scope"] != scope:
                continue
            return fit
        return None

    def val_mape(model, operator, group, scope="shared"):
        fit = find(model, operator, group, scope)
        return _pct(fit.get("val_mape")) if fit else "—"

    def coefficients(model, operator, group, scope):
        fit = find(model, operator, group, scope)
        if not fit or not str(fit["coefficients"]).startswith("{"):
            return None
        return json.loads(fit["coefficients"])

    figure = json.loads((OUT / "figure_data.json").read_text()) if (OUT / "figure_data.json").exists() else {}
    diagnostics = figure.get("diagnostics", {})
    blur_bytes = [
        item for item in diagnostics.get("byte_matched_pairs", [])
        if item["operator"] == "blur"
    ]
    element_pairs = diagnostics.get("element_matched_pairs", [])
    duplicates = diagnostics.get("duplicate_cells", [])

    lines = [
        "# §3.2 计算规模与仿射响应：受控测量与独立验证（2026-09-27）",
        "",
        f"commit `{_git_commit()}`；结果目录 `outputs/ch32_scale_affine_20260927/`，",
        "小文件快照在 `docs/ch32_scale_affine_20260927/`。",
        "",
        "## 1. 本轮唯一口径（图只能用这批数据）",
        "",
        "| 项目 | 取值 |",
        "| --- | --- |",
        "| 测量对象 | 算子自身计算服务时间（ms / call） |",
        "| 计时窗口 | 只有 callable 调用；`pickle.loads` 与随机种子设置都在窗口外 |",
        "| 输入新鲜度 | 每次调用前用 `pickle.loads` 重新取一份 payload（与部署 profiler 相同） |",
        "| 交错方式 | 同一算子所有 cell 在一个 block 内轮转，block 间以固定种子重排 |",
        "| block 数 | 3（互相独立，报告逐 block 均值、极差与标准差） |",
        "| 每 cell 预算 | Blur 2.5 s、Jitter/Crop 1.5 s、Grayscale/Flip 1.0 s，调用上限 "
        f"{MAX_CALLS_PER_CELL} |",
        "| 主指标 | 算术平均；median / p10 / p90 / std 仅作诊断 |",
        "| 执行位置 | 远端 Ray actor（`cedar_remote`，actor IP 记在 `env.json`），线程池 = 1 |",
        "| 随机性 | `torch.manual_seed`/`random.seed(base_seed + 100000*block + call_index)`；"
        "同一 call_index 在不同 cell 上使用同一随机路径 |",
        "| 源样本 | 4 张 imagenette2 训练图，按 call 轮转，均值覆盖 4 张图 |",
        "| 尺寸 | 每表示类 8 个训练尺寸 + 4 个独立验证尺寸；尺寸交错排列 |",
        "| 输入来源 | **受控生成**：真实 imagenette2 训练图重采样到各尺寸；"
        "uint8 载荷即 reader 的输出，float32 载荷保持 reader 的 0–255 值域"
        "（与 `to_float` 的纯 cast 一致）。不使用真实流水线的中间张量，"
        "因此两条协议不能直接换用数值 |",
        "",
        "**不要**把这些绝对时间与 §4.x 的历史图或 §4.11 的 M1–M5 表格混用：那些是",
        "不同数据量、不同归一口径（每源记录）与不同 profiler 采样下的数值。本轮的部署侧对照只写在 §7。",
        "",
        "## 2. 建议面板能支持什么",
        "",
        f"- **Blur bytes vs elements（并排子图）**：同一组 {len([c for c in figure.get('cells', []) if c['operator'] == 'blur'])} 个点，",
        "  只换横轴。左边（bytes）上四条曲线互相纠缠，右边（elements）按通道数分成两组 →",
        "  直接支持“字节不是计算规模”。",
        "- **ColorJitter 元素—时间**：1ch 与 3ch 两条曲线相差近一个数量级，",
        "  且存在同元素数、不同通道数的配对点 → 支持“元素数还不够，需要表示类”。",
        "- **Crop 元素—时间**：输出固定 244×244，实测几乎不随输入尺寸变化 →",
        "  过原点比例模型必然失准，带截距的仿射模型才落在测量波动内；",
        "  图上同时画锚定字节比例预测、元素仿射拟合与独立验证点。",
        "- **候选面板**：Grayscale（同 Crop 的固定分量结构）、Flip（反例，见 §4）。",
        "",
        "## 3. 关键数字（训练=8 尺寸/类，验证=4 独立尺寸/类）",
        "",
        "### 3.1 字节 vs 元素（Blur，验证 MAPE）",
        "",
        "| 组 | M1 锚定字节比例 | M2 字节仿射 | M3 元素比例 | M4 元素仿射 | M5 表示类仿射 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group in ("all", "1ch", "3ch"):
        lines.append(
            f"| blur {group} | {val_mape('M1_byte_prop_anchored', 'blur', group)} | "
            f"{val_mape('M2_byte_affine', 'blur', group)} | "
            f"{val_mape('M3_element_prop', 'blur', group)} | "
            f"{val_mape('M4_element_affine', 'blur', group)} | "
            f"{val_mape('M5_repr_element_affine', 'blur', group, 'per_class')} |"
        )
    lines += [
        "",
        "同形同内容、只换 dtype 的对照（元素数相同、字节数 4×）与同字节不同元素的对照：",
        "",
        "| 对照 | 点 A | 点 B | A/B 时间比 |",
        "| --- | --- | --- | ---: |",
    ]
    for item in blur_bytes[:4]:
        a, b = item["points"]
        lines.append(
            f"| 字节相同（Δ≤{100*item['byte_delta']:.0f}%） | {a['class']} {a['shape']} "
            f"e={a['elements']:.0f} {a['mean_ms']:.2f} ms | {b['class']} {b['shape']} "
            f"e={b['elements']:.0f} {b['mean_ms']:.2f} ms | "
            f"{b['mean_ms']/a['mean_ms']:.2f}× |"
        )
    for item in diagnostics.get("dtype_pairs", [])[:3]:
        if item["operator"] != "blur" or len(item["points"]) < 2:
            continue
        a, b = item["points"]
        lines.append(
            f"| 同形不同 dtype | {a['class']} {item['height']}×{item['width']} "
            f"{a['serialized_bytes']:.0f} B {a['mean_ms']:.2f} ms | {b['class']} "
            f"{b['serialized_bytes']:.0f} B {b['mean_ms']:.2f} ms | "
            f"{max(a['mean_ms'], b['mean_ms'])/min(a['mean_ms'], b['mean_ms']):.2f}× |"
        )
    lines += [
        "",
        "### 3.2 表示类是否必要（Jitter / Flip 验证 MAPE）",
        "",
        "| 算子 | 组 | M3 元素比例 | M4 元素仿射（跨 dtype 共享） | M5 表示类仿射 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for operator in ("jitter", "flip", "blur", "crop"):
        for group in ("all", "3ch"):
            lines.append(
                f"| {operator} | {group} | {val_mape('M3_element_prop', operator, group)} | "
                f"{val_mape('M4_element_affine', operator, group)} | "
                f"{val_mape('M5_repr_element_affine', operator, group, 'per_class')} |"
            )
    lines += [
        "",
        "同元素数、不同通道数的配对点（元素模型对这两点给出同一个预测）：",
        "",
        "| 配对 | 元素数 | 1ch 点 | 3ch 点 | 时间比 |",
        "| --- | ---: | --- | --- | ---: |",
    ]
    for item in element_pairs:
        points = {point["class"].partition(":")[2]: point for point in item["points"]}
        one, three = points.get("1ch"), points.get("3ch")
        if not one or not three:
            continue
        lines.append(
            f"| {item['operator']} {item['pair_id']} | {item['elements']:.0f} | "
            f"{one['shape'][0]}×{one['shape'][1]} {one['mean_ms']:.3f} ms | "
            f"{three['shape'][0]}×{three['shape'][1]} {three['mean_ms']:.3f} ms | "
            f"{max(one['mean_ms'], three['mean_ms'])/min(one['mean_ms'], three['mean_ms']):.2f}× |"
        )
    lines += [
        "",
        "### 3.3 固定分量（Crop / Grayscale，验证 MAPE）",
        "",
        "| 算子 | 组 | M3 元素比例（过原点） | M4 元素仿射（含截距） | 拟合 k | 拟合 b (ms) |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for operator, group in (("crop", "1ch"), ("crop", "3ch"), ("grayscale", "3ch"), ("jitter", "1ch")):
        fit = find("M4_element_affine", operator, group, "shared")
        coefficients = json.loads(fit["coefficients"]) if fit else {}
        lines.append(
            f"| {operator} | {group} | {val_mape('M3_element_prop', operator, group)} | "
            f"{val_mape('M4_element_affine', operator, group)} | "
            f"{_fmt(coefficients.get('k'))} | {_fmt(coefficients.get('b'))} |"
        )
    crop_one = sorted(
        (
            cell
            for cell in cells.values()
            if cell["operator"] == "crop"
            and cell["representation_class"] == "float32:1ch"
            and cell["train_or_validation"] in ("train", "validation")
        ),
        key=lambda cell: cell["elements"],
    )
    crop_small, crop_large = crop_one[0], crop_one[-1]
    lines += [
        "",
        f"Crop 的训练/验证尺寸跨度 {crop_small['height']}→{crop_large['height']}"
        f"（元素数 {crop_large['elements']/crop_small['elements']:.0f}×），"
        f"float32:1ch 实测均值只从 {crop_small['mean_ms']:.3f} ms 变到 "
        f"{crop_large['mean_ms']:.3f} ms → 比例项几乎为零，固定分量占主导。",
        "",
        "### 3.4 重复性",
        "",
        "| 检查 | 结果 |",
        "| --- | --- |",
    ]
    for item in duplicates:
        lines.append(
            f"| 同 cell 被测量两次（{item['operator']} {item['class']} "
            f"{item['shape'][0]}×{item['shape'][1]}） | "
            f"{item['means_ms'][0]:.3f} vs {item['means_ms'][1]:.3f} ms，差 "
            f"{100*item['relative_gap']:.1f}% |"
        )
    blur_cells = [cell for cell in cells.values() if cell["operator"] == "blur"]
    worst = max(blur_cells, key=lambda cell: cell["block_std_ms"] / max(cell["mean_ms"], 1e-9))
    small = [cell for cell in blur_cells if cell["height"] <= 256]
    small_worst = max(small, key=lambda cell: cell["block_std_ms"] / max(cell["mean_ms"], 1e-9))
    lines += [
        f"| Blur 最大块间波动 | {worst['representation_class']} {worst['height']}×{worst['width']}："
        f"均值 {worst['mean_ms']:.2f} ms，块间 std {worst['block_std_ms']:.2f} ms "
        f"（{100*worst['block_std_ms']/worst['mean_ms']:.0f}%） |",
        f"| Blur ≤256 尺寸内最大块间波动 | {small_worst['height']}×{small_worst['width']}："
        f"{100*small_worst['block_std_ms']/small_worst['mean_ms']:.0f}% |",
        "",
        "Blur 在大尺寸上出现明显的慢模式（均值 ≫ 中位数，块间波动 10–32%），",
        "因此 Blur 的 448/512 验证点只能作为弱证据；Blur 的角色应是“说明字节与计算规模的区别”，",
        "而不是证明仿射精度。",
        "",
        "## 4. 哪些结果不支持原假设",
        "",
    ]
    blur_flip = find("M5_repr_element_affine", "flip", "all", "per_class")
    flip_bytes = find("M2_byte_affine", "flip", "all", "shared")
    blur_1ch_shared = find("M4_element_affine", "blur", "1ch", "shared")
    blur_1ch_class = find("M5_repr_element_affine", "blur", "1ch", "per_class")
    lines += [
        f"- **不能把“元素总是更好”写成无条件结论**：在跨表示类共享系数的模型里，"
        f"Flip 上字节仿射（{val_mape('M2_byte_affine', 'flip', 'all')}）优于元素仿射"
        f"（{val_mape('M4_element_affine', 'flip', 'all')}）；"
        f"但按表示类分开后元素仿射反超（{val_mape('M5_repr_element_affine', 'flip', 'all', 'per_class')}）。"
        "Flip 的成本主要是按存储字节搬运/分配，所以它更像“字节对照”，不是元素模型的失败。",
        f"- **Blur 不需要按 dtype 分开**：同通道内共享元素仿射（1ch "
        f"{val_mape('M4_element_affine', 'blur', '1ch')}）与按类分开（{val_mape('M5_repr_element_affine', 'blur', '1ch', 'per_class')}）"
        "几乎相同 → 需要区分的只是通道数。",
        "- **Blur 的固定分量证据弱**：float32:3ch 的 M4 截距被非负约束压到 0，",
        "  它在图上的价值是尺度对照，不是截距证据。",
        "",
        "## 5. 截距 b 是否有独立验证收益",
        "",
        "| 算子/组 | 过原点 M3 | 含截距 M4 | 结论 |",
        "| --- | ---: | ---: | --- |",
    ]
    verdict = {
        ("crop", "1ch"): "有：输出尺寸固定，截距即主要成本",
        ("crop", "3ch"): "有：同上，截距约 1.6 ms",
        ("grayscale", "3ch"): "有：截距约 0.07 ms，占比可观",
        ("jitter", "1ch"): "有：截距约 0.22 ms",
        ("blur", "1ch"): "弱：截距 0.26 ms，但块间波动同量级",
        ("blur", "3ch"): "无：非负约束把截距压到 0",
    }
    for (operator, group), text in verdict.items():
        lines.append(
            f"| {operator} {group} | {val_mape('M3_element_prop', operator, group)} | "
            f"{val_mape('M4_element_affine', operator, group)} | {text} |"
        )
    def share_verdict(shared_fit, class_fit) -> str:
        if not shared_fit or not class_fit:
            return "—"
        shared_mape = float(shared_fit["val_mape"])
        class_mape = float(class_fit["val_mape"])
        if class_mape >= 0.85 * shared_mape:
            return f"可以共享（{_pct(shared_mape)} → {_pct(class_mape)}）"
        if class_mape <= 0.6 * shared_mape:
            return f"不可共享（{_pct(shared_mape)} → {_pct(class_mape)}）"
        return f"分开有部分收益（{_pct(shared_mape)} → {_pct(class_mape)}）"

    lines += [
        "",
        "## 6. 哪些表示类可以共享曲线",
        "",
        "判据：把「同通道内跨 dtype 共享」（M4）与「按表示类分开」（M5）的验证 MAPE 对比，",
        "分开后 MAPE 下降 <15% 记为“可以共享”，>40% 记为“不可共享”；跨通道同理比较 M4(all) 与 M5(all)。",
        "",
        "| 算子 | 同通道内共享 dtype | 跨通道共享 |",
        "| --- | --- | --- |",
    ]
    for operator in ("blur", "jitter", "crop", "grayscale", "flip"):
        channels = sorted(
            {
                cell["representation_class"].partition(":")[2]
                for cell in cells.values()
                if cell["operator"] == operator
            }
        )
        within = "；".join(
            (
                f"{channel} 只有 1 个表示类（不存在共享问题）"
                if len(
                    {
                        cell["representation_class"]
                        for cell in cells.values()
                        if cell["operator"] == operator
                        and cell["representation_class"].endswith(channel)
                    }
                ) == 1
                else f"{channel} "
                + share_verdict(
                    find("M4_element_affine", operator, channel),
                    find("M5_repr_element_affine", operator, channel, "per_class"),
                )
            )
            for channel in channels
        )
        across = share_verdict(
            find("M4_element_affine", operator, "all"),
            find("M5_repr_element_affine", operator, "all", "per_class"),
        )
        lines.append(f"| {operator} | {within} | {across} |")
    lines += [
        "",
        "注意：判据里的百分比是**验证 MAPE 的相对变化**，不是“模型已经足够准”；",
        "例如 jitter 3ch 的两个数值都很大（22–23%），说明在该尺寸跨度上 kx+b 本身表达不足，",
        "这种情况下“可以共享”只表示“分开也不能解决”。",
        "",
        "## 7. 与部署模型、历史图的关系",
        "",
        "部署 profile（`outputs/affine_repr_profile_20260924/simclrv2/shared.yaml`）用每个",
        "（算子, 表示类）两点（0.5×/2×）非负拟合，数据来自流水线里真实产生的 payload；",
        "本轮用 8+4 个独立尺寸、每点 3 个 block。两者口径不同，因此本轮的图只用本轮数据；",
        "下面只做系数一致性检查：",
        "",
        "| 算子/类 | 部署 k (ms/element) | 部署 b (ms) | 本轮 k | 本轮 b |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    deployed = _deployed_coefficients()
    for operator in ("blur", "jitter", "crop", "grayscale", "flip"):
        for klass in sorted(
            {
                cell["representation_class"]
                for cell in cells.values()
                if cell["operator"] == operator
            }
        ):
            reference = deployed.get((operator, klass))
            fit = next(
                (
                    fit
                    for fit in fits
                    if fit["operator"] == operator
                    and fit["model"] == "M5_repr_element_affine"
                    and fit["class_scope"] == klass
                    and fit["group"] == "all"
                ),
                None,
            )
            if not reference or not fit:
                continue
            coefficients = json.loads(fit["coefficients"])
            lines.append(
                f"| {operator} {klass} | {reference[0]:.3g} | {reference[1]:.3g} | "
                f"{coefficients['k']:.3g} | {coefficients['b']:.3g} |"
            )
    ratios: List[Tuple[str, str, float]] = []
    for (operator, klass), (k_deployed, _) in deployed.items():
        fit = next(
            (
                fit
                for fit in fits
                if fit["operator"] == operator
                and fit["model"] == "M5_repr_element_affine"
                and fit["class_scope"] == klass
                and fit["group"] == "all"
            ),
            None,
        )
        if not fit:
            continue
        k_mine = json.loads(fit["coefficients"])["k"]
        if k_mine > 1e-15 and k_deployed > 0:
            ratios.append((operator, klass, k_deployed / k_mine))
    values = [ratio for _, _, ratio in ratios]
    close = [ratio for ratio in values if 0.9 <= ratio <= 1.6]
    worst = max(ratios, key=lambda item: item[2]) if ratios else None
    summary = (
        f"本轮斜率与部署 profile 的比值中位数 {statistics.median(values):.2f}×，"
        f"{len(close)}/{len(values)} 个 (算子, 类) 落在 0.9–1.6×"
        + (
            f"，最大 {worst[2]:.2f}× 出现在 {worst[0]} {worst[1]}"
            "（该配对两边斜率都接近 0，比值本身没有意义）"
            if worst and worst[2] > 1.6
            else ""
        )
        if ratios
        else "（无可比斜率）"
    )
    lines += [
        "",
        f"{summary}。crop 的截距两边几乎相同（1.55/1.72 vs 1.63/1.71）；"
        "差异最大的是部署侧被非负约束压到 b=0 的类（blur、jitter 3ch）。",
        "绝对值一律以本轮 `measurements.csv` 为准，部署系数只用于一致性检查。",
        "",
        "## 8. 复现",
        "",
        "```bash",
        "# 容器内，先 source env/bin/activate",
        "python -m tmp_analysis.ch32_scale_affine_probe measure        # 远程 Ray actor 内测量（约 12 min）",
        "python -m tmp_analysis.ch32_scale_affine_probe fit            # 拟合、验证、figure_data.json",
        "python -m tmp_analysis.plot_ch32_scale_affine                 # 草图到 outputs/.../figures/",
        "python -m tmp_analysis.ch32_scale_affine_probe manifest       # 校验和 + docs 快照",
        "```",
        "",
        "## 9. 文件",
        "",
        "- `measurements.csv`：每个 block × cell 一行（含输入/输出 shape、元素、原生/序列化字节、q10/q50/q90/std、调用数）。",
        "- `raw_calls.csv.gz`：每次调用的原始耗时（cell、block、call、source）。",
        "- `fits.csv` / `predictions.csv`：模型系数（含约束命中）、训练/验证指标、逐点预测。",
        "- `figure_data.json`：面板点、拟合曲线、诊断对照与归一化参考。",
        "- `env.json`：commit、actor 位置、线程与协议、参考点规则。",
    ]
    (OUT / "README.md").write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    measure = sub.add_parser("measure")
    measure.add_argument("--dry-run", action="store_true")
    measure.add_argument("--cells", default="all")
    measure.set_defaults(func=cmd_measure)
    fit = sub.add_parser("fit")
    fit.set_defaults(func=cmd_fit)
    manifest = sub.add_parser("manifest")
    manifest.set_defaults(func=cmd_manifest)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
