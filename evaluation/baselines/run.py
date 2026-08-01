"""Unified runner for native PyTorch, tf.data, and Ray Data baselines.

Data-Juicer, Plumber, and FastFlow use separate pinned environments. Their
entries remain visible through ``--matrix`` and are launched by the top-level
orchestration script rather than imported into Cedar's Python environment.
"""

from __future__ import annotations

import argparse
import importlib.util
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional

from evaluation.baselines.registry import (
    SYSTEMS,
    WORKLOADS,
    get_entry,
    iter_entries,
    matrix_as_dict,
)
REPO_ROOT = Path(__file__).resolve().parents[2]
NATIVE_SYSTEMS = ("pytorch", "tensorflow", "ray")


class RayEvalSpec:
    def __init__(
        self,
        *,
        batch_size: int,
        num_workers: int,
        num_total_samples: Optional[int],
        kwargs: Dict[str, Any],
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_total_samples = num_total_samples
        self.kwargs = dict(kwargs)


def _import_module(path: Path):
    rel = path.resolve().relative_to(REPO_ROOT)
    name = ".".join(rel.with_suffix("").parts)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _git_state(path: Path) -> Dict[str, Any]:
    safe_arg = f"safe.directory={path}"

    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-c", safe_arg, "-C", str(path), *args],
            text=True,
        ).strip()

    try:
        return {
            "commit": run("rev-parse", "HEAD"),
            "dirty": bool(run("status", "--short")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _build_dataset(args: argparse.Namespace, entry):
    if args.system == "tensorflow" and args.workload in {
        "llava_pretrain",
        "wikitext103",
        "wikitext103_cache",
        "pile_europarl",
        "redpajama_code",
        "pile_hackernews",
        "pile_pubmed_abstracts",
        "pile_freelaw",
        "pile_uspto_backgrounds",
    }:
        # These modules import Hugging Face Transformers, which probes the
        # PyTorch backend.  In the pinned CUDA environment, letting that probe
        # load PyTorch after TensorFlow can abort in std::random_device.
        # Establish the backend order before importing the tf.data module.
        import torch  # noqa: F401

    module_path = REPO_ROOT / str(entry.implementation)
    module = _import_module(module_path)
    getter = getattr(module, "get_dataset")
    kwargs = dict(args.dataset_kwargs)
    if args.dataset_path:
        kwargs["dataset_path"] = args.dataset_path
    if args.image_root:
        kwargs["image_root"] = args.image_root

    if args.system == "pytorch":
        from evaluation.torch_utils import TorchEvalSpec

        spec = TorchEvalSpec(
            args.batch_size,
            args.workers,
            args.epochs,
            args.num_samples,
            kwargs=kwargs,
        )
        return getter(spec)
    if args.system == "tensorflow":
        # LLaVA's reference operators include PyTorch-backed CLIP/BLIP
        # callbacks.  Load those operator modules before TensorFlow initializes
        # CUDA/runtime state; importing PyTorch after TensorFlow can abort in
        # libstdc++ random_device initialization in this pinned environment.
        if args.workload in {
            "llava_pretrain",
            "redpajama_c4",
            "stackexchange",
            "pile_europarl",
            "redpajama_code",
            "pile_hackernews",
            "pile_pubmed_abstracts",
            "pile_freelaw",
            "pile_uspto_backgrounds",
        }:
            from evaluation.pipelines.datajuicer_workloads import build_stages

            build_stages(
                args.workload,
                image_root=str(kwargs.get("image_root") or ""),
            )

        import tensorflow as tf
        from evaluation.tf_utils import TFEvalSpec

        parallelism = (
            tf.data.AUTOTUNE if args.workers == -1 else args.workers
        )
        spec = TFEvalSpec(
            args.batch_size,
            parallelism,
            args.epochs,
            args.num_samples,
            kwargs=kwargs,
        )
        dataset = getter(spec)
        if args.workload.endswith("_cache"):
            cache_file = (
                Path(args.cache_dir).resolve()
                / args.workload
                / "tf_data_cache"
            )
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            dataset = dataset.cache(str(cache_file))
        return dataset

    import inspect

    spec = RayEvalSpec(
        batch_size=args.batch_size,
        num_workers=args.workers,
        num_total_samples=args.num_samples,
        kwargs=kwargs,
    )
    if len(inspect.signature(getter).parameters) == 0:
        return getter()
    return getter(spec)


def _iter_dataset(system: str, dataset: Any) -> Iterator[Any]:
    if system == "ray":
        return iter(dataset.iter_rows())
    return iter(dataset)


def _consume(
    system: str,
    dataset: Any,
    *,
    num_samples: Optional[int],
    batch_size: int,
) -> int:
    count = 0
    increment = 1 if system == "ray" else batch_size
    for _ in _iter_dataset(system, dataset):
        count += increment
        if num_samples is not None and count >= num_samples:
            break
    return count


def _version(module_name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(module_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _run_native(args: argparse.Namespace, entry) -> Dict[str, Any]:
    dataset = None
    try:
        if args.system == "ray":
            import ray

            if not ray.is_initialized():
                ray.init(
                    address=args.ray_address or None,
                    ignore_reinit_error=True,
                )
            # Configure Dataset resources only after the driver has explicitly
            # joined the requested cluster.  Calling DataContext before
            # ray.init() implicitly starts a core worker and a later explicit
            # initialization aborts the process.
            ray.data.DataContext.get_current().execution_options.resource_limits.cpu = (
                args.workers
            )

        setup_start = time.perf_counter()
        dataset = _build_dataset(args, entry)
        if args.system == "ray" and args.num_samples is not None:
            # Breaking a Ray iterator early leaves upstream map tasks running
            # and can make ray.shutdown() terminate the driver with an error.
            # Limit the logical Dataset so Ray can finish the requested cell
            # cleanly and the measured cardinality is exact.
            dataset = dataset.limit(args.num_samples)
        setup_time = time.perf_counter() - setup_start

        warmup_time = 0.0
        if args.workload.endswith("_cache") and args.system in {
            "tensorflow",
            "ray",
        }:
            warmup_start = time.perf_counter()
            if args.system == "ray":
                dataset = dataset.materialize()
            else:
                _consume(
                    args.system,
                    dataset,
                    num_samples=args.num_samples,
                    batch_size=args.batch_size,
                )
            warmup_time = time.perf_counter() - warmup_start

        measured_start = time.perf_counter()
        count = _consume(
            args.system,
            dataset,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
        )
        measured_time = time.perf_counter() - measured_start
    finally:
        close = getattr(dataset, "close", None)
        if callable(close):
            close()
        # Each benchmark cell runs in a dedicated process.  Do not explicitly
        # call ray.shutdown() here: Ray Dataset may still be retiring iterator
        # worker threads, and shutting the core worker down underneath them can
        # race with their cleanup and abort before the JSON result is written.
        # Normal process teardown releases the driver after the result is saved.

    return {
        "schema_version": 1,
        "system": args.system,
        "workload": args.workload,
        "entry": entry.to_dict(),
        "num_samples": count,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "setup_time_sec": setup_time,
        "cache_warmup_time_sec": warmup_time,
        "measured_time_sec": measured_time,
        "throughput_samples_per_sec": (
            count / measured_time if measured_time > 0 else None
        ),
        "versions": {
            "python": platform.python_version(),
            "torch": _version("torch"),
            "tensorflow": _version("tensorflow"),
            "ray": _version("ray"),
        },
        "optimalcedar_git": _git_state(REPO_ROOT),
        "datajuicer_git": _git_state(REPO_ROOT / "data-juicer"),
    }


def _validate_entry(entry) -> Dict[str, Any]:
    if entry.status != "supported":
        return {
            "system": entry.system,
            "workload": entry.workload,
            "status": "unsupported",
            "reason": entry.reason,
        }
    path = REPO_ROOT / str(entry.implementation)
    result = {
        "system": entry.system,
        "workload": entry.workload,
        "status": "ok" if path.is_file() else "missing",
        "implementation": str(path),
    }
    if path.is_file() and entry.system in NATIVE_SYSTEMS:
        try:
            module = _import_module(path)
            if not callable(getattr(module, "get_dataset", None)):
                raise AttributeError("missing callable get_dataset")
        except Exception as exc:
            result["status"] = "import_error"
            result["error"] = repr(exc)
    return result


def _print_markdown_matrix() -> None:
    print("| workload | " + " | ".join(SYSTEMS) + " |")
    print("|---|" + "|".join("---" for _ in SYSTEMS) + "|")
    for workload in WORKLOADS:
        cells = []
        for system in SYSTEMS:
            entry = get_entry(system, workload)
            cells.append("supported" if entry.status == "supported" else "—")
        print(f"| {workload} | " + " | ".join(cells) + " |")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", action="store_true")
    parser.add_argument("--matrix-json", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--system", choices=SYSTEMS)
    parser.add_argument("--workload", choices=WORKLOADS)
    parser.add_argument("--dataset-path")
    parser.add_argument("--image-root")
    parser.add_argument("--dataset-kwargs", type=json.loads, default={})
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--num-samples", type=int)
    parser.add_argument(
        "--cache-dir",
        default="evaluation/baselines/cache",
    )
    parser.add_argument("--ray-address", default="")
    parser.add_argument("--results-path")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.matrix:
        _print_markdown_matrix()
        return
    if args.matrix_json:
        print(json.dumps(matrix_as_dict(), indent=2, sort_keys=True))
        return
    if args.validate:
        entries: Iterable[Any]
        if args.system and args.workload:
            entries = [get_entry(args.system, args.workload)]
        else:
            entries = iter_entries()
        results = [_validate_entry(entry) for entry in entries]
        print(json.dumps(results, indent=2, sort_keys=True))
        if any(
            result["status"] in {"missing", "import_error"}
            for result in results
        ):
            raise SystemExit(1)
        return

    if not args.system or not args.workload:
        raise SystemExit("--system and --workload are required for a run")
    entry = get_entry(args.system, args.workload)
    if entry.status != "supported":
        raise SystemExit(
            f"{args.system}/{args.workload} is unsupported: {entry.reason}"
        )
    if args.system not in NATIVE_SYSTEMS:
        raise SystemExit(
            f"{args.system} requires its separate environment; use the "
            "orchestration command documented for this registry entry."
        )

    result = _run_native(args, entry)
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.results_path:
        output = Path(args.results_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
