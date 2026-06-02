import argparse
import gc
import logging
import multiprocessing as mp
import os
import shutil
import sys
import time
import traceback
from queue import Empty
from pathlib import Path
from typing import Dict

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.cedar_utils import CedarEvalSpec
from evaluation.pipelines.bloom_oscar.cedar_dataset import get_dataset


def _has_cache(plan_path: Path) -> bool:
    with plan_path.open("r", encoding="utf-8") as f:
        plan = yaml.safe_load(f)
    pipes = plan.get("physical_plan", {}).get("pipes", {})
    return any(desc.get("name") == "ObjectDiskCachePipe" for desc in pipes.values())


def _generate_plan(
    name: str,
    selector: int,
    dataset_path: Path,
    profile_path: Path,
    out_dir: Path,
) -> Path:
    spec = CedarEvalSpec(
        batch_size=1,
        num_total_samples=None,
        num_epochs=1,
        kwargs={"dataset_path": str(dataset_path)},
        use_ray=True,
        profiled_stats=str(profile_path),
        run_profiling=False,
        disable_controller=True,
        disable_prefetch=True,
        disable_offload=False,
        disable_parallelism=True,
        disable_reorder=False,
        disable_fusion=False,
        disable_caching=False,
        use_my_optimizer=selector,
        generate_plan=True,
    )
    try:
        get_dataset(spec)
    except SystemExit as exc:
        if exc.code not in (0, None):
            raise

    generated = Path("/tmp/cedar_optimized_plan.yml")
    if not generated.exists():
        raise FileNotFoundError(generated)
    out_path = out_dir / f"{name}_plan.yml"
    shutil.copyfile(generated, out_path)
    return out_path


def _build_dataset(plan_path: Path, dataset_path: Path):
    spec = CedarEvalSpec(
        batch_size=1,
        num_total_samples=None,
        num_epochs=1,
        config=str(plan_path),
        kwargs={"dataset_path": str(dataset_path)},
        use_ray=True,
        disable_controller=True,
        disable_optimizer=True,
    )
    return get_dataset(spec)


def _prepare_dataset_subset(dataset_path: Path, out_dir: Path, num_input_lines: int) -> Path:
    if num_input_lines <= 0:
        return dataset_path

    subset_path = out_dir / f"dataset_first_{num_input_lines}.jsonl"
    with dataset_path.open("r", encoding="utf-8") as src, subset_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for idx, line in enumerate(src):
            if idx >= num_input_lines:
                break
            dst.write(line)
    return subset_path


def _consume_dataset(ds) -> Dict:
    start = time.perf_counter()
    count = 0
    for _ in ds:
        count += 1
    elapsed = time.perf_counter() - start
    return {
        "outputs": count,
        "seconds": elapsed,
        "samples_per_second": count / elapsed if elapsed > 0 else 0.0,
    }


def _clean_process_cache() -> None:
    cache_dir = Path("/tmp") / f"cedar_{os.getpid()}"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)


def time_plan(
    name: str,
    plan_path: Path,
    dataset_path: Path,
) -> Dict:
    uses_cache = _has_cache(plan_path)
    _clean_process_cache()

    ds = _build_dataset(plan_path, dataset_path)
    warmup = None
    if uses_cache:
        warmup = _consume_dataset(ds)

    timed = _consume_dataset(ds)
    del ds
    gc.collect()
    _clean_process_cache()
    return {
        "name": name,
        "plan_path": str(plan_path),
        "uses_cache": uses_cache,
        "warmup": warmup,
        "timed": timed,
    }


def _time_plan_child(
    name: str,
    plan_path: str,
    dataset_path: str,
    queue,
) -> None:
    try:
        logging.getLogger().setLevel(logging.WARNING)
        result = time_plan(
            name,
            Path(plan_path),
            Path(dataset_path),
        )
        queue.put(("ok", result))
    except Exception:
        queue.put(("error", traceback.format_exc()))


def time_plan_in_subprocess(
    name: str,
    plan_path: Path,
    dataset_path: Path,
    timeout_sec: float,
) -> Dict:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(
        target=_time_plan_child,
        args=(
            name,
            str(plan_path),
            str(dataset_path),
            queue,
        ),
    )
    proc.start()
    try:
        status, payload = queue.get(timeout=timeout_sec)
    except Empty as exc:
        proc.terminate()
        proc.join(30)
        if proc.is_alive():
            proc.kill()
            proc.join()
        raise TimeoutError(
            f"Timing subprocess timed out for {name} after {timeout_sec} seconds."
        ) from exc
    proc.join()
    if proc.exitcode != 0 or status != "ok":
        raise RuntimeError(
            f"Timing subprocess failed for {name} with exit code "
            f"{proc.exitcode}:\n{payload}"
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Cedar staged and joint DP plans for BLOOM/OSCAR, then "
            "run real wall-clock timing. Cached plans are warmed once and "
            "timed on the second pass."
        )
    )
    parser.add_argument(
        "--dataset_path",
        default="/tmp/redpajama_backup_3gb_for_bloom_oscar.jsonl",
    )
    parser.add_argument(
        "--profile_path",
        default="/tmp/bloom_oscar_stagewise_joint_profile_50_with_smp.yml",
    )
    parser.add_argument("--out_dir", default="/tmp/bloom_oscar_plan_timing")
    parser.add_argument(
        "--num_input_lines",
        type=int,
        default=200,
        help=(
            "Use the first K input lines as a finite real-data subset and "
            "fully consume it for each timed run. Set <=0 to use the full file."
        ),
    )
    parser.add_argument("--timeout_sec", type=float, default=600.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    dataset_path = Path(args.dataset_path)
    profile_path = Path(args.profile_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)
    if not profile_path.exists():
        raise FileNotFoundError(profile_path)

    run_dataset_path = _prepare_dataset_subset(
        dataset_path, out_dir, args.num_input_lines
    )

    plans = {
        "cedar_staged": _generate_plan(
            "cedar_staged", 5, run_dataset_path, profile_path, out_dir
        ),
        "joint_dp": _generate_plan("joint_dp", 2, run_dataset_path, profile_path, out_dir),
    }

    results = {}
    for name, plan_path in plans.items():
        results[name] = time_plan_in_subprocess(
            name, plan_path, run_dataset_path, args.timeout_sec
        )
    result_path = out_dir / "timing_results.yml"
    with result_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(results, f, sort_keys=False)

    print(f"dataset_path: {dataset_path}")
    print(f"run_dataset_path: {run_dataset_path}")
    print(f"profile_path: {profile_path}")
    print(f"num_input_lines: {args.num_input_lines}")
    print(f"result_path: {result_path}")
    for name, result in results.items():
        timed = result["timed"]
        print(
            f"{name}: uses_cache={result['uses_cache']} "
            f"timed_seconds={timed['seconds']:.6f} "
            f"outputs={timed['outputs']} "
            f"throughput={timed['samples_per_second']:.6f}"
        )
        if result["warmup"] is not None:
            warmup = result["warmup"]
            print(
                f"{name}_warmup: seconds={warmup['seconds']:.6f} "
                f"outputs={warmup['outputs']} "
                f"throughput={warmup['samples_per_second']:.6f}"
            )


if __name__ == "__main__":
    main()
