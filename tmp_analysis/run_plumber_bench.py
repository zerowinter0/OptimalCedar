"""Offline matrix: Cedar vs Plumber-style vs our model on five workloads.

For every target_pipeline workload the driver
  1. profiles once (layered: cm_model + scaling + object boundary),
  2. runs the three optimizers in one round-robin harness with three repeats,
  3. keeps profile, plans, logs and JSON results under the output directory.

Run inside the container with nohup; see --samples/--repeats/--workloads.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/workspace/OptimalCedar")
MANIFESTS = ROOT / "datasets/target_pipeline_bench"
TOKENIZER = ROOT / "evaluation/datasets/target_pipeline/clip_tokenizer"
DATASET = "evaluation/pipelines/target_pipeline/cedar_dataset.py"
ADDRESS = "172.23.166.105:6379"
SNAPSHOT_SOURCE = ROOT / "outputs/simclrv2_four_remote_20260911"
WORKLOADS = ["blip", "clip", "dino", "simclr", "swav"]
OPTIMIZERS = ["optimizer", "plumber_optimizer", "dp_optimizer"]

# Text / long-filtering workloads live outside target_pipeline and take no
# dataset kwargs: their entry modules resolve their own frozen dataset path.
EXTRA_WORKLOADS = {
    "alpaca_cot": "evaluation/pipelines/alpaca_cot/cedar_dataset.py",
    "pile_hackernews": "evaluation/pipelines/pile_hackernews/cedar_dataset.py",
    "pile_pubmed_abstracts": (
        "evaluation/pipelines/pile_pubmed_abstracts/cedar_dataset.py"
    ),
    "pile_uspto_backgrounds": (
        "evaluation/pipelines/pile_uspto_backgrounds/cedar_dataset.py"
    ),
    "stackexchange": "evaluation/pipelines/stackexchange/cedar_dataset.py",
    "wikitext103": "evaluation/pipelines/wikitext103/cedar_dataset.py",
}


def status(out, state, **kwargs):
    (out / "status.json").write_text(
        json.dumps(
            dict(state=state,
                 utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                 **kwargs),
            indent=2,
        )
    )


def prepare(out):
    (out / "logs").mkdir(parents=True, exist_ok=True)
    (out / "results").mkdir(parents=True, exist_ok=True)
    if not (out / "modules").exists():
        modules = out / "modules"
        modules.mkdir(parents=True)
        ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
        for package in ("cedar", "pico_multimodal"):
            source = ROOT / package
            if source.is_dir():
                shutil.copytree(source, modules / package, ignore=ignore)
        # Only the importable evaluation code is needed by Ray actors; the
        # datasets directory is hundreds of gigabytes and must never be copied.
        evaluation = modules / "evaluation"
        evaluation.mkdir()
        for name in ("__init__.py", "cedar_utils.py", "profiler.py"):
            source = ROOT / "evaluation" / name
            if source.is_file():
                shutil.copy2(source, evaluation / name)
        shutil.copytree(
            ROOT / "evaluation/pipelines", evaluation / "pipelines", ignore=ignore
        )
        # The CLIP tokenizer must be visible to remote backends: ship it inside
        # the package directory that the Ray working directory carries.
        tokenizer = (
            ROOT / "evaluation/datasets/target_pipeline/clip_tokenizer"
        )
        if tokenizer.is_dir():
            shutil.copytree(
                tokenizer,
                evaluation / "pipelines/target_pipeline/datasets/"
                "target_pipeline/clip_tokenizer",
                ignore=ignore,
            )
    if not (out / "entry.py").exists():
        shutil.copy2(SNAPSHOT_SOURCE / "entry.py", out / "entry.py")


def dataset_args(workload, samples, views=None):
    if workload in EXTRA_WORKLOADS:
        return EXTRA_WORKLOADS[workload], []
    if workload == "simclr":
        args = [f"workload=simclr"]
        dataset_file = "evaluation/pipelines/target_pipeline/simclr/cedar_dataset.py"
    else:
        args = [f"workload={workload}",
                f"dataset_path={MANIFESTS / (workload + '.jsonl')}"]
        if views is not None and workload in ("dino", "swav"):
            args.append(f"views={views}")
        if workload == "clip":
            args.append(f"tokenizer_path={TOKENIZER}")
        dataset_file = DATASET
    return dataset_file, args


def run(out, tag, args, env_extra):
    env = dict(os.environ)
    env.update(env_extra)
    log = out / "logs" / (tag + ".log")
    started = time.time()
    with log.open("w") as handle:
        code = subprocess.run(
            [sys.executable, "-u", str(out / "entry.py"), *args],
            cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, env=env,
        ).returncode
    print(f"[bench] {tag} exit={code} wall={time.time() - started:.1f}s "
          f"log={log}", flush=True)
    return code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workloads", nargs="+", default=WORKLOADS)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--profile-time", type=float, default=10.0)
    parser.add_argument("--adaptive-min", type=float, default=3.0)
    parser.add_argument("--adaptive-max", type=float, default=30.0)
    parser.add_argument("--views", type=int, default=None,
                        help="view count for dino/swav (default keeps the recipe)")
    parser.add_argument("--cedar-timeout", type=float, default=600.0)
    parser.add_argument("--optimizer-limit", type=float, default=1800.0)
    parser.add_argument("--workers", type=int, default=8,
                        help="fixed local worker count (execution W)")
    parser.add_argument("--skip-profile", action="store_true")
    args = parser.parse_args()
    out = args.output
    prepare(out)
    (out / "metadata.json").write_text(
        json.dumps(
            {
                "workloads": args.workloads,
                "optimizers": OPTIMIZERS,
                "samples": args.samples,
                "batch_size": args.batch_size,
                "repeats": args.repeats,
                "W": args.workers,
                "cpu_budget": 64,
                "ray_cpu_budget": 64,
                "profile_time_sec": args.profile_time,
                "adaptive_profile_min_sec": args.adaptive_min,
                "adaptive_profile_max_sec": args.adaptive_max,
                "manifests": str(MANIFESTS),
                "ray_ip": ADDRESS,
                "note": "optimizer=original Cedar, plumber_optimizer=Plumber-style per-stage "
                        "width allocation, dp_optimizer=our model",
            },
            indent=2,
        )
    )
    profile_env = {
        "CEDAR_RAY_PLACEMENT_RESOURCE": "cedar_remote",
        "CEDAR_REUSE_BOUNDARY_MODEL": "0",
        "CEDAR_LAYERED_ADAPTIVE_PROFILE": "1",
        "CEDAR_PROFILE_TIME_SEC": str(args.profile_time),
        "CEDAR_CM_SWEEP_TIME_SEC": str(args.profile_time),
        "CEDAR_PROFILE_SCALING_WIDTHS": "1,2,4,8",
        "CEDAR_PROFILE_SCALING_TOP_K": "5",
        "CEDAR_PROFILE_SCALING_MAX_SEC": "10",
        "CEDAR_ADAPTIVE_PROFILE_MIN_SEC": str(args.adaptive_min),
        "CEDAR_ADAPTIVE_PROFILE_MAX_SEC": str(args.adaptive_max),
        "CEDAR_PROFILE_MATCH_FIXED_LOCAL_WORKERS": str(args.workers),
    }
    summary = {}
    existing = out / "summary.json"
    if existing.is_file():
        try:
            summary.update(json.loads(existing.read_text()))
        except Exception:  # noqa: BLE001
            pass
    for workload in args.workloads:
        status(out, f"profile:{workload}", done=list(summary))
        dataset_file, kwargs = dataset_args(workload, args.samples, args.views)
        profile = out / f"{workload}_profile.yaml"
        kwargs_flag = ["--dataset_kwargs", ",".join(kwargs)] if kwargs else []
        if not (args.skip_profile and profile.is_file()):
            code = run(out, f"profile_{workload}", [
                "evaluation/eval_cedar.py",
                "--dataset_file", dataset_file,
                *kwargs_flag,
                "--batch_size", str(args.batch_size),
                "--num_total_samples", str(args.samples),
                "--run_profiling",
                "--profiled_stats", str(profile),
                "--use_my_optimizer", "2",
                "--use_ray", "--ray_ip", ADDRESS,
                "--disable_controller", "--disable_caching",
            ], profile_env)
            if code != 0 or not profile.is_file():
                summary[workload] = {"status": "profile_failed"}
                continue
        status(out, f"compare:{workload}", done=list(summary))
        results = out / "results" / f"{workload}.json"
        code = run(out, f"compare_{workload}", [
            "evaluation/compare_optimizer_perf.py",
            "--dataset_file", dataset_file,
            *kwargs_flag,
            "--batch_size", str(args.batch_size),
            "--num_total_samples", str(args.samples),
            "--full_data_run",
            "--profiled_stats", str(profile),
            "--use_ray", "--ray_ip", ADDRESS,
            "--enable_local_parallelism",
            "--disable_caching",
            "--match_profile_resources",
            "--cpu_budget", "64",
            "--ray_cpu_budget", "64",
            "--fixed_local_workers_ablation", str(args.workers),
            "--optimizers", *OPTIMIZERS,
            "--optimizer_time_limit_sec", str(int(args.optimizer_limit)),
            "--cedar_reorder_timeout_sec", str(int(args.cedar_timeout)),
            "--disable_cedar_runtime_timeout",
            "--num_repeats", str(args.repeats),
            "--results_path", str(results),
        ], {
            "CEDAR_RAY_PLACEMENT_RESOURCE": "cedar_remote",
            "CEDAR_PROFILE_MATCH_FIXED_LOCAL_WORKERS": str(args.workers),
        })
        summary[workload] = {"status": "ok" if code == 0 else "compare_failed"}
        if results.is_file():
            try:
                payload = json.loads(results.read_text())
                summary[workload]["runs"] = {
                    run["optimizer"]: round(run["perf_time_sec"], 3)
                    for run in payload.get("runs", [])
                }
            except Exception as exc:  # noqa: BLE001
                summary[workload]["results_error"] = str(exc)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    status(out, "completed", summary=summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
