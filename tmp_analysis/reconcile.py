"""P5: reconcile every modeled stage term against the executing plan.

Usage (inside the container):
  python -u tmp_analysis/reconcile.py <profile.yaml> name=plan.yaml [name2=plan2.yaml]

For every plan it (1) executes the plan with per-stage runtime tracing enabled,
(2) aggregates the wall-clock service reported by the eight local workers, and
(3) prints the DP's own per-stage terms next to those measurements.
Set CEDAR_RECONCILE_SKIP_RUN=1 to re-print the report from cached measurements.
"""
import json
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path("/workspace/OptimalCedar")
sys.path.insert(0, str(ROOT))

os.environ.setdefault("CEDAR_MATCH_PROFILE_RESOURCES", "1")
os.environ.setdefault("CEDAR_PROFILE_MATCH_FIXED_LOCAL_WORKERS", "8")
os.environ.setdefault("CEDAR_PROFILE_MATCH_CPU_BUDGET", "64")
os.environ.setdefault("CEDAR_PROFILE_MATCH_RAY_CPU_BUDGET", "64")

import yaml  # noqa: E402

from cedar.compose import OptimizerOptions  # noqa: E402
from cedar.compose.dp_optimizer import (  # noqa: E402
    BlockCandidateProvider,
    DpOptimizer,
)
from cedar.compose.optimizer import PhysicalPlan  # noqa: E402
from cedar.sources import LocalFSSource  # noqa: E402
from evaluation.pipelines.simclrv2 import cedar_dataset as original_simclrv2  # noqa: E402
from evaluation.pipelines.target_pipeline.simclr.cedar_dataset import (  # noqa: E402
    DATASET_LOC,
    SimCLRV2Feature,
)

ENTRY = ROOT / "outputs/simclrv2_four_remote_20260911/entry.py"
DATA = "evaluation/pipelines/target_pipeline/simclr/cedar_dataset.py"
ADDRESS = "172.23.166.105:6379"
DEFAULT_PROFILE = ROOT / "outputs/simclrv2_scaling_20260911/profile.yaml"
WORK = ROOT / "tmp_analysis/reconcile"
PLANS = ROOT / "tmp_analysis/plan_runs"


def normalize(name, path):
    data = yaml.safe_load(Path(path).read_text())
    plan = data.get("physical_plan", data)
    payload = plan.get("feature_r0", plan)
    payload["graph"] = {int(k): v for k, v in payload["graph"].items()}
    payload["pipes"] = {int(k): v for k, v in payload["pipes"].items()}
    for pipe in payload["pipes"].values():
        # Dumped plans may omit the variant entirely (fused inner pipes) or
        # carry an explicit ``null``; both must fall back to INPROCESS and the
        # matching variant context so PhysicalPlan.from_dict can rebuild them.
        pipe["variant"] = pipe.get("variant") or "INPROCESS"
        pipe.setdefault("variant_ctx", {"variant_type": pipe["variant"]})
    payload["n_local_workers"] = int(os.environ.get("CEDAR_PLAN_WORKERS", "8"))
    PLANS.mkdir(parents=True, exist_ok=True)
    target = PLANS / f"{name}_reconcile.yaml"
    target.write_text(yaml.safe_dump({"physical_plan": payload}))
    return target, payload


def execute(name, plan_path, profile, samples):
    trace_dir = WORK / name
    log = WORK / f"{name}.log"
    WORK.mkdir(parents=True, exist_ok=True)
    batch = os.environ.get("CEDAR_RECONCILE_BATCH", "1")
    env = dict(os.environ)
    env.update(
        {
            "CEDAR_RAY_PLACEMENT_RESOURCE": "cedar_remote",
            "CEDAR_RECONCILE_DIR": str(trace_dir),
        }
    )
    cmd = [
        sys.executable,
        "-u",
        str(ENTRY),
        "evaluation/eval_cedar.py",
        "--dataset_file",
        DATA,
        "--batch_size",
        batch,
        "--num_total_samples",
        str(samples),
        "--profiled_stats",
        str(profile),
        "--master_feature_config",
        str(plan_path),
        "--use_ray",
        "--ray_ip",
        ADDRESS,
        "--disable_controller",
        "--disable_caching",
    ]
    with log.open("w") as handle:
        code = subprocess.run(
            cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, env=env
        ).returncode
    text = log.read_text(errors="replace")
    match = re.search(r"Total time: ([0-9.]+)s", text)
    total = float(match.group(1)) if match else None
    measured = {}
    raw = {}
    buffers = {}
    services = {}
    workers = 0
    for path in sorted(trace_dir.glob("worker_*.json")):
        payload = json.loads(path.read_text())
        workers += 1
        for pid, value in payload["wall_latency_ns_per_sample"].items():
            measured.setdefault(int(pid), []).append(value)
        for pid, values in payload.get("wall_latency_samples", {}).items():
            raw.setdefault(int(pid), []).extend(values)
        for pid, values in payload.get("buffer_size_samples", {}).items():
            buffers.setdefault(int(pid), []).extend(values)
        for pid, entry in (payload.get("service_stats") or {}).items():
            services.setdefault(int(pid), []).append(entry)
    median = {
        pid: statistics.median(values) for pid, values in measured.items()
    }
    stats = {
        pid: {
            "n": len(values),
            "min_ms": min(values) / 1e6,
            "p25_ms": statistics.quantiles(values, n=4)[0] / 1e6
            if len(values) > 3
            else min(values) / 1e6,
            "median_ms": statistics.median(values) / 1e6,
            "mean_ms": statistics.fmean(values) / 1e6,
            "max_ms": max(values) / 1e6,
            "mean_buffer": (
                statistics.fmean(buffers[pid]) if buffers.get(pid) else None
            ),
        }
        for pid, values in raw.items()
    }
    summary = {
        "plan": name,
        "total_time_sec": total,
        "workers_with_trace": workers,
        "measured_wall_ns_per_sample": median,
        "stage_statistics": stats,
        "service_stats": {
            pid: {
                "workers": len(entries),
                "count": sum(e.get("count", 0) for e in entries),
                "mean_ms_per_sample": statistics.fmean(
                    e["mean_ms_per_sample"] for e in entries
                    if e.get("mean_ms_per_sample") is not None
                )
                if any(e.get("mean_ms_per_sample") is not None for e in entries)
                else None,
            }
            for pid, entries in services.items()
        },
        "measured_wall_samples": {
            str(pid): len(values) for pid, values in measured.items()
        },
    }
    (WORK / f"{name}.measured.json").write_text(json.dumps(summary, indent=2))
    print(
        f"[reconcile] {name}: exit={code} total={total}s "
        f"workers={workers} log={log}",
        flush=True,
    )
    return summary


def build_feature():
    data_dir = (
        Path(original_simclrv2.__file__).resolve().parents[2].joinpath(DATASET_LOC)
    )
    batch_size = int(os.environ.get("CEDAR_RECONCILE_BATCH", "1"))
    feature = SimCLRV2Feature(batch_size=batch_size)
    feature.apply(
        LocalFSSource(str(data_dir / "imagenette2" / "train"), recursive=True)
    )
    return feature


def options():
    return OptimizerOptions(
        enable_prefetch=True,
        est_throughput=None,
        available_local_cpus=64,
        enable_offload=True,
        enable_reorder=True,
        enable_local_parallelism=True,
        enable_fusion=True,
        num_samples=9469,
        use_my_optimizer=2,
        reorder_timeout_sec=3600.0,
    )


def stage_path(plan):
    graph = {int(k): v for k, v in plan["graph"].items()}
    children = {
        child
        for value in graph.values()
        for child in (
            [int(x) for x in value.split(",") if x.strip()]
            if isinstance(value, str)
            else list(value)
        )
    }
    current = next(pid for pid in graph if pid not in children)
    path = []
    while True:
        path.append(current)
        value = graph.get(current)
        nxt = (
            [int(x) for x in value.split(",") if x.strip()]
            if isinstance(value, str)
            else list(value)
        )
        if not nxt:
            break
        current = nxt[0]
    return path


def analyze(
    name,
    payload,
    profile,
    measured,
    stats=None,
    services=None,
    workers=None,
    total_time=None,
    samples=None,
):
    workers = max(1, int(workers or payload.get("n_local_workers") or 1))
    total_time = float(total_time or 0.0)
    samples = int(samples or 0)
    feature = build_feature()
    optimizer = DpOptimizer()
    feature.set_optimizer(optimizer)
    optimizer.run(str(profile), options())
    inner_ops = optimizer._dp_inner_ops
    plan = PhysicalPlan.from_dict(payload)
    blocks = optimizer._dp_blocks_from_physical_plan(plan, inner_ops)
    provider = BlockCandidateProvider(optimizer, inner_ops)
    provider.prepare()
    prev_mask = 0
    rows = []
    for order, variant, _, parallelism in blocks:
        block = provider.candidate_for_order(
            order, variant, prefix_mask=prev_mask, parallelism=parallelism
        )
        boundary_local, boundary_external = (
            optimizer._dp_stage_boundary_components(prev_mask, block)
        )
        # The cross-host payload budget is part of the stage's per-record
        # service, and the prefetch window a plan declares decides whether it
        # pays the oversized-window penalty.  Read both here so the printed
        # row is exactly what the DP objective charges for this stage.
        stage_bytes = optimizer._dp_stage_transport_bytes(prev_mask, block)
        declared = None
        for p_id, desc in payload["pipes"].items():
            fused = desc.get("fused_pipes")
            members = [int(x) for x in fused] if fused else [int(p_id)]
            if members == [inner_ops[i] for i in order]:
                ctx = desc.get("variant_ctx") or {}
                declared = ctx.get("max_inflight")
                break
        window_bytes = optimizer._dp_ray_window_bytes(
            order,
            parallelism,
            declared_inflight=declared,
            stage_bytes=stage_bytes,
        )
        cross_host = optimizer._dp_cross_host_lane_ms(
            stage_bytes, workers, window_bytes=window_bytes
        )
        rtt = optimizer._dp_cross_host_round_trip_ms(stage_bytes)
        floor = None
        if variant.name in ("RAY", "TF_RAY"):
            floor = cross_host
            if floor is None:
                floor = optimizer._dp_transport_floor_ms(
                    stage_bytes, workers=workers
                )
        offload_path = (
            block.cost / max(1, parallelism)
            + boundary_local
            + boundary_external
            + rtt
        )
        # Per-record term this stage contributes to the objective exactly as
        # ``_replay_concurrency_aware_objective`` charges it: an INPROCESS
        # block adds to the worker chain, an SMP stage owns its own lane, and
        # a Ray stage puts its offload path on the worker chain but can never
        # beat the cross-host payload budget.
        if variant.name in ("RAY", "TF_RAY"):
            charged = max(offload_path, floor)
        elif variant.name == "SMP":
            charged = (
                block.cost
                * optimizer._dp_stage_factor(variant)
                / max(1, parallelism)
                + boundary_external
                + boundary_local
            )
        else:
            charged = block.cost + boundary_external
        rows.append(
            {
                "mask": block.mask,
                "ops": [inner_ops[i] for i in order],
                "variant": variant.name,
                "width": parallelism,
                "compute": block.cost,
                "compute_per_width": block.cost / parallelism,
                "boundary_local": boundary_local,
                "boundary_external": boundary_external,
                "stage_bytes": stage_bytes,
                "window_bytes": window_bytes,
                "cross_host_floor": floor,
                "round_trip": rtt,
                "charged": charged,
                "lane_total": (
                    block.cost / parallelism + boundary_external
                    if variant.name in ("RAY", "TF_RAY", "SMP")
                    else block.cost + boundary_external
                ),
            }
        )
        prev_mask |= block.mask

    path = stage_path(payload)
    prefetch = [
        pid for pid in path if payload["pipes"][pid]["name"] == "PrefetcherPipe"
    ]
    stages = [pid for pid in path if pid not in prefetch]
    # Drop the source (lister): the DP does not model it as a block.
    stages = [
        pid
        for pid in stages
        if payload["pipes"][pid]["name"] != "LocalFSListerPipe"
    ]
    measured_stages = stages[: len(rows)]

    out = {
        "plan": name,
        "profile": str(profile),
        "path": path,
        "rows": rows,
        "stages": measured_stages,
    }
    (WORK / f"{name}.model.json").write_text(json.dumps(out, indent=2))

    print(f"\n===== {name} =====", flush=True)
    print(
        "  stage_pipe  ops                variant  w   compute/w  bnd(ms)"
        "   windowMB   floorMB  stage_term  meas_svc  meas_lat  svc_ratio",
        flush=True,
    )
    local_serial = 0.0
    smp_serial = 0.0
    gpu_serial = 0.0
    total_measured = 0.0
    for row, pid in zip(rows, measured_stages):
        value = measured.get(str(pid)) or measured.get(pid)
        measured_ms = value / 1e6 if value else float("nan")
        stage_term = row["charged"]
        if row["variant"] in ("RAY", "TF_RAY", "INPROCESS"):
            local_serial += stage_term
        elif row["variant"] == "SMP":
            smp_serial = max(smp_serial, stage_term)
        else:
            gpu_serial += stage_term
        total_measured += measured_ms
        st = (stats or {}).get(str(pid), {})
        svc = (
            (services or {}).get(str(pid))
            or (services or {}).get(pid)
            or {}
        )
        service_ms = svc.get("mean_ms_per_sample")
        per_record_service_ms = (
            service_ms / max(1, int(row["width"]))
            if service_ms is not None
            else None
        )
        svc_ratio = (
            per_record_service_ms / (row["compute"] / max(1, int(row["width"])))
            if per_record_service_ms is not None
            else float("nan")
        )
        boundary = row["boundary_local"] + row["boundary_external"]
        floor_ms = row["cross_host_floor"]
        print(
            f"  {pid:<10} {str(row['ops']):<19} {row['variant']:<8} "
            f"{row['width']:<3} {row['compute'] / row['width']:9.3f} "
            f"{boundary:7.3f} "
            f"{row['window_bytes'] / 1e6:9.1f} "
            f"{(f'{floor_ms:.3f}' if floor_ms is not None else '-'):>8} "
            f"{stage_term:10.3f} "
            f"{(f'{per_record_service_ms:.3f}' if per_record_service_ms is not None else 'n/a'):>8} "
            f"{st.get('median_ms', measured_ms):9.3f} "
            f"{svc_ratio:9.2f}",
            flush=True,
        )
    score = max(local_serial, smp_serial, gpu_serial)
    measured_lane = (
        total_time * 1000.0 / samples * workers if total_time else float("nan")
    )
    print(
        f"  score: local={local_serial:.3f} smp={smp_serial:.3f} "
        f"gpu={gpu_serial:.3f} -> model={score:.3f} ms/record/worker; "
        f"measured={measured_lane:.3f} ms/record/worker "
        f"(end-to-end {total_time}s, {samples} records, W={workers}); "
        f"ratio={measured_lane / score if score else float('nan'):.2f}; "
        f"stage latencies sum to {total_measured:.1f} ms",
        flush=True,
    )


def main():
    args = [a for a in sys.argv[1:]]
    profile = Path(args[0]) if args else DEFAULT_PROFILE
    specs = []
    for arg in args[1:]:
        name, _, path = arg.partition("=")
        specs.append((name, path))
    samples = int(os.environ.get("CEDAR_RECONCILE_SAMPLES", "9469"))
    skip = os.environ.get("CEDAR_RECONCILE_SKIP_RUN") == "1"
    for name, path in specs:
        plan_path, payload = normalize(name, path)
        if skip:
            summary = json.loads((WORK / f"{name}.measured.json").read_text())
        else:
            summary = execute(name, plan_path, profile, samples)
        analyze(
            name,
            payload,
            profile,
            summary["measured_wall_ns_per_sample"],
            summary.get("stage_statistics"),
            summary.get("service_stats"),
            workers=payload.get("n_local_workers"),
            total_time=summary.get("total_time_sec"),
            samples=samples,
        )


if __name__ == "__main__":
    main()
