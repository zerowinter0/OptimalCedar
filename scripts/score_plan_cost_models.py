"""Score materialized plans with the three cost models we report.

  cedar : Optimizer.calculate_cost -- per-operator share of the profiled
          whole-pipeline time, Amdahl-inverted for offloaded stages, and a
          fused block charged sum(members) * fused_io/baseline_io.
          Unit: ms/source-record of ONE worker (the model has no W).
  plumber: per-stage per-core rate 1/latency, stage rate = width * rate, the
          plan's declared widths give the single-worker bottleneck X; the
          system rate is X * W, so cost = 1000 / (X * W) [ms/source-record].
  pico  : SimpleDpWorkersWidthBoundaryOptimizer.calculate_dp_objective_cost;
          its score S is W-conditioned, reported as S / W.

Usage (inside the container):
  python -u scripts/score_plan_cost_models.py \
      --workload simclrv2 \
      --profile outputs/.../simclrv2/profiles/shared.yaml \
      --plan local=outputs/.../plans/cedar_opt_local_w1.yaml \
      --plan ray=outputs/.../plans/cedar_opt_ray_w1.yaml \
      --output /tmp/costs.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from cedar.compose import OptimizerOptions  # noqa: E402
from cedar.compose.optimizer import Optimizer, PhysicalPlan  # noqa: E402
from cedar.compose.simple_dp_ablation_optimizer import (  # noqa: E402
    SimpleDpWorkersWidthBoundaryOptimizer,
)

DATASETS = {
    "simclrv2": (
        "evaluation/pipelines/simclrv2/cedar_dataset.py",
        "evaluation/datasets/imagenette2/imagenette2/train",
        4,
    ),
    "commonvoice": (
        "evaluation/pipelines/commonvoice/cedar_dataset.py",
        "datasets/commonvoice/cv15_en_train_300000",
        1,
    ),
}


def build_feature(workload: str):
    from cedar.sources import LocalFSSource
    from evaluation.eval_cedar import import_module_from_path

    module_path, data_dir, batch_size = DATASETS[workload]
    module = import_module_from_path(str((ROOT / module_path).resolve()))
    feature = module.CommonvoiceFeature(batch_size=batch_size) if workload == "commonvoice" \
        else module.SimCLRV2Feature(batch_size=batch_size)
    feature.apply(LocalFSSource(str(ROOT / data_dir), recursive=True, max_samples=64))
    return feature


def load_plan(path: Path) -> PhysicalPlan:
    payload = yaml.safe_load(path.read_text())
    payload = payload.get("physical_plan", payload)
    if "feature" in payload:
        payload = payload["feature"]
    if "feature_r0" in payload:
        payload = payload["feature_r0"]
    payload["graph"] = {int(k): v for k, v in payload["graph"].items()}
    payload["pipes"] = {int(k): v for k, v in payload["pipes"].items()}
    for desc in payload["pipes"].values():
        # Recorded campaign plans leave the variant out for operators that were
        # fused away; they are not executed, but must still deserialize.
        desc.setdefault("variant", "INPROCESS")
        desc.setdefault("variant_ctx", {"variant_type": desc["variant"]})
    return PhysicalPlan.from_dict(payload)


def plan_chain(plan: PhysicalPlan) -> str:
    """Render source -> ... -> sink with fused groups and stage widths."""
    predecessors = {child for child in plan.graph.values() for child in child}
    starts = [p_id for p_id in plan.graph if p_id not in predecessors]
    order, node = [], starts[0]
    while True:
        order.append(node)
        if not plan.graph[node]:
            break
        node = next(iter(plan.graph[node]))
    names = {}
    for p_id in order:
        desc = plan.pipe_descs[p_id]
        name = (desc.name or f"pipe{p_id}").replace("MapperPipe_", "")
        fused = list(desc.fused_pipes or [])
        if fused:
            members = ",".join(str(m) for m in fused)
            name = f"FusedPipe{{{members}}}"
        variant = (desc.variant_type or None)
        if variant is not None and variant.name not in ("INPROCESS",):
            ctx = desc.variant_ctx
            width = getattr(ctx, "n_actors", None) or getattr(ctx, "n_procs", None)
            name = f"{name}[{variant.name} w={width}]"
        names[p_id] = name
    return " -> ".join(names[p_id] for p_id in order)


def plumber_cost(plan: PhysicalPlan, profile: dict):
    """Plan-width Plumber bottleneck, converted to a per-record cost."""
    latencies = profile["baseline"]["latencies"]
    worst = 0.0
    stages = []
    for p_id, desc in plan.pipe_descs.items():
        if (desc.name or "") == "PrefetcherPipe":
            continue
        members = list(desc.fused_pipes or [p_id])
        service_ms = sum(
            float(latencies[m]) / 1e6 for m in members if m in latencies
        )
        if service_ms <= 0.0:
            continue
        ctx = desc.variant_ctx
        width = 1
        if desc.variant_type is not None and desc.variant_type.name != "INPROCESS":
            width = int(
                getattr(ctx, "n_actors", 0) or getattr(ctx, "n_procs", 0) or 1
            )
        per_record = service_ms / max(1, width)
        worst = max(worst, per_record)
        stages.append(
            {
                "pipe": p_id,
                "members": members,
                "variant": (desc.variant_type.name if desc.variant_type else "INPROCESS"),
                "width": width,
                "service_ms": round(service_ms, 4),
                "per_record_ms": round(per_record, 4),
            }
        )
    workers = max(1, int(plan.n_local_workers or 1))
    single_worker_rate = 1000.0 / worst if worst else 0.0
    cost = 1000.0 / (single_worker_rate * workers) if single_worker_rate else 0.0
    return {
        "single_worker_bottleneck_rate_rec_s": round(single_worker_rate, 3),
        "workers": workers,
        "cost_ms_per_source_record": round(cost, 4),
        "stages": stages,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", choices=sorted(DATASETS), required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--plan", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cpu-budget", type=int, default=64)
    args = parser.parse_args()

    logging.disable(logging.INFO)
    profile = yaml.safe_load(args.profile.read_text())

    # Cedar's own model.
    feature = build_feature(args.workload)
    cedar = Optimizer()
    feature.set_optimizer(cedar)
    cedar.profiled_stats = profile
    cedar._init_stats()
    cedar_unfused = 1000.0 / profile["baseline"]["throughput"]

    # PICO: the latest DP design (affine operators + boundary + workers + width).
    pico = SimpleDpWorkersWidthBoundaryOptimizer()
    feature.set_optimizer(pico)
    pico.profiled_stats = profile
    pico.options = OptimizerOptions(
        enable_prefetch=True,
        est_throughput=None,
        available_local_cpus=args.cpu_budget,
        enable_offload=True,
        enable_reorder=True,
        enable_local_parallelism=True,
        enable_fusion=True,
        num_samples=0,
        use_my_optimizer=27,
        reorder_timeout_sec=7200.0,
    )
    pico._validate_stats()
    pico._init_stats()
    inner_ops = pico._get_linear_inner_ops()
    pico._prepare_dp_metadata(inner_ops)

    rows = []
    for raw in args.plan:
        label, _, path = raw.partition("=")
        plan = load_plan(Path(path))
        workers = max(1, int(plan.n_local_workers or 1))
        cedar_cost = cedar.calculate_cost(plan.graph, plan=plan)
        try:
            pico_score = pico.calculate_dp_objective_cost(plan=plan)
            pico_info = {
                "score": round(pico_score, 4),
                "workers": workers,
                "cost_ms_per_source_record": round(pico_score / workers, 4),
            }
        except Exception as exc:  # noqa: BLE001
            # e.g. a stage whose variant is outside PICO's search space.
            pico_info = {"error": f"{type(exc).__name__}: {exc}"}
        plumber = plumber_cost(plan, profile)
        row = {
            "label": label,
            "plan_path": path,
            "workers": workers,
            "chain": plan_chain(plan),
            "cedar_cost_ms_per_source_record": round(cedar_cost, 4),
            "plumber": plumber,
            "pico": pico_info,
        }
        rows.append(row)

    print(f"unfused-local Cedar baseline: {cedar_unfused:.4f} ms/source-record")
    print(f"DP inner ops: {inner_ops}")
    header = f"{'plan':<10}{'W':>4}{'cedar':>10}{'plumber/W':>12}{'pico S':>10}{'pico/W':>10}"
    print(header)
    for row in rows:
        pico = row["pico"]
        pico_cells = (
            f"{pico['score']:>10.4f}{pico['cost_ms_per_source_record']:>10.4f}"
            if "score" in pico
            else f"{'n/a':>10}{'n/a':>10}"
        )
        print(
            f"{row['label']:<10}{row['workers']:>4}"
            f"{row['cedar_cost_ms_per_source_record']:>10.4f}"
            f"{row['plumber']['cost_ms_per_source_record']:>12.4f}"
            f"{pico_cells}"
        )
        if "error" in pico:
            print(f"    pico: {pico['error']}")
    for row in rows:
        print(f"\n{row['label']}: {row['chain']}")
        print("  plumber stages: " + json.dumps(row["plumber"]["stages"]))
    if args.output:
        args.output.write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
