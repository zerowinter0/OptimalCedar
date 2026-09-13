"""Score the saved SimCLRv2 plans under the affine and the legacy DP objective.

Read-only analysis helper: it rebuilds the SimCLR feature, replays the four
saved physical plans through both cost models, and prints per-operator cost
estimates plus the lane decomposition of the DP objective.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path("/workspace/OptimalCedar")
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

for key, value in {
    "CEDAR_MATCH_PROFILE_RESOURCES": "1",
    "CEDAR_PROFILE_MATCH_FIXED_LOCAL_WORKERS": "8",
    "CEDAR_PROFILE_MATCH_CPU_BUDGET": "64",
    "CEDAR_PROFILE_MATCH_RAY_CPU_BUDGET": "64",
}.items():
    os.environ[key] = value

import yaml  # noqa: E402

from cedar.compose import OptimizerOptions  # noqa: E402
from cedar.compose.dp_optimizer import (  # noqa: E402
    BlockCandidateProvider,
    CacheTransitionPolicy,
    DpOptimizer,
    ExtensibleDpSearch,
)
from cedar.compose.old_dp_optimizer import OldDpOptimizer  # noqa: E402
from cedar.compose.optimizer import (  # noqa: E402
    Optimizer,
    PhysicalPlan,
    PipeDesc,
    PipeVariantType,
)
from cedar.sources import LocalFSSource  # noqa: E402
from evaluation.pipelines.target_pipeline.simclr.cedar_dataset import (  # noqa: E402
    DATASET_LOC,
    SimCLRV2Feature,
)
from evaluation.pipelines.simclrv2 import cedar_dataset as original_simclrv2  # noqa: E402

OUT = Path(
    sys.argv[1]
    if len(sys.argv) > 1
    else "/workspace/OptimalCedar/outputs/simclrv2_four_remote_20260911"
)
PROFILE = Path(
    os.environ.get("CEDAR_PROBE_PROFILE") or (OUT / "profile.yaml")
)
NAMES = ["optimizer", "cm_optimizer", "old_dp_optimizer", "dp_optimizer"]


def build_feature():
    data_dir = (
        Path(original_simclrv2.__file__).resolve().parents[2].joinpath(DATASET_LOC)
    )
    train_filepath = data_dir / "imagenette2" / "train"
    feature = SimCLRV2Feature(batch_size=1)
    feature.apply(LocalFSSource(str(train_filepath), recursive=True))
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


def load_plans():
    plans = {}
    for name in NAMES:
        path = OUT / "plans" / name / "round_1.yaml"
        data = yaml.safe_load(path.read_text())["feature_r0"]
        data["graph"] = {int(k): v for k, v in data["graph"].items()}
        data["pipes"] = {int(k): v for k, v in data["pipes"].items()}
        # Pipes absorbed into a FusedPipe keep their desc without a variant.
        for pipe in data["pipes"].values():
            pipe.setdefault("variant", "INPROCESS")
        plans[name] = PhysicalPlan.from_dict(data)
    return plans


def lane_report(optimizer, plan, label):
    inner_ops = optimizer._dp_inner_ops
    blocks = optimizer._dp_blocks_from_physical_plan(plan, inner_ops)
    objective = optimizer._replay_dp_objective(blocks, inner_ops)
    pretty = [
        ([inner_ops[i] for i in order], variant.name, parallelism)
        for order, variant, _, parallelism in blocks
    ]
    print(
        f"  {label:<18} score={objective.score:8.4f} "
        f"local={objective.local_serial:8.4f} "
        f"ray={objective.ray_serial:8.4f} smp={objective.smp_serial:8.4f}"
    )
    print(f"      blocks={pretty}")
    provider = BlockCandidateProvider(optimizer, inner_ops)
    provider.prepare()
    prev_mask = 0
    for order, variant, _, parallelism in blocks:
        block = provider.candidate_for_order(
            order, variant, prefix_mask=prev_mask, parallelism=parallelism
        )
        boundary_local, boundary_parallel = (
            optimizer._dp_stage_boundary_components(prev_mask, block)
        )
        print(
            f"        {tuple(inner_ops[i] for i in order)} {variant.name:<9} "
            f"w={parallelism} compute={block.cost:8.4f} "
            f"compute/w={block.cost / parallelism:8.4f} "
            f"bnd_local={boundary_local:8.4f} bnd_par={boundary_parallel:8.4f}"
        )
        prev_mask |= block.mask
    return objective


def operator_table(optimizer, label):
    stats = optimizer.profiled_stats
    print(f"  --- per-operator cost estimate ({label}, ms/source-record) ---")
    print("   pid name                     baseline_in  INPROCESS      RAY      SMP")
    for pid in sorted(optimizer.logical_pipes):
        pipe = optimizer.logical_pipes[pid]
        baseline = stats["baseline"]["input_sizes"].get(pid)
        if baseline is None:
            continue
        row = [f"  {pid:>3} {pipe.get_logical_name():<24} {baseline:>10.1f}"]
        for vt in (None, PipeVariantType.RAY, PipeVariantType.SMP):
            desc = None
            if vt is not None:
                desc = PipeDesc(name=None, variant_type=vt, variant_ctx=None)
            try:
                value = optimizer._calculate_pipe_cost(pid, float(baseline), desc)
            except Exception:  # noqa: BLE001
                value = float("nan")
            row.append(f"{value:9.4f}")
        print(" ".join(row))


def amdahl_table(optimizer):
    """Show why Cedar's own model can price an offload as (almost) free."""
    stats = optimizer.profiled_stats
    baseline_tput = stats["baseline"]["throughput"]
    print("   pid name                    latency_ms  frac_latency   R_ray   R_smp  ray_cost  smp_cost")
    for pid in sorted(optimizer.logical_pipes):
        pipe = optimizer.logical_pipes[pid]
        baseline = stats["baseline"]["input_sizes"].get(pid)
        if baseline is None:
            continue
        fractions = getattr(optimizer, "_fractional_latencies", {})
        fraction = fractions.get(pid, float("nan"))
        raw = stats["baseline"]["latencies"].get(pid, 0.0) / 1e6
        values = []
        for vt in (PipeVariantType.RAY, PipeVariantType.SMP):
            entry = stats.get("offloads", {}).get(vt.name, {}).get(pid)
            if entry is None or not entry.get("throughput"):
                values.append((float("nan"), float("nan")))
                continue
            ratio = entry["throughput"] / baseline_tput
            try:
                desc = PipeDesc(name=None, variant_type=vt, variant_ctx=None)
                cost = optimizer._calculate_pipe_cost(pid, float(baseline), desc)
            except Exception:  # noqa: BLE001
                cost = float("nan")
            values.append((ratio, cost))
        print(
            f"  {pid:>3} {pipe.get_logical_name():<24} {raw:10.4f} {fraction:12.5f} "
            f"{values[0][0]:7.3f} {values[1][0]:7.3f} {values[0][1]:9.4f} {values[1][1]:9.4f}"
        )


def main():
    plans = load_plans()

    for cls, label in (
        (DpOptimizer, "affine"),
        (OldDpOptimizer, "legacy"),
        (Optimizer, "cedar"),
    ):
        feature = build_feature()
        optimizer = cls()
        feature.set_optimizer(optimizer)
        optimizer.run(str(PROFILE), options())
        print(f"===== {label} cost model =====")
        print(
            f"  affine_enabled="
            f"{getattr(optimizer, '_dp_affine_enabled', False)} "
            f"inner_ops={getattr(optimizer, '_dp_inner_ops', None)}"
        )
        if label == "affine":
            print(f"  reach={optimizer._dp_affine_baseline_reach}")
        print(f"  size_ratios={optimizer._data_size_ratio_map}")
        operator_table(optimizer, label)
        if label == "cedar":
            amdahl_table(optimizer)
        print(
            f"  source base cost={optimizer._base_cost_map[optimizer._get_source_p_id()]:.4f}"
        )
        if getattr(optimizer, "_dp_inner_ops", None):
            for name in NAMES:
                lane_report(optimizer, plans[name], name)
        print()

    summary = json.loads((OUT / "results/comparison.json").read_text())
    print("===== measured =====")
    for run in summary["runs"]:
        print(
            f"  {run['optimizer']:<18} perf={run['perf_time_sec']:.3f}s "
            f"tps={run['throughput_samples_per_sec']:.1f} "
            f"plan_cost={run['plan_cost']:.3f} "
            f"cedar={run['cedar_plan_cost']:.3f} "
            f"pico={run['pico_plan_cost']:.3f}"
        )


if __name__ == "__main__":
    main()
