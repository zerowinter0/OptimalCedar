"""Why old-dp-opt splits the RAY run 6-5-4-3 into [6,5] and [4,3].

Cedar's scalar objective (which SimpleDpOptimizer minimizes exactly) charges a
fused block ``sum(member costs) * fused_io/baseline_io``.  The IO ratio is not
monotone in the block length, and the RAY cost of some members saturates to 0
by Amdahl's law, so the search has an incentive to cut a long RAY run.

This script prints the per-operator local/RAY costs, the block ratios and the
total objective for the produced plan and for merged alternatives.

Usage (inside the container):
  python -u tmp_analysis/why_old_dp_commonvoice_fusion.py
"""

import logging
import os
import sys
from pathlib import Path

ROOT = Path("/workspace/OptimalCedar")
sys.path.insert(0, str(ROOT))

os.environ.setdefault("CEDAR_MATCH_PROFILE_RESOURCES", "1")
os.environ.setdefault("CEDAR_PROFILE_MATCH_CPU_BUDGET", "64")
os.environ.setdefault("CEDAR_PROFILE_MATCH_RAY_CPU_BUDGET", "64")

import yaml  # noqa: E402

from cedar.compose import OptimizerOptions  # noqa: E402
from cedar.compose.optimizer import PhysicalPlan, PipeDesc  # noqa: E402
from cedar.compose.simple_dp_optimizer import SimpleDpOptimizer  # noqa: E402
from cedar.sources import LocalFSSource  # noqa: E402
from evaluation.pipelines.commonvoice.cedar_dataset import (  # noqa: E402
    CommonvoiceFeature,
)

RUN = ROOT / "outputs/ultimate_eight_optimizers_20260920/commonvoice"
PROFILE = RUN / "profiles/shared.yaml"
PRODUCED = RUN / "plans/round1__old_dp_legacy_optimizer.yaml"
DATA_DIR = ROOT / "datasets/commonvoice/cv15_en_train_300000"
NAMES = {
    0: "mel",
    1: "frequency_mask",
    2: "time_mask",
    3: "_stretch",
    4: "_spec",
    5: "_resample",
    6: "_read",
    7: "LocalFSListerPipe",
}


def new_optimizer():
    feature = CommonvoiceFeature(batch_size=1)
    feature.apply(LocalFSSource(str(DATA_DIR), recursive=True, max_samples=64))
    optimizer = SimpleDpOptimizer()
    feature.set_optimizer(optimizer)
    optimizer.profiled_stats = yaml.safe_load(PROFILE.read_text())
    optimizer.options = OptimizerOptions(
        enable_prefetch=True,
        est_throughput=None,
        available_local_cpus=64,
        enable_offload=True,
        enable_reorder=True,
        enable_local_parallelism=True,
        enable_fusion=True,
        num_samples=300000,
        use_my_optimizer=11,
        reorder_timeout_sec=7200.0,
    )
    optimizer._validate_stats()
    optimizer._init_stats()
    return feature, optimizer


def ray_ctx(template):
    for rec in template["pipes"].values():
        if rec.get("variant") == "RAY":
            return dict(rec["variant_ctx"])
    raise SystemExit("no RAY stage in the produced plan")


def build_plan(template, groups):
    """groups = [([pids], 'RAY'|'INPROCESS'), ...] in chain order."""
    pipes = {
        p_id: dict(rec)
        for p_id, rec in template["pipes"].items()
        if p_id < 8
    }
    graph = {}
    previous, node = 7, 8
    for members, variant in groups:
        ctx = ray_ctx(template) if variant == "RAY" else {"variant_type": "INPROCESS"}
        pipes[node] = {
            "name": "FusedPipe",
            "fused_pipes": list(members),
            "variant": variant,
            "variant_ctx": ctx,
        }
        graph[previous] = str(node)
        previous, node = node, node + 1
    pipes[node] = {"name": "PrefetcherPipe", "variant": "INPROCESS",
                   "variant_ctx": {"variant_type": "INPROCESS"}}
    graph[previous] = str(node)
    graph[node] = ""
    return {"graph": graph, "pipes": pipes,
            "n_local_workers": template.get("n_local_workers", 21)}


def main() -> int:
    logging.disable(logging.INFO)
    _, optimizer = new_optimizer()
    raw = yaml.safe_load(PRODUCED.read_text())
    template = raw["feature_r0"]
    template["graph"] = {int(k): v for k, v in template["graph"].items()}
    template["pipes"] = {int(k): v for k, v in template["pipes"].items()}
    base = optimizer.profiled_stats["baseline"]

    print("per-operator cost at the profiled chain sizes (ms/source-record)")
    print(f"{'pipe':<18}{'local':>10}{'RAY':>10}   amortised by Amdahl")
    per_pipe = {}
    for p_id in sorted(NAMES):
        if p_id > 6:
            continue
        size = float(base["input_sizes"][p_id])
        local = optimizer._calculate_pipe_cost(p_id, size, None)
        ray = optimizer._calculate_pipe_cost(
            p_id,
            size,
            PipeDesc(name=None, variant_type=optimizer._iter_candidate_backend_stats().__next__()[0], variant_ctx=None),
        )
        per_pipe[p_id] = (local, ray)
        tag = "saturated -> 0" if ray == 0 else ("kept" if ray < local else "worse than local")
        print(f"{NAMES[p_id]:<18}{local:>10.4f}{ray:>10.4f}   {tag}")

    print("\nfused-block accounting (RAY unless noted), ms/source-record")
    print(f"{'block':<22}{'sum members':>13}{'io ratio':>10}{'charged':>10}")

    # Cedar's IO ratio (the same formula as _cedar_fusion_io_ratio), built from
    # the profiled chain sizes: the block input is 1, every later member adds
    # 2x its own input to the unfused baseline IO, and the fused IO is just the
    # block's input plus its output.
    def cedar_io_ratio(members):
        local_ratio = 1.0
        baseline_io = 1.0
        for position, p_id in enumerate(members):
            if position > 0:
                baseline_io += 2.0 * local_ratio
            local_ratio *= (
                float(base["output_sizes"][p_id]) / float(base["input_sizes"][p_id])
            )
        baseline_io += local_ratio
        return (1.0 + local_ratio) / baseline_io

    for members, variant in (
        ([6], "RAY"),
        ([6, 5], "RAY"),
        ([6, 5, 4], "RAY"),
        ([6, 5, 4, 3], "RAY"),
        ([4, 3], "RAY"),
        ([4], "RAY"),
        ([3], "RAY"),
        ([2, 1], "INPROCESS"),
        ([2, 1, 0], "INPROCESS"),
        ([0], "RAY"),
        ([6, 5, 4, 3, 2, 1, 0], "RAY"),
    ):
        ratio = cedar_io_ratio(members)
        column = 1 if variant == "RAY" else 0
        total = sum(per_pipe[p][column] for p in members)
        charged = total * ratio if len(members) > 1 else total
        label = variant if len(members) > 1 else "single"
        print(f"{str(members):<22}{total:>13.4f}{ratio:>10.4f}{charged:>10.4f}  "
              f"({'x'.join(str(p) for p in members)} {label})")

    print("\ntotal Cedar objective for candidate plans")
    plans = {
        "produced  [6,5]R [4,3]R [2,1]L [0]R": [
            ([6, 5], "RAY"), ([4, 3], "RAY"), ([2, 1], "INPROCESS"), ([0], "RAY")
        ],
        "merge 6543: [6,5,4,3]R [2,1]L [0]R": [
            ([6, 5, 4, 3], "RAY"), ([2, 1], "INPROCESS"), ([0], "RAY")
        ],
        "merge everything RAY except [2,1]L": [
            ([6, 5, 4, 3, 0], "RAY"), ([2, 1], "INPROCESS")
        ],
        "all RAY [6..0]": [([6, 5, 4, 3, 2, 1, 0], "RAY")],
        "all local [6..0]": [([6, 5, 4, 3, 2, 1, 0], "INPROCESS")],
    }
    for label, groups in plans.items():
        plan = PhysicalPlan.from_dict(build_plan(template, groups))
        cost = optimizer._calculate_materialized_cedar_cost(plan)
        print(f"  {label:<40} {cost:>8.4f} ms/source-record")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
