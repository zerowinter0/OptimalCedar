"""Time PICO planning on the hard workloads under different complexity knobs.

Usage (inside the container), e.g.
  python -u tmp_analysis/probe_pico_complexity.py --workload llava_pretrain \
      --profile outputs/llava_profile_check_20260919/llava_pretrain/profiles/shared.yaml \
      --mode chain --deadline 600
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path("/workspace/OptimalCedar")
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from cedar.compose import OptimizerOptions  # noqa: E402
from cedar.compose.simple_dp_ablation_optimizer import (  # noqa: E402
    SimpleDpWorkersWidthBoundaryOptimizer,
)
from cedar.sources import LocalFSSource  # noqa: E402
from evaluation.eval_cedar import import_module_from_path  # noqa: E402

WORKLOADS = {
    "llava_pretrain": (
        "evaluation/pipelines/llava_pretrain/cedar_dataset.py",
        {"dataset_path": str(ROOT / "outputs/ultimate_eight_optimizers_fix_20260921/inputs/llava_pretrain.jsonl"),
         "image_root": str(ROOT / "evaluation/datasets/llava_pretrain")},
        "LlavaPretrainFeature",
    ),
    "stackexchange": (
        "evaluation/pipelines/stackexchange/cedar_dataset.py",
        {"dataset_path": str(ROOT / "outputs/ultimate_eight_optimizers_fix_20260921/inputs/stackexchange.jsonl")},
        None,
    ),
}


def build_feature(workload: str, profile_path: Path):
    module_path, kwargs, cls_name = WORKLOADS[workload]
    module = import_module_from_path(str((ROOT / module_path).resolve()))
    if hasattr(module, "get_dataset"):
        from evaluation.cedar_utils import CedarEvalSpec

        spec = CedarEvalSpec(
            batch_size=4,
            num_total_samples=0,
            num_epochs=0,
            config=None,
            kwargs=kwargs,
            use_ray=True,
            ray_ip="172.23.166.105:6379",
            profiled_stats=str(profile_path),
            disable_optimizer=True,
            disable_controller=True,
            disable_caching=True,
        )
        dataset = module.get_dataset(spec)
        return next(iter(dataset.features.values()))
    raise SystemExit(f"no get_dataset in {module_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", choices=sorted(WORKLOADS), required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--mode", default="general")
    parser.add_argument("--deadline", type=float, default=600.0)
    parser.add_argument("--worker-set", default=None)
    parser.add_argument("--width-ladder", default=None)
    parser.add_argument("--cpus", type=int, default=64)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    else:
        logging.disable(logging.INFO)
    os.environ["CEDAR_MATCH_PROFILE_RESOURCES"] = "1"
    os.environ["CEDAR_PROFILE_MATCH_CPU_BUDGET"] = str(args.cpus)
    os.environ["CEDAR_PROFILE_MATCH_RAY_CPU_BUDGET"] = str(args.cpus)
    os.environ["CEDAR_DP_SEARCH_MODE"] = args.mode
    if args.worker_set:
        os.environ["CEDAR_WORKER_SEARCH_SET"] = args.worker_set
    if args.width_ladder:
        os.environ["CEDAR_DP_WIDTH_LADDER"] = args.width_ladder

    profile = yaml.safe_load(args.profile.read_text())
    feature = build_feature(args.workload, args.profile)
    optimizer = SimpleDpWorkersWidthBoundaryOptimizer()
    feature.set_optimizer(optimizer)
    options = OptimizerOptions(
        enable_prefetch=True,
        est_throughput=None,
        available_local_cpus=args.cpus,
        enable_offload=True,
        enable_reorder=True,
        enable_local_parallelism=True,
        enable_fusion=True,
        num_samples=0,
        use_my_optimizer=27,
        reorder_timeout_sec=args.deadline,
    )
    started = time.monotonic()
    plan = optimizer.run(profile, options)
    elapsed = time.monotonic() - started
    cost = optimizer.calculate_dp_objective_cost(plan=plan)
    print(
        f"workload={args.workload} mode={args.mode} "
        f"worker_set={args.worker_set or 'default'} "
        f"deadline={args.deadline:.0f}s -> elapsed={elapsed:.1f}s "
        f"cost={cost:.6f} W={plan.n_local_workers}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
