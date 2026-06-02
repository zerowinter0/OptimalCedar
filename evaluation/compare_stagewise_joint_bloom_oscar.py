import argparse
import io
import logging
import re
import sys
from pathlib import Path
from typing import Dict, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cedar.client import DataSet
from evaluation.cedar_utils import CedarEvalSpec
from evaluation.pipelines.bloom_oscar.cedar_dataset import get_dataset


COST_RE = re.compile(
    r"\[(DpOptimizer|DpSeperateOptimizer)\] Optimized plan cost = ([0-9eE+\-.]+)"
    r"|\[(DpCedarOptimizer)\] Optimized plan cost \(calculate_cost\) = ([0-9eE+\-.]+)"
)
ORDER_RE = re.compile(
    r"\[(DpOptimizer|DpSeperateOptimizer|DpCedarOptimizer)\] "
    r"(?:Best inner order \(DP\)|Stage-1 reorder order|DP reorder order): (.+)"
)
CACHE_RE = re.compile(r"DP suggests inserting cache after pipe ([0-9]+)")
FUSION_RE = re.compile(r"Pipe ([0-9]+): .*fused=\[([^\]]+)\]")


def profile_workload(
    dataset_path: Path,
    profile_path: Path,
    num_samples: int,
    include_smp: bool,
) -> Dict:
    original_profile = DataSet._profile
    original_profile_tf = DataSet._profile_tf
    original_profile_smp = DataSet._profile_smp
    profile_path.parent.mkdir(parents=True, exist_ok=True)

    def bounded_profile(self, f_name, n_samples=None, output_file=None):
        return original_profile(
            self,
            f_name,
            n_samples=num_samples,
            output_file=output_file,
        )

    def skip_tf_profile(self, d, feature_to_profile, f_name, n_samples):
        return None

    def maybe_profile_smp(self, d, feature_to_profile, f_name, n_samples):
        if include_smp:
            return original_profile_smp(self, d, feature_to_profile, f_name, n_samples)
        d.setdefault("offloads", {})["SMP"] = {}
        return None

    DataSet._profile = bounded_profile
    DataSet._profile_tf = skip_tf_profile
    DataSet._profile_smp = maybe_profile_smp
    spec = CedarEvalSpec(
        batch_size=1,
        num_total_samples=num_samples,
        num_epochs=1,
        kwargs={"dataset_path": str(dataset_path)},
        use_ray=True,
        profiled_stats=str(profile_path),
        run_profiling=True,
        disable_controller=True,
        disable_optimizer=True,
        disable_prefetch=True,
        disable_offload=False,
        disable_parallelism=True,
        disable_reorder=False,
        disable_fusion=False,
        disable_caching=False,
    )
    try:
        get_dataset(spec)
    except SystemExit as exc:
        if exc.code not in (0, None):
            raise
    finally:
        DataSet._profile = original_profile
        DataSet._profile_tf = original_profile_tf
        DataSet._profile_smp = original_profile_smp

    with profile_path.open("r", encoding="utf-8") as f:
        stats: Dict = yaml.safe_load(f)
    return stats


def _run_optimizer(
    dataset_path: Path,
    profile_path: Path,
    optimizer_selector: int,
) -> str:
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
        use_my_optimizer=optimizer_selector,
        generate_plan=True,
    )

    compose_logger = logging.getLogger("cedar.compose")
    old_level = compose_logger.level
    old_handlers = list(compose_logger.handlers)
    old_propagate = compose_logger.propagate

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    compose_logger.handlers = [handler]
    compose_logger.setLevel(logging.INFO)
    compose_logger.propagate = False
    try:
        get_dataset(spec)
    except SystemExit as exc:
        if exc.code not in (0, None):
            raise
        return stream.getvalue()
    else:
        return stream.getvalue()
    finally:
        compose_logger.handlers = old_handlers
        compose_logger.setLevel(old_level)
        compose_logger.propagate = old_propagate


def _extract_cost(log_text: str, optimizer_name: str) -> float:
    matches = []
    for dp_name, dp_cost, cedar_name, cedar_cost in COST_RE.findall(log_text):
        matches.append((dp_name or cedar_name, dp_cost or cedar_cost))
    for name, cost in reversed(matches):
        if name == optimizer_name:
            return float(cost)
    raise RuntimeError(f"Missing cost line for {optimizer_name}.")


def _extract_order(log_text: str, optimizer_name: str) -> str:
    matches = [value for name, value in ORDER_RE.findall(log_text) if name == optimizer_name]
    return matches[-1] if matches else "-"


def _extract_cache(log_text: str) -> str:
    matches = CACHE_RE.findall(log_text)
    return matches[-1] if matches else "-"


def _extract_fusion(log_text: str) -> str:
    blocks = []
    for pipe_id, fused in FUSION_RE.findall(log_text):
        members = ",".join(part.strip() for part in fused.split(","))
        if "," in members:
            blocks.append(f"{pipe_id}=[{members}]")
    return ";".join(blocks) if blocks else "-"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Profile the real BLOOM/OSCAR Cedar workload, then compare "
            "stage-wise DP against joint DP on the generated profile."
        )
    )
    parser.add_argument(
        "--dataset_path",
        default="/tmp/redpajama_backup_3gb_for_bloom_oscar.jsonl",
    )
    parser.add_argument(
        "--profile_path",
        default="/tmp/bloom_oscar_stagewise_joint_profile.yml",
    )
    parser.add_argument("--num_profile_samples", type=int, default=200)
    parser.add_argument("--skip_profile", action="store_true")
    parser.add_argument(
        "--disable_smp_profile",
        action="store_true",
        help="Debug only: omit SMP profiling. Do not use for paper results.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    dataset_path = Path(args.dataset_path)
    profile_path = Path(args.profile_path)
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    if not args.skip_profile or not profile_path.exists():
        profile_workload(
            dataset_path,
            profile_path,
            args.num_profile_samples,
            include_smp=not args.disable_smp_profile,
        )

    cedar_staged_log = _run_optimizer(dataset_path, profile_path, optimizer_selector=5)
    stagewise_log = _run_optimizer(dataset_path, profile_path, optimizer_selector=4)
    joint_log = _run_optimizer(dataset_path, profile_path, optimizer_selector=2)

    cedar_staged_cost = _extract_cost(cedar_staged_log, "DpCedarOptimizer")
    stagewise_cost = _extract_cost(stagewise_log, "DpSeperateOptimizer")
    joint_cost = _extract_cost(joint_log, "DpOptimizer")
    cedar_improvement = (cedar_staged_cost - joint_cost) / cedar_staged_cost * 100.0
    separate_improvement = (stagewise_cost - joint_cost) / stagewise_cost * 100.0

    with profile_path.open("r", encoding="utf-8") as f:
        stats = yaml.safe_load(f)
    offload_counts = {
        backend: len(entries)
        for backend, entries in stats.get("offloads", {}).items()
    }

    print("workload: bloom_oscar")
    print(f"dataset_path: {dataset_path}")
    print(f"profile_path: {profile_path}")
    print(f"profiled_offload_variants: {offload_counts}")
    print(f"cedar_staged_dp_reorder_cost: {cedar_staged_cost:.12f}")
    print(f"stagewise_dp_cost: {stagewise_cost:.12f}")
    print(f"joint_cost: {joint_cost:.12f}")
    print(f"joint_vs_cedar_staged_improvement_pct: {cedar_improvement:.2f}")
    print(f"joint_vs_stagewise_dp_improvement_pct: {separate_improvement:.2f}")
    print(f"cedar_staged_order: {_extract_order(cedar_staged_log, 'DpCedarOptimizer')}")
    print(f"stagewise_order: {_extract_order(stagewise_log, 'DpSeperateOptimizer')}")
    print(f"joint_order: {_extract_order(joint_log, 'DpOptimizer')}")
    print(f"cedar_staged_cache_after: {_extract_cache(cedar_staged_log)}")
    print(f"stagewise_cache_after: {_extract_cache(stagewise_log)}")
    print(f"joint_cache_after: {_extract_cache(joint_log)}")
    print(f"cedar_staged_fusion: {_extract_fusion(cedar_staged_log)}")
    print(f"stagewise_fusion: {_extract_fusion(stagewise_log)}")
    print(f"joint_fusion: {_extract_fusion(joint_log)}")


if __name__ == "__main__":
    main()
