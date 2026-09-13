"""Attribute the new profile's effect to width scaling vs object boundary.

For every profile variant the DP optimizers are run (plan generation only) and
the chosen blocks, widths and DP objective coordinates are printed.
"""
import copy
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
from cedar.compose.dp_optimizer import DpOptimizer  # noqa: E402
from cedar.compose.old_dp_optimizer import OldDpOptimizer  # noqa: E402
from cedar.sources import LocalFSSource  # noqa: E402
from evaluation.pipelines.simclrv2 import cedar_dataset as original_simclrv2  # noqa: E402
from evaluation.pipelines.target_pipeline.simclr.cedar_dataset import (  # noqa: E402
    DATASET_LOC,
    SimCLRV2Feature,
)

PROFILES = ROOT / "tmp_analysis/profiles"
NEW = ROOT / "outputs/simclrv2_scaling_20260911/profile.yaml"
OLD = ROOT / "outputs/simclrv2_four_remote_20260911/profile.yaml"


def write_variants():
    PROFILES.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load(NEW.read_text())
    variants = {}
    for tag, drop in (
        ("full", ()),
        ("no_scaling", ("scaling",)),
        ("no_object_boundary", ("object_boundary",)),
        ("neither", ("scaling", "object_boundary")),
    ):
        profile = copy.deepcopy(base)
        for section in drop:
            profile["physical_model"].pop(section, None)
        path = PROFILES / f"{tag}.yaml"
        path.write_text(yaml.safe_dump(profile))
        variants[tag] = path
    variants["reference"] = OLD
    variants["calibrated"] = Path(
        "/workspace/OptimalCedar/outputs/simclrv2_scaling_20260911/profile_calibrated.yaml"
    )
    return variants


def build_feature():
    data_dir = (
        Path(original_simclrv2.__file__).resolve().parents[2].joinpath(DATASET_LOC)
    )
    feature = SimCLRV2Feature(batch_size=1)
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


def describe(optimizer, plan):
    from cedar.compose.dp_optimizer import (
        BlockCandidateProvider,
        CacheTransitionPolicy,
        ExtensibleDpSearch,
    )

    inner_ops = optimizer._dp_inner_ops
    blocks = optimizer._dp_blocks_from_physical_plan(plan, inner_ops)
    objective = optimizer._replay_dp_objective(blocks, inner_ops)
    pretty = [
        (tuple(inner_ops[i] for i in order), variant.name, parallelism)
        for order, variant, _, parallelism in blocks
    ]
    return objective, pretty


def main():
    variants = write_variants()
    rows = []
    for tag, path in variants.items():
        for cls in (DpOptimizer, OldDpOptimizer):
            feature = build_feature()
            optimizer = cls()
            feature.set_optimizer(optimizer)
            plan = optimizer.run(str(path), options())
            objective, blocks = describe(optimizer, plan)
            row = {
                "profile": tag,
                "optimizer": cls.__name__,
                "score": objective.score,
                "local": objective.local_serial,
                "ray": objective.ray_serial,
                "smp": objective.smp_serial,
                "blocks": blocks,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
            if cls is DpOptimizer:
                out = PROFILES.parent / "plans" / f"abl_{tag}_dp.yaml"
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(
                    yaml.safe_dump({"physical_plan": plan.to_dict()})
                )
    (PROFILES.parent / "ablation.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
