"""Report why the joint DP cannot build a block cover for a workload."""

import importlib
import logging
import os
import sys

import yaml

sys.path.insert(0, "/workspace/OptimalCedar")

from evaluation.cedar_utils import CedarEvalSpec  # noqa: E402

logging.basicConfig(level=logging.WARNING)


def main() -> int:
    dataset_file, kwargs_raw, profile = sys.argv[1:4]
    module = importlib.import_module(
        dataset_file.replace("/", ".").removesuffix(".py")
    )
    fields = {}
    for token in kwargs_raw.split(","):
        if token.strip():
            key, _, value = token.partition("=")
            fields[key.strip()] = value.strip()
    spec = CedarEvalSpec(1, None, 1, kwargs=fields, profiled_stats=profile)
    spec.use_my_optimizer = 2
    spec.disable_controller = True
    spec.disable_caching = True
    os.environ["CEDAR_MATCH_PROFILE_RESOURCES"] = "1"
    os.environ["CEDAR_PROFILE_MATCH_CPU_BUDGET"] = "64"
    os.environ["CEDAR_PROFILE_MATCH_RAY_CPU_BUDGET"] = "64"

    dataset = module.get_dataset(spec)
    feature = next(iter(dataset.features.values()))
    optimizer = feature.optimizer
    if not getattr(optimizer, "profiled_stats", None):
        optimizer.profiled_stats = yaml.safe_load(open(profile))
        optimizer.options = dataset.optimizer_options
        optimizer._init_stats()
    inner = optimizer._get_linear_inner_ops()
    print("inner ops:", inner)

    from cedar.compose.dp_optimizer import BlockCandidateProvider
    from cedar.compose.optimizer import PipeVariantType

    optimizer._prepare_dp_metadata(inner)
    provider = BlockCandidateProvider(optimizer, inner)
    provider.prepare()
    print(
        "variants:",
        [v.name for v in getattr(provider, "_candidate_variants", ())],
    )
    prefix = 0
    for idx in range(len(inner)):
        mask = 1 << idx
        blocks = list(provider.candidates_for_prefix(prefix, mask))
        valid = [
            block
            for block in blocks
            if optimizer._dp_valid_single_last(prefix, idx)
        ]
        follow = [
            block
            for block in valid
            if optimizer._dp_valid_single_last(prefix, idx)
        ]
        print(
            f"  step {idx}: candidates={len(blocks)} valid={len(valid)} "
            f"can_follow={len(follow)}"
        )
        if not follow:
            print("    -> no legal single-op block for this prefix")
            for block in valid[:3]:
                print(
                    "       rejected:",
                    block.variant.name,
                    "order",
                    block.order,
                    "exec",
                    block.execution_resource,
                )
            break
        chosen = follow[0]
        print(
            "    chosen:",
            chosen.variant.name,
            "cost",
            round(chosen.cost, 3),
            "parallelism",
            chosen.parallelism,
        )
        prefix |= mask
    return 0


if __name__ == "__main__":
    sys.exit(main())
