"""Trace why a planner's joint DP cannot cover the inner pipeline.

    python tmp_analysis/block_cover_probe.py <dataset_file> <kwargs> <profile> \
        [simple_dp|pico]

Prints the candidate variants, the per-operator cost rows the block provider
built, and the legal single-operator chain step by step, so an infeasible
"no feasible final state" DP run can be attributed to a specific operator or
placement instead of guessed at.
"""

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
    which = sys.argv[4] if len(sys.argv) > 4 else "simple_dp"
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

    dataset = module.get_dataset(spec)
    feature = next(iter(dataset.features.values()))
    optimizer = feature.optimizer
    if which == "simple_dp":
        from cedar.compose.simple_dp_optimizer import (
            _CedarCostBlockCandidateProvider as Provider,
        )
    else:
        from cedar.compose.dp_optimizer import BlockCandidateProvider as Provider

    if not getattr(optimizer, "profiled_stats", None):
        optimizer.profiled_stats = yaml.safe_load(open(profile))
        optimizer.options = dataset.optimizer_options
        optimizer._init_stats()

    inner = optimizer._get_linear_inner_ops()
    print("inner ops:", inner)
    def _name(pipe_id):
        pipe = optimizer.logical_pipes.get(pipe_id)
        return pipe.name if pipe is not None else f"<missing {pipe_id}>"

    print("names:", [_name(p) for p in inner])
    optimizer._prepare_dp_metadata(inner)
    provider = Provider(optimizer, inner)
    provider.prepare()

    variants = list(getattr(provider, "_candidate_variants", ()))
    print("variants:", [v.name for v in variants])
    costs = getattr(provider, "_variant_costs", None)
    if isinstance(costs, (list, tuple)):
        for vt, row in zip(variants, costs):
            print(f"  {vt.name:<10}", [round(float(c), 3) for c in row])
    else:
        for vt, row in getattr(provider, "variant_compute_costs", {}).items():
            print(f"  {vt}", [round(float(c), 3) for c in row])

    prefix = 0
    for idx in range(len(inner)):
        mask = 1 << idx
        blocks = list(provider.candidates_for_prefix(prefix, mask))
        follow = [
            b for b in blocks if optimizer._dp_valid_single_last(prefix, idx)
        ]
        print(
            f"  step {idx} ({_name(inner[idx])}): "
            f"candidates={len(blocks)} can_follow={len(follow)}"
        )
        if not follow:
            for block in blocks[:5]:
                print(
                    "     rejected:",
                    block.variant.name,
                    "cost",
                    round(float(block.cost), 3),
                    "parallelism",
                    block.parallelism,
                    "exec",
                    block.execution_resource,
                )
            print("  -> chain cover stops here")
            break
        print(
            "     chosen:",
            follow[0].variant.name,
            round(float(follow[0].cost), 3),
        )
        prefix |= mask
    return 0


if __name__ == "__main__":
    sys.exit(main())
