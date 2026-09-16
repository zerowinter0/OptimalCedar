"""Is the fixed-order incumbent's block partition reachable in the DP search?"""

import importlib
import logging
import os
import sys

import yaml

sys.path.insert(0, "/workspace/OptimalCedar")

from evaluation.cedar_utils import CedarEvalSpec  # noqa: E402

logging.basicConfig(level=logging.ERROR)


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
    optimizer._prepare_dp_metadata(inner)

    from cedar.compose.dp_optimizer import (
        BlockCandidateProvider,
        ExtensibleDpSearch,
    )
    from cedar.compose.optimizer import PipeVariantType

    provider = BlockCandidateProvider(optimizer, inner)
    provider.prepare()
    search = ExtensibleDpSearch(
        optimizer=optimizer,
        inner_ops=inner,
        block_provider=provider,
        cache_policy=__import__(
            "cedar.compose.dp_optimizer", fromlist=["CacheTransitionPolicy"]
        ).CacheTransitionPolicy(optimizer, inner),
        parallel_stage_cpu_limit=optimizer._dp_limits_for_workers(
            int(os.environ.get("PROBE_WORKERS", "32"))
        ),
    )
    n = len(inner)
    blocks = list(provider.candidates_for_prefix(0, (1 << (n - 1)) - 1))
    print(f"n={n} fused-prefix candidates={len(blocks)}")
    reachable = [
        block
        for block in blocks
        if search._block_can_follow(0, block)
    ]
    print(
        "  orders that may start the plan:",
        [block.order for block in reachable][:6],
    )
    incumbent = search._fixed_order_physical_incumbent()
    print("incumbent:", None if incumbent is None else
          (round(incumbent.cost, 4), incumbent.blocks,
           incumbent.variants_by_idx, incumbent.parallelism_by_idx))
    if incumbent is not None:
        prefix = 0
        total = 0.0
        for block_ops in incumbent.blocks:
            mask = 0
            for idx in block_ops:
                mask |= 1 << idx
            variant = incumbent.variants_by_idx.get(block_ops[0], PipeVariantType.INPROCESS)
            parallelism = incumbent.parallelism_by_idx.get(block_ops[0], 1)
            try:
                block = provider.candidate_for_order(
                    tuple(block_ops), variant, prefix_mask=prefix,
                    parallelism=parallelism,
                )
            except ValueError as exc:
                print("  block rejected by the search space:", block_ops, exc)
                prefix |= mask
                continue
            follows = search._block_can_follow(prefix, block)
            delta = optimizer._dp_regular_transition_cost(prefix, block)
            objective = search._accumulate_objective(
                type(incumbent.objective)(), delta, block, False, prefix
            )
            print(
                f"  block {block_ops} variant={variant.name} width={parallelism} "
                f"follows={follows} delta={delta:.4f} score={objective.score:.4f}"
            )
            total += objective.score
            prefix |= mask
        print(f"  recomputed total={total:.4f} vs incumbent.cost={incumbent.cost:.4f}")
    bound = search._initial_incumbent_score()
    print("mandatory local suffix costs:",
          {bin(k): round(v, 3) for k, v in
           sorted(search._mandatory_local_suffix_cost.items())})
    print(f"pruning bound (initial incumbent score) = {bound:.4f}")
    if search._greedy_full_block_result is not None:
        g = search._greedy_full_block_result
        print("greedy full-block plan:", round(g.cost, 4), g.blocks)
    full = (1 << n) - 1
    offered = [
        block
        for block in provider.candidates_for_prefix(0, full)
        if tuple(block.order) == tuple(range(n))
    ]
    print("full fused block offered by the provider:", len(offered))
    for block in offered[:4]:
        print(
            "   variant=", block.variant.name,
            "width=", block.parallelism,
            "can_follow=", search._block_can_follow(0, block),
            "cost=", round(block.cost, 3),
        )
    tail_idx = n - 1
    print(
        "  last operator can follow the prefix:",
        optimizer._dp_valid_single_last((1 << (n - 1)) - 1, tail_idx),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
