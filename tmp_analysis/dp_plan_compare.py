"""Plan one workload twice (affine on/off) and print the chosen plan summary."""

import importlib
import logging
import os
import sys

sys.path.insert(0, "/workspace/OptimalCedar")

from evaluation.cedar_utils import CedarEvalSpec  # noqa: E402

logging.basicConfig(level=logging.WARNING)


def plan_once(dataset_file, kwargs_raw, profile, affine):
    os.environ["CEDAR_DP_OPERATOR_AFFINE"] = "1" if affine else "0"
    module = importlib.import_module(
        dataset_file.replace("/", ".").removesuffix(".py")
    )
    fields = {}
    for token in kwargs_raw.split(","):
        if token.strip():
            key, _, value = token.partition("=")
            fields[key.strip()] = value.strip()
    spec = CedarEvalSpec(1, None, 1, kwargs=fields, profiled_stats=profile)
    spec.disable_controller = True
    spec.disable_caching = True
    dataset = module.get_dataset(spec)
    plan = next(iter(dataset.feature_plans.values()))
    stages = []
    for pid, desc in sorted(plan.pipe_descs.items(), key=lambda kv: int(kv[0])):
        variant = desc.variant_type.name if desc.variant_type else "?"
        ctx = desc.variant_ctx
        width = getattr(ctx, "n_actors", None) or getattr(ctx, "n_procs", None)
        if desc.fused_pipes or variant not in ("INPROCESS",):
            stages.append((desc.name or "", variant, width, tuple(desc.fused_pipes or ())))
    import yaml
    from cedar.compose.dp_optimizer import DpOptimizer

    feature = next(iter(dataset.features.values()))
    scorer = DpOptimizer()
    scorer.init(feature.logical_pipes, feature.logical_adj_list)
    scorer.profiled_stats = yaml.safe_load(open(profile))
    scorer.options = dataset.optimizer_options
    scorer._init_stats()
    inner = scorer._get_linear_inner_ops()
    scorer._prepare_dp_metadata(inner)
    cost = scorer.calculate_dp_objective_cost(plan=plan)
    return plan.n_local_workers, cost, stages


def main() -> int:
    dataset_file, kwargs_raw, profile = sys.argv[1:4]
    for affine in (False, True):
        workers, cost, stages = plan_once(dataset_file, kwargs_raw, profile, affine)
        print(f"affine={affine} workers={workers} objective={cost:.2f}")
        for name, variant, width, fused in stages:
            print(f"    {name[:26]:<28} {variant:<9} width={width} fused={fused}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
