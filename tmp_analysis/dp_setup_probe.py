"""Run one optimizer's setup on a workload and print the real traceback.

    python tmp_analysis/dp_setup_probe.py <dataset_file> <dataset_kwargs> <profile> [optimizer]
"""

import importlib
import json
import logging
import sys
import traceback

sys.path.insert(0, "/workspace/OptimalCedar")

from evaluation.cedar_utils import CedarEvalSpec  # noqa: E402

logging.basicConfig(level=logging.WARNING)


def main() -> int:
    dataset_file, kwargs_raw, profile = sys.argv[1:4]
    optimizer_name = sys.argv[4] if len(sys.argv) > 4 else "dp_optimizer"
    module = importlib.import_module(dataset_file.replace("/", ".").removesuffix(".py"))
    kwargs = {}
    for token in kwargs_raw.split(","):
        if not token.strip():
            continue
        key, _, value = token.partition("=")
        kwargs[key.strip()] = value.strip()
    spec = CedarEvalSpec(1, None, 1, profiled_stats=profile)
    spec.kwargs = kwargs
    spec.run_profiling = False
    spec.generate_plan = False
    spec.disable_controller = True
    spec.disable_caching = True
    from cedar.compose import dp_optimizer as dp_module

    original = getattr(dp_module.DpOptimizer, "run", None)
    try:
        dataset = module.get_dataset(spec)
        print("dataset ready", flush=True)
        optimizer_cls = getattr(dp_module, "DpOptimizer")
        if optimizer_name == "simple_dp_optimizer":
            from cedar.compose import simple_dp_optimizer as sdp

            optimizer_cls = sdp.SimpleDpOptimizer
        # The dataset already built the optimizer; re-run its planning step so a
        # failure surfaces with a stack trace instead of the harness' summary.
        feature = next(iter(dataset.features.values()))
        options = dataset.optimizer_options
        plan = feature.optimize(options, profile)
        print("plan ops:", len(getattr(plan, "graph", {}) or {}), flush=True)
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
