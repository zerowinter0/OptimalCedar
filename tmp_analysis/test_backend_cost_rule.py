"""Acceptance test for the un-truncated backend compute cost.

Builds a synthetic profile in which one backend is *slower* than the local
pipeline for one operator and *faster* for another, then checks that the
representation-aware optimizer prices exactly the measured values (the previous
``min(mean, local)`` rule would have reported the local cost for the slow one)
and that a missing backend measurement is reported as unpriceable instead of
falling back to the local anchor.

Usage (inside the container):
  python -u tmp_analysis/test_backend_cost_rule.py
"""

import copy
import json
import sys
from pathlib import Path

import yaml

ROOT = Path("/workspace/OptimalCedar")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from block_mechanism_common import build_feature  # noqa: E402
from cedar.compose import OptimizerOptions  # noqa: E402
from cedar.compose.optimizer import PipeDesc, PipeVariantType  # noqa: E402
from cedar.compose.simple_dp_ablation_optimizer import (  # noqa: E402
    SimpleDpWorkersBoundaryAffineReprOptimizer,
)

PROFILE = ROOT / "outputs/affine_repr_profile_20260924/simclrv2/shared.yaml"


def main() -> int:
    profile = yaml.safe_load(PROFILE.read_text())
    feature = build_feature(batch_size=4)
    optimizer = SimpleDpWorkersBoundaryAffineReprOptimizer()
    feature.set_optimizer(optimizer)
    optimizer.profiled_stats = copy.deepcopy(profile)
    optimizer.options = OptimizerOptions(
        enable_prefetch=True,
        est_throughput=None,
        available_local_cpus=64,
        enable_offload=True,
        enable_reorder=True,
        enable_local_parallelism=True,
        enable_fusion=True,
        enable_caching=False,
        num_samples=0,
        use_my_optimizer=39,
        reorder_timeout_sec=600.0,
    )
    optimizer._validate_stats()
    optimizer._init_stats()
    optimizer._prepare_dp_metadata(optimizer._get_linear_inner_ops())

    # Two operators: one backend slower than local, one faster.
    slow_id, fast_id = 2, 4
    for p_id, factor in ((slow_id, 3.0), (fast_id, 0.25)):
        local = optimizer._repr_compute_cost(
            p_id, *optimizer._repr_declared_features(p_id)
        )
        entry = optimizer.profiled_stats["offloads"]["SMP"][p_id]
        entry["backend_compute"] = {
            "mean_ms_per_sample": local * factor,
            "stderr_ms_per_sample": 0.0,
            "count": 32,
        }
    desc = PipeDesc(name=None, variant_type=PipeVariantType.SMP, variant_ctx=None)
    results = {}
    for p_id, factor in ((slow_id, 3.0), (fast_id, 0.25)):
        base = optimizer.profiled_stats["baseline"]["input_sizes"][p_id]
        measured = float(
            optimizer.profiled_stats["offloads"]["SMP"][p_id][
                "backend_compute"
            ]["mean_ms_per_sample"]
        )
        priced = optimizer._calculate_pipe_cost(p_id, base, desc)
        results[p_id] = {
            "factor": factor,
            "local_ms": optimizer._repr_compute_cost(
                p_id, *optimizer._repr_declared_features(p_id)
            ),
            "measured_backend_ms": measured,
            "priced_ms": priced,
            "clamped": priced < measured - 1e-12,
        }
    # A backend without a measurement must be reported, not charged locally.
    del optimizer.profiled_stats["offloads"]["RAY"][slow_id]["backend_compute"]
    try:
        optimizer._calculate_pipe_cost(
            slow_id,
            optimizer.profiled_stats["baseline"]["input_sizes"][slow_id],
            PipeDesc(name=None, variant_type=PipeVariantType.RAY, variant_ctx=None),
        )
        unpriceable = "no exception (BAD)"
    except RuntimeError as exc:
        unpriceable = str(exc)[:120]
    report = {
        "slower_backend": results[slow_id],
        "faster_backend": results[fast_id],
        "missing_backend_measurement": unpriceable,
        "unpriced_backends": optimizer.repr_unpriced_backends(),
        "passed": (
            not results[slow_id]["clamped"]
            and not results[fast_id]["clamped"]
            and abs(results[slow_id]["priced_ms"] - results[slow_id]["measured_backend_ms"]) < 1e-9
            and abs(results[fast_id]["priced_ms"] - results[fast_id]["measured_backend_ms"]) < 1e-9
            and "no exception" not in unpriceable
        ),
    }
    print(json.dumps(report, indent=1))
    out = ROOT / "outputs/pico_final_w_only_20260924/backend_cost_rule.json"
    out.write_text(json.dumps(report, indent=1))
    print("PASS" if report["passed"] else "FAIL")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
