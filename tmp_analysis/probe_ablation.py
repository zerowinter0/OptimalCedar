"""Isolate which part of the new cost model changes Cedar's plan.

`AnchoredLegacy` keeps the new measured offload anchor (max of the worker
measurement and the Amdahl inference) but restores the historical k=1,b=0
size scaling.  Comparing it with CmOptimizer shows how much of Cedar's plan
change comes from the affine curve versus the offload anchor.
"""
import math
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
    "CEDAR_PROFILE_MATCH_RAY_BUDGET": "64",
    "CEDAR_PROFILE_MATCH_RAY_CPU_BUDGET": "64",
}.items():
    os.environ[key] = value

from cedar.compose import OptimizerOptions  # noqa: E402
from cedar.compose.affine_cost_utils import affine_value  # noqa: E402
from cedar.compose.cm_optimizer import CmOptimizer, entry  # noqa: E402
from cedar.compose.optimizer import Optimizer, PipeVariantType  # noqa: E402
from cedar.pipes.context import PipeVariantContextFactory  # noqa: E402
from cedar.sources import LocalFSSource  # noqa: E402
from evaluation.pipelines.simclrv2 import cedar_dataset as original_simclrv2  # noqa: E402
from evaluation.pipelines.target_pipeline.simclr.cedar_dataset import (  # noqa: E402
    DATASET_LOC,
    SimCLRV2Feature,
)

RUN = ROOT / "outputs/simclrv2_four_remote_20260911"
PROFILE = RUN / "profile.yaml"


class AnchoredLegacy(CmOptimizer):
    """Affine anchors, historical proportional size scaling."""

    def _calculate_pipe_cost(self, p_id, input_size, desc):
        model = entry(self._cm_models, p_id, {})
        size0 = self.profiled_stats["baseline"]["input_sizes"][p_id]
        original_reach = self._cm_baseline_reach.get(p_id, 1.0)
        reach = self._cm_reach.get(p_id, 1.0)
        if reach == 0:
            return 0.0
        if "k" in model and "b" in model:
            reference = max(1e-12, affine_value(model, size0))
        else:
            reference = self._base_cost_map[p_id] / max(original_reach, 1e-12)
        local = reference * input_size / size0 if size0 > 0 else reference
        if desc is None or desc.variant_type in (None, PipeVariantType.INPROCESS):
            return reach * local
        measurement = entry(
            self.profiled_stats.get("offloads", {}).get(
                desc.variant_type.name, {}
            ),
            p_id,
            {},
        )
        direct = measurement.get("backend_compute", {})
        mean = direct.get("mean_ms_per_sample")
        if (
            direct.get("count", 0) > 0
            and mean is not None
            and math.isfinite(mean)
            and mean >= 0
        ):
            return reach * mean * local / max(reference, 1e-12)
        old = Optimizer._calculate_pipe_cost(self, p_id, size0, desc)
        if math.isfinite(old) and old > 0:
            return (
                reach
                * old
                / max(original_reach, 1e-12)
                * local
                / max(reference, 1e-12)
            )
        return reach * local


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
        use_my_optimizer=16,
        reorder_timeout_sec=3600.0,
    )


def describe(plan):
    fused = {
        p_id: tuple(desc.fused_pipes)
        for p_id, desc in plan.pipe_descs.items()
        if getattr(desc, "fused_pipes", None) and len(desc.fused_pipes) > 1
    }
    rows = []
    for p_id in sorted(plan.pipe_descs):
        desc = plan.pipe_descs[p_id]
        width = ""
        ctx = getattr(desc, "variant_ctx", None)
        if desc.variant_type == PipeVariantType.RAY and ctx is not None:
            width = ctx.n_actors
        elif desc.variant_type == PipeVariantType.SMP and ctx is not None:
            width = ctx.n_procs
        rows.append(
            f"{p_id}:{desc.variant_type.name if desc.variant_type else None}"
            f"({width}){fused.get(p_id, '')}"
        )
    return " ".join(rows)


def main():
    for cls in (CmOptimizer, AnchoredLegacy):
        feature = build_feature()
        optimizer = cls()
        feature.set_optimizer(optimizer)
        plan = optimizer.run(str(PROFILE), options())
        print(f"===== {cls.__name__} =====", flush=True)
        print("  graph:", plan.graph)
        print("  pipes:", describe(plan))
        print("  workers:", plan.n_local_workers)
        try:
            print(
                "  cedar cost:",
                optimizer.calculate_cost(
                    plan.graph, physical_specs=plan.pipe_descs, plan=plan
                ),
            )
        except Exception as exc:  # noqa: BLE001
            print("  cedar cost failed:", exc)


if __name__ == "__main__":
    main()
