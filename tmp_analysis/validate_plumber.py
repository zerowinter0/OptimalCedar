"""Validate the Plumber-style optimizer on the native SimCLR feature."""
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

from cedar.compose import OptimizerOptions  # noqa: E402
from cedar.compose.plumber_optimizer import PlumberOptimizer  # noqa: E402
from cedar.sources import LocalFSSource  # noqa: E402
from evaluation.pipelines.simclrv2 import cedar_dataset as original_simclrv2  # noqa: E402
from evaluation.pipelines.target_pipeline.simclr.cedar_dataset import (  # noqa: E402
    SimCLRV2Feature,
)

PROFILE = ROOT / "outputs/simclrv2_scaling_20260911/profile.yaml"
DATASET_LOC = "datasets/imagenette2"


def main():
    data_dir = (
        Path(original_simclrv2.__file__).resolve().parents[2].joinpath(DATASET_LOC)
    )
    feature = SimCLRV2Feature(batch_size=1)
    feature.apply(LocalFSSource(str(data_dir / "imagenette2" / "train"), recursive=True))
    feature.set_optimizer(PlumberOptimizer())
    options = OptimizerOptions(
        enable_prefetch=True,
        est_throughput=None,
        available_local_cpus=64,
        enable_offload=True,
        enable_reorder=True,
        enable_local_parallelism=True,
        enable_fusion=True,
        num_samples=9469,
        use_my_optimizer=18,
        reorder_timeout_sec=None,
    )
    plan = feature.optimize(options, str(PROFILE))
    print("local workers:", plan.n_local_workers)
    for p_id in sorted(plan.pipe_descs):
        desc = plan.pipe_descs[p_id]
        ctx = desc.variant_ctx
        print(
            f"  pipe {p_id}: {desc.name} variant={getattr(desc.variant_type, 'name', None)}"
            f" n_procs={getattr(ctx, 'n_procs', None)}"
        )


if __name__ == "__main__":
    main()
