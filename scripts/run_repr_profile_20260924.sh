#!/usr/bin/env bash
# Regenerate the SimCLRv2 shared profile (legacy + layered + representation
# aware compute model) against the remote Ray cluster.
#
# The campaign runners ship a *snapshot* of the source to the Ray workers via
# ``runtime_env.working_dir``; profiling must use the same mechanism, otherwise
# the remote actors import a stale tree (the failure mode is
# ``No module named 'pipelines.target_pipeline'``).
#
# Usage (inside the container):
#   bash scripts/run_repr_profile_20260924.sh [out_dir]
set -euo pipefail
cd /workspace/OptimalCedar
source env/bin/activate

OUT=${1:-outputs/affine_repr_profile_20260924}
SNAPSHOT=${OUT}/modules
mkdir -p "$OUT"
rm -rf "$SNAPSHOT"
mkdir -p "$SNAPSHOT"
# Only source is shipped to the Ray workers: evaluation/ also holds the 100 GB
# dataset tree, so copy the module files and the pipeline packages explicitly.
cp -r cedar "$SNAPSHOT"/cedar
mkdir -p "$SNAPSHOT/evaluation"
find evaluation -maxdepth 1 -type f -name '*.py' -exec cp {} "$SNAPSHOT/evaluation/" \;
cp -r evaluation/pipelines "$SNAPSHOT/evaluation/pipelines"
find "$SNAPSHOT" -name '__pycache__' -type d -prune -exec rm -rf {} +
cp outputs/ultimate_eight_optimizers_fix_20260921/entry.py "$OUT/entry.py"

export CEDAR_RAY_PLACEMENT_RESOURCE=cedar_remote
export CEDAR_RAY_REQUIRE_REMOTE=1
export CEDAR_PROFILE_RAY_ACTORS=1
export CEDAR_PROFILE_SMP_PROCS=1
export CEDAR_PROFILE_TIME_SEC=10
export CEDAR_PROFILE_BOUNDARY_MODEL=1
export CEDAR_REUSE_BOUNDARY_MODEL=0
export CEDAR_PROFILE_INFER_COMPUTE_SCALING=1
export CEDAR_LAYERED_ADAPTIVE_PROFILE=1
export CEDAR_ADAPTIVE_PROFILE_MIN_SEC=3
export CEDAR_ADAPTIVE_PROFILE_MAX_SEC=30
export CEDAR_ADAPTIVE_PROFILE_TARGET_RSE=0.10
export CEDAR_ADAPTIVE_PROFILE_MIN_OBS=30
export CEDAR_PROFILE_POOL_SAMPLES=64
export CEDAR_PROFILE_POOL_BYTES_PER_PIPE=$((64 * 1024 * 1024))
export CEDAR_PROFILE_POOL_BYTES_TOTAL=$((512 * 1024 * 1024))
export CEDAR_PROFILE_SCALING_WIDTHS=1,2,4,8
export CEDAR_PROFILE_SCALING_TOP_K=5
export CEDAR_PROFILE_SCALING_MAX_SEC=10
# Representation-aware compute curves (per point window in seconds).
export CEDAR_PROFILE_COMPUTE_MODEL=1
export CEDAR_PROFILE_COMPUTE_TARGET_SEC=${CEDAR_PROFILE_COMPUTE_TARGET_SEC:-5.0}
export CEDAR_PROFILE_COMPUTE_MAX_CALLS=${CEDAR_PROFILE_COMPUTE_MAX_CALLS:-400}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CEDAR_DATA_JUICER_ROOT=/workspace/OptimalCedar/data-juicer
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy || true

python -u "$OUT/entry.py" evaluation/eval_cedar.py \
  --dataset_file evaluation/pipelines/simclrv2/cedar_dataset.py \
  --dataset_kwargs "dataset_path=/workspace/OptimalCedar/evaluation/datasets/imagenette2/imagenette2/train" \
  --batch_size 4 \
  --num_epochs 20 \
  --num_total_samples 189380 \
  --profiled_stats "$OUT/shared.yaml" \
  --use_ray --ray_ip 172.23.166.105:6379 \
  --run_profiling --disable_controller --disable_optimizer --disable_prefetch \
  --disable_caching

OUT="$OUT" python - <<'PY'
import hashlib, json, os, yaml
from pathlib import Path
out = Path(os.environ["OUT"])
profile = out / "shared.yaml"
data = yaml.safe_load(profile.read_text())
model = data.get("physical_model", {}).get("compute_model", {})
ops = model.get("operators", {})
summary = {
    "profile": str(profile),
    "sha256": hashlib.sha256(profile.read_bytes()).hexdigest(),
    "compute_model_operators": len(ops),
    "classes": {k: sorted(v.get("by_class", {})) for k, v in sorted(ops.items(), key=lambda kv: int(kv[0]))},
    "source": [model.get("source_class"), model.get("source_elements")],
}
print(json.dumps(summary, indent=1))
(out / "profile_summary.json").write_text(json.dumps(summary, indent=1))
PY
