#!/usr/bin/env bash
# Regenerate the SimCLRv2 shared profile with the representation-aware compute
# model, under the same protocol the six-workload campaign used.
#
# Usage (inside the container):
#   bash scripts/profile_simclrv2_repr_20260924.sh
set -euo pipefail
cd /workspace/OptimalCedar
source env/bin/activate

OUT=${1:-outputs/affine_repr_profile_20260924}
mkdir -p "$OUT"

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
# Representation-aware compute curves: per-point window in seconds.
export CEDAR_PROFILE_COMPUTE_MODEL=1
export CEDAR_PROFILE_COMPUTE_TARGET_SEC=5.0
export CEDAR_PROFILE_COMPUTE_MAX_CALLS=400
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy || true

python -u evaluation/eval_cedar.py \
  --dataset_file evaluation/pipelines/target_pipeline/simclr/cedar_dataset.py \
  --dataset_kwargs "dataset_path=/workspace/OptimalCedar/evaluation/datasets/imagenette2/imagenette2/train" \
  --batch_size 4 \
  --num_epochs 20 \
  --num_total_samples 189380 \
  --profiled_stats "$OUT/shared.yaml" \
  --use_ray --ray_ip 172.23.166.105:6379 \
  --run_profiling --disable_controller --disable_optimizer --disable_prefetch \
  --disable_caching

OUT="$OUT" python - <<'PY'
import os, yaml, hashlib
from pathlib import Path
out = Path(os.environ["OUT"])
profile = out / "shared.yaml"
data = yaml.safe_load(profile.read_text())
model = data.get("physical_model", {}).get("compute_model", {})
ops = model.get("operators", {})
print("profile:", profile, "sha256:", hashlib.sha256(profile.read_bytes()).hexdigest()[:16])
print("compute_model operators:", len(ops))
print("classes per operator:", {k: sorted(v.get("by_class", {})) for k, v in sorted(ops.items(), key=lambda kv: int(kv[0]))})
print("source:", model.get("source_class"), model.get("source_elements"))
PY
