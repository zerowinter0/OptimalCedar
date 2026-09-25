#!/usr/bin/env bash
# Regenerate the layered profile *with* the representation-aware compute model
# for every workload of the final W-only campaign.
#
# Usage (inside the container):
#   bash scripts/pico_final_profiles_20260924.sh [workloads...]
set -uo pipefail
cd /workspace/OptimalCedar
source env/bin/activate

OUT=${OUT:-outputs/affine_repr_profile_20260924}
ENTRY=outputs/ultimate_eight_optimizers_fix_20260921/entry.py
RAY_IP=172.23.166.105:6379
mkdir -p "$OUT"
cp -f "$ENTRY" "$OUT/entry.py"
# Always refresh the shipped snapshot: a stale copy silently profiles with old
# code (the first COCO attempt died inside an already-fixed guard because the
# snapshot predated the fix).
rm -rf "$OUT/modules"
mkdir -p "$OUT/modules/evaluation"
cp -r cedar "$OUT/modules"/cedar
find evaluation -maxdepth 1 -type f -name '*.py' -exec cp {} "$OUT/modules/evaluation/" \;
cp -r evaluation/pipelines "$OUT/modules/evaluation/pipelines"
find "$OUT/modules" -name '__pycache__' -type d -prune -exec rm -rf {} +

export CEDAR_RAY_PLACEMENT_RESOURCE=cedar_remote CEDAR_RAY_REQUIRE_REMOTE=1
export CEDAR_PROFILE_RAY_ACTORS=1 CEDAR_PROFILE_SMP_PROCS=1
export CEDAR_PROFILE_TIME_SEC=10 CEDAR_PROFILE_BOUNDARY_MODEL=1
export CEDAR_REUSE_BOUNDARY_MODEL=0 CEDAR_PROFILE_INFER_COMPUTE_SCALING=1
export CEDAR_LAYERED_ADAPTIVE_PROFILE=1 CEDAR_ADAPTIVE_PROFILE_MIN_SEC=3
export CEDAR_ADAPTIVE_PROFILE_MAX_SEC=30 CEDAR_ADAPTIVE_PROFILE_TARGET_RSE=0.10
export CEDAR_ADAPTIVE_PROFILE_MIN_OBS=30 CEDAR_PROFILE_POOL_SAMPLES=64
export CEDAR_PROFILE_POOL_BYTES_PER_PIPE=$((64 * 1024 * 1024))
export CEDAR_PROFILE_POOL_BYTES_TOTAL=$((512 * 1024 * 1024))
export CEDAR_PROFILE_SCALING_WIDTHS=1,2,4,8 CEDAR_PROFILE_SCALING_TOP_K=5
export CEDAR_PROFILE_SCALING_MAX_SEC=10
export CEDAR_PROFILE_COMPUTE_MODEL=1
export CEDAR_PROFILE_COMPUTE_TARGET_SEC=${CEDAR_PROFILE_COMPUTE_TARGET_SEC:-5.0}
export CEDAR_PROFILE_COMPUTE_MAX_CALLS=${CEDAR_PROFILE_COMPUTE_MAX_CALLS:-400}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CEDAR_DATA_JUICER_ROOT=/workspace/OptimalCedar/data-juicer
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy || true

profile_workload() {
  local workload=$1 dataset_file=$2 dataset_kwargs=$3 epochs=$4 samples=$5 batch=$6
  local work="$OUT/$workload"
  mkdir -p "$work"
  local profile="$work/shared.yaml"
  if [ -s "$profile" ]; then
    echo "SKIP profile $workload (exists)"
    return 0
  fi
  echo "PROFILE $workload $(date -Is)"
  python -u "$OUT/entry.py" evaluation/eval_cedar.py \
    --dataset_file "$dataset_file" --dataset_kwargs "$dataset_kwargs" \
    --batch_size "$batch" --num_epochs "$epochs" --num_total_samples "$samples" \
    --profiled_stats "$profile" \
    --use_ray --ray_ip "$RAY_IP" \
    --run_profiling --disable_controller --disable_optimizer --disable_prefetch \
    --disable_caching > "$work/profile.log" 2>&1
  echo "PROFILE-DONE $workload exit=$? $(date -Is)"
}

for workload in "$@"; do
  case "$workload" in
    simclrv2)
      profile_workload simclrv2 evaluation/pipelines/simclrv2/cedar_dataset.py \
        "dataset_path=/workspace/OptimalCedar/evaluation/datasets/imagenette2/imagenette2/train" \
        20 189380 4 ;;
    simclrv2_cache)
      profile_workload simclrv2_cache evaluation/pipelines/simclrv2/cedar_cache_dataset.py \
        "dataset_path=/workspace/OptimalCedar/evaluation/datasets/imagenette2/imagenette2/train" \
        20 189380 4 ;;
    commonvoice)
      profile_workload commonvoice evaluation/pipelines/commonvoice/cedar_dataset.py \
        "dataset_path=/workspace/OptimalCedar/datasets/commonvoice/cv-corpus-15.0-delta-2023-09-08/en/clips,max_samples=300000" \
        1 300000 1 ;;
    coco)
      profile_workload coco evaluation/pipelines/coco/cedar_dataset.py \
        "dataset_path=/workspace/OptimalCedar/evaluation/datasets/coco,split=train2017" \
        1 50000 1 ;;
    llava_pretrain)
      profile_workload llava_pretrain evaluation/pipelines/llava_pretrain/cedar_dataset.py \
        "dataset_path=/workspace/OptimalCedar/outputs/ultimate_eight_optimizers_fix_20260921/inputs/llava_pretrain.jsonl,image_root=/workspace/OptimalCedar/evaluation/datasets/llava_pretrain" \
        1 50000 1 ;;
    wikitext103)
      profile_workload wikitext103 evaluation/pipelines/wikitext103/cedar_dataset.py \
        "dataset_path=/workspace/OptimalCedar/evaluation/datasets/wikitext103,max_samples=100000" \
        1 100000 1 ;;
    *)
      echo "unknown workload $workload" ;;
  esac
done
echo "PROFILES-FINISHED $(date -Is)"
