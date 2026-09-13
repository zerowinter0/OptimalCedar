#!/usr/bin/env bash
set -euo pipefail

cd /workspace/OptimalCedar
source env/bin/activate

readonly WORKFLOW_ROOT="/workspace/OptimalCedar/outputs/motivation_multimodal/simclrv2_multimodal_local_clip_w1_20260909"
readonly PROFILE_ROOT="${WORKFLOW_ROOT}/profile"
readonly PLAN_ROOT="${WORKFLOW_ROOT}/plans"
readonly RUNTIME_ROOT="${WORKFLOW_ROOT}/runtime"
readonly PROFILE_PATH="${PROFILE_ROOT}/simclrv2_multimodal.yaml"
readonly PLAN_PATH="${PLAN_ROOT}/four_plans.json"
readonly RUNTIME_PATH="${RUNTIME_ROOT}/four_plans.json"
readonly DATASET_FILE="evaluation/pipelines/simclrv2_multimodal/cedar_dataset.py"
readonly DATASET_PATH="/workspace/OptimalCedar/outputs/motivation_multimodal/fixture_pilot2000/pilot.jsonl"
readonly IMAGE_ROOT="/workspace/OptimalCedar/datasets/coco/val2017"
readonly THRESHOLD_PATH="/workspace/OptimalCedar/outputs/motivation_multimodal/four_plan_perplexity_alternative_20260908/calibration/thresholds/p50-q70-a10-c80-b80.json"
readonly DATASET_KWARGS="dataset_path=${DATASET_PATH},image_root=${IMAGE_ROOT},threshold_path=${THRESHOLD_PATH}"

mkdir -p "${PROFILE_ROOT}" "${PLAN_ROOT}" "${RUNTIME_ROOT}"
rm -f \
    "${WORKFLOW_ROOT}/COMPLETE" \
    "${WORKFLOW_ROOT}/FAILED" \
    "${PLAN_PATH}" \
    "${RUNTIME_PATH}"
trap 'touch "${WORKFLOW_ROOT}/FAILED"' ERR

export CEDAR_PROFILE_TIME_SEC=10
export CEDAR_PROFILE_RAY_ACTORS=1
export CEDAR_PROFILE_SMP_PROCS=1
export CEDAR_PROFILE_FILTER_SELECTIVITY=1
export CEDAR_MATCH_PROFILE_RESOURCES=1
export CEDAR_PROFILE_MATCH_CPU_BUDGET=64
export CEDAR_PROFILE_MATCH_RAY_CPU_BUDGET=64
export CEDAR_PROFILE_MATCH_FIXED_LOCAL_WORKERS=1
export CEDAR_LAYERED_ADAPTIVE_PROFILE=1
export CEDAR_ADAPTIVE_PROFILE_MIN_SEC=10
export CEDAR_ADAPTIVE_PROFILE_MAX_SEC=10
export CEDAR_ADAPTIVE_PROFILE_TARGET_RSE=0.10
export CEDAR_ADAPTIVE_PROFILE_MIN_OBS=30
export CEDAR_PROFILE_POOL_SAMPLES=64
export CEDAR_PROFILE_POOL_BYTES_PER_PIPE=$((64 * 1024 * 1024))
export CEDAR_PROFILE_POOL_BYTES_TOTAL=$((512 * 1024 * 1024))
export CEDAR_PROFILE_SCALING_TOP_K=2
export CEDAR_PROFILE_SCALING_WIDTH=8
export CEDAR_PROFILE_BOUNDARY_MODEL=1
export CEDAR_PROFILE_INFER_COMPUTE_SCALING=1

if [[ "${SKIP_PROFILE:-0}" == "1" ]]; then
    echo "[1/3] reuse validated fixed-local-CLIP W=1 profile"
    test -s "${PROFILE_PATH}"
else
    echo "[1/3] profile fixed-local-CLIP workload at W=1"
    rm -f "${PROFILE_PATH}" "${PROFILE_ROOT}/validation.json"
    python evaluation/eval_cedar.py \
        --dataset_file "${DATASET_FILE}" \
        --batch_size 1 \
        --num_epochs 1 \
        --dataset_kwargs "${DATASET_KWARGS}" \
        --profiled_stats "${PROFILE_PATH}" \
        --run_profiling \
        --disable_optimizer \
        --disable_controller \
        --disable_caching \
        --use_ray
fi

python evaluation/chapter6_experiments/validate_adaptive_layered_profiles.py \
    --profile-root "${PROFILE_ROOT}" \
    --workloads simclrv2_multimodal \
    --output "${PROFILE_ROOT}/validation.json"

echo "[2/3] materialize and cost four plans"
python evaluation/compare_optimizer_perf.py \
    --dataset_file "${DATASET_FILE}" \
    --batch_size 1 \
    --num_epochs 1 \
    --num_total_samples 2000 \
    --dataset_kwargs "${DATASET_KWARGS}" \
    --profiled_stats "${PROFILE_PATH}" \
    --optimizers optimizer simple_dp_optimizer dp_optimizer simple_dp_ray_candidate_optimizer \
    --calculate_plan_cost \
    --num_repeats 1 \
    --optimizer_time_limit_sec 3600 \
    --cedar_reorder_timeout_sec 3600 \
    --disable_cedar_runtime_timeout \
    --disable_caching \
    --enable_local_parallelism \
    --use_ray \
    --ray_available_parallelism 64 \
    --match_profile_resources \
    --cpu_budget 64 \
    --ray_cpu_budget 64 \
    --fixed_local_workers_ablation 1 \
    --results_path "${PLAN_PATH}"

python evaluation/pipelines/simclrv2_multimodal/validate_gpu_prefix_plans.py \
    --results "${PLAN_PATH}" \
    --optimizers optimizer simple_dp_optimizer dp_optimizer simple_dp_ray_candidate_optimizer

echo "[3/3] execute three round-robin repeats at W=1"
python evaluation/compare_optimizer_perf.py \
    --dataset_file "${DATASET_FILE}" \
    --batch_size 1 \
    --num_epochs 1 \
    --num_total_samples 2000 \
    --full_data_run \
    --dataset_kwargs "${DATASET_KWARGS}" \
    --profiled_stats "${PROFILE_PATH}" \
    --optimizers optimizer simple_dp_optimizer dp_optimizer simple_dp_ray_candidate_optimizer \
    --num_repeats 3 \
    --optimizer_time_limit_sec 3600 \
    --cedar_reorder_timeout_sec 3600 \
    --disable_cedar_runtime_timeout \
    --disable_caching \
    --enable_local_parallelism \
    --use_ray \
    --ray_available_parallelism 64 \
    --match_profile_resources \
    --cpu_budget 64 \
    --ray_cpu_budget 64 \
    --fixed_local_workers_ablation 1 \
    --results_path "${RUNTIME_PATH}"

python evaluation/pipelines/simclrv2_multimodal/validate_gpu_prefix_plans.py \
    --results "${RUNTIME_PATH}" \
    --optimizers optimizer simple_dp_optimizer dp_optimizer simple_dp_ray_candidate_optimizer

touch "${WORKFLOW_ROOT}/COMPLETE"
rm -f "${WORKFLOW_ROOT}/FAILED"
echo "[workflow] complete"
