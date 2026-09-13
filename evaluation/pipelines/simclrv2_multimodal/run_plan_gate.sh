#!/usr/bin/env bash
set -euo pipefail

cd /workspace/OptimalCedar
source env/bin/activate

readonly OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/OptimalCedar/outputs/motivation_multimodal/simclrv2_multimodal_gpu_prefix_gate_20260908}"
readonly DATASET_FILE="evaluation/pipelines/simclrv2_multimodal/cedar_dataset.py"
readonly DATASET_PATH="/workspace/OptimalCedar/outputs/motivation_multimodal/fixture_pilot2000/pilot.jsonl"
readonly IMAGE_ROOT="/workspace/OptimalCedar/datasets/coco/val2017"
readonly THRESHOLD_PATH="/workspace/OptimalCedar/outputs/motivation_multimodal/four_plan_perplexity_alternative_20260908/calibration/thresholds/p50-q70-a10-c80-b80.json"
readonly PROFILE_PATH="${OUTPUT_ROOT}/profile/simclrv2_multimodal.yaml"
readonly PLAN_PATH="${OUTPUT_ROOT}/plans/cedar_simple_dp_pico.json"
readonly DATASET_KWARGS="dataset_path=${DATASET_PATH},image_root=${IMAGE_ROOT},threshold_path=${THRESHOLD_PATH}"

mkdir -p "${OUTPUT_ROOT}/profile" "${OUTPUT_ROOT}/plans"
rm -f "${OUTPUT_ROOT}/COMPLETE" "${OUTPUT_ROOT}/FAILED"
rm -f "${PLAN_PATH}" "${OUTPUT_ROOT}/profile/validation.json"
trap 'touch "${OUTPUT_ROOT}/FAILED"' ERR

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
    test -s "${PROFILE_PATH}"
else
    rm -f "${PROFILE_PATH}"
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
    --profile-root "${OUTPUT_ROOT}/profile" \
    --workloads simclrv2_multimodal \
    --output "${OUTPUT_ROOT}/profile/validation.json"

python evaluation/compare_optimizer_perf.py \
    --dataset_file "${DATASET_FILE}" \
    --batch_size 1 \
    --num_epochs 1 \
    --num_total_samples 2000 \
    --dataset_kwargs "${DATASET_KWARGS}" \
    --profiled_stats "${PROFILE_PATH}" \
    --optimizers optimizer simple_dp_optimizer dp_optimizer \
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
    --results "${PLAN_PATH}"

touch "${OUTPUT_ROOT}/COMPLETE"
rm -f "${OUTPUT_ROOT}/FAILED"
