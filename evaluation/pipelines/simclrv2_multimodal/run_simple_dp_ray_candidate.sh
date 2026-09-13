#!/usr/bin/env bash
set -euo pipefail

cd /workspace/OptimalCedar
source env/bin/activate

readonly OUTPUT_ROOT="/workspace/OptimalCedar/outputs/motivation_multimodal/simclrv2_simple_dp_ray_candidate_w8_20260909"
readonly PROFILE_PATH="/workspace/OptimalCedar/outputs/motivation_multimodal/simclrv2_multimodal_gpu_profile_fixed_20260909/gate/profile/simclrv2_multimodal.yaml"
readonly DATASET_PATH="/workspace/OptimalCedar/outputs/motivation_multimodal/fixture_pilot2000/pilot.jsonl"
readonly IMAGE_ROOT="/workspace/OptimalCedar/datasets/coco/val2017"
readonly THRESHOLD_PATH="/workspace/OptimalCedar/outputs/motivation_multimodal/four_plan_perplexity_alternative_20260908/calibration/thresholds/p50-q70-a10-c80-b80.json"
readonly RESULTS_PATH="${OUTPUT_ROOT}/results/pico_cedar_candidate.json"
readonly DATASET_KWARGS="dataset_path=${DATASET_PATH},image_root=${IMAGE_ROOT},threshold_path=${THRESHOLD_PATH}"

test -s "${PROFILE_PATH}"
mkdir -p "${OUTPUT_ROOT}/results" "${OUTPUT_ROOT}/logs"
rm -f "${OUTPUT_ROOT}/COMPLETE" "${OUTPUT_ROOT}/FAILED" "${RESULTS_PATH}"
trap 'touch "${OUTPUT_ROOT}/FAILED"' ERR

export CEDAR_MATCH_PROFILE_RESOURCES=1
export CEDAR_PROFILE_MATCH_CPU_BUDGET=64
export CEDAR_PROFILE_MATCH_RAY_CPU_BUDGET=64
export CEDAR_PROFILE_MATCH_FIXED_LOCAL_WORKERS=8

python evaluation/compare_optimizer_perf.py \
    --dataset_file evaluation/pipelines/simclrv2_multimodal/cedar_dataset.py \
    --batch_size 1 \
    --num_epochs 1 \
    --num_total_samples 2000 \
    --full_data_run \
    --dataset_kwargs "${DATASET_KWARGS}" \
    --profiled_stats "${PROFILE_PATH}" \
    --optimizers dp_optimizer optimizer simple_dp_optimizer simple_dp_ray_candidate_optimizer \
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
    --fixed_local_workers_ablation 8 \
    --results_path "${RESULTS_PATH}"

python evaluation/pipelines/simclrv2_multimodal/validate_gpu_prefix_plans.py \
    --results "${RESULTS_PATH}"

touch "${OUTPUT_ROOT}/COMPLETE"
rm -f "${OUTPUT_ROOT}/FAILED"
