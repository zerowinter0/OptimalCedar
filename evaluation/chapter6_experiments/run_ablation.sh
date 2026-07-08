#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

ch6_init

ablation_dir="${CH6_OUT_DIR}/raw/ablation"
mkdir -p "${ablation_dir}"

for workload in $(ch6_workloads); do
  dataset_file="$(ch6_workload_dataset_file "${workload}")"
  profile_path="$(ch6_workload_profile "${workload}")"
  dataset_kwargs="$(ch6_workload_kwargs "${workload}")"
  num_samples="$(ch6_workload_samples "${workload}")"
  output_json="${ablation_dir}/${workload}_dp_ablation.json"
  log_path="${CH6_OUT_DIR}/logs/ablation_${workload}.log"

  if [[ ! -f "${profile_path}" ]]; then
    ch6_log "Profile not found for ${workload}: ${profile_path}"
    continue
  fi

  ablation_conditions=()
  read -r -a ablation_conditions <<< "${CH6_ABLATION_CONDITIONS:-full_dp stagewise_dp no_reorder no_fusion no_offload no_cache no_parallelism}"

  args=(
    python "${SCRIPT_DIR}/run_dp_ablation.py"
    --dataset_file "${dataset_file}"
    --profiled_stats "${profile_path}"
    --num_epochs "${CH6_NUM_EPOCHS}"
    --num_total_samples "${num_samples}"
    --data_num_total_samples "${CH6_DATA_NUM_TOTAL_SAMPLES}"
    --num_repeats "${CH6_REPEATS}"
    --results_path "${output_json}"
    --plans_dir "${CH6_OUT_DIR}/plans/${workload}_ablation"
    --log_level "${CH6_LOG_LEVEL}"
    --conditions "${ablation_conditions[@]}"
  )
  if [[ -n "${dataset_kwargs}" ]]; then
    args+=(--dataset_kwargs "${dataset_kwargs}")
  fi
  if [[ "${CH6_USE_RAY}" == "1" ]]; then
    args+=(--use_ray)
    if [[ -n "${CH6_RAY_IP}" ]]; then
      args+=(--ray_ip "${CH6_RAY_IP}")
    fi
  fi
  if [[ "${CH6_ENABLE_LOCAL_PARALLELISM}" == "1" ]]; then
    args+=(--enable_local_parallelism)
  fi
  if [[ "${CH6_FULL_DATA_RUN}" == "1" ]]; then
    args+=(--full_data_run)
  fi
  if [[ "${CH6_ALLOW_TORCH_PARALLELISM}" == "1" ]]; then
    args+=(--allow_torch_parallelism)
  fi
  if [[ "${CH6_DISABLE_CACHING}" == "1" ]]; then
    args+=(--disable_caching)
  fi

  ch6_run_and_log "${log_path}" "${args[@]}"
done
