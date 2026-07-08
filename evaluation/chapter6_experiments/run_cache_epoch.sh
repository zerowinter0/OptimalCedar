#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

ch6_init

cache_dir="${CH6_OUT_DIR}/raw/cache_epoch"
mkdir -p "${cache_dir}"

for workload in $(ch6_workloads); do
  dataset_file="$(ch6_workload_dataset_file "${workload}")"
  profile_path="$(ch6_workload_profile "${workload}")"
  dataset_kwargs="$(ch6_workload_kwargs "${workload}")"
  num_samples="$(ch6_workload_samples "${workload}")"
  output_json="${cache_dir}/${workload}_cache_epoch.json"
  log_path="${CH6_OUT_DIR}/logs/cache_epoch_${workload}.log"

  if [[ ! -f "${profile_path}" ]]; then
    ch6_log "Profile not found for ${workload}: ${profile_path}"
    continue
  fi

  args=(
    python "${SCRIPT_DIR}/run_dp_ablation.py"
    --dataset_file "${dataset_file}"
    --profiled_stats "${profile_path}"
    --num_epochs "${CH6_CACHE_EPOCHS:-3}"
    --num_total_samples "${num_samples}"
    --data_num_total_samples "${CH6_DATA_NUM_TOTAL_SAMPLES}"
    --num_repeats "${CH6_REPEATS}"
    --results_path "${output_json}"
    --plans_dir "${CH6_OUT_DIR}/plans/${workload}_cache_epoch"
    --log_level "${CH6_LOG_LEVEL}"
    --conditions full_dp no_cache
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

  ch6_run_and_log "${log_path}" "${args[@]}"
done
