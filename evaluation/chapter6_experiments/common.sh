#!/usr/bin/env bash
set -euo pipefail

CH6_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CH6_REPO_ROOT="$(cd "${CH6_SCRIPT_DIR}/../.." && pwd)"

if [[ -n "${CH6_ENV:-}" ]]; then
  # shellcheck source=/dev/null
  source "${CH6_ENV}"
elif [[ -f "${CH6_SCRIPT_DIR}/experiments.env" ]]; then
  # shellcheck source=/dev/null
  source "${CH6_SCRIPT_DIR}/experiments.env"
fi

: "${CH6_OUT_DIR:=${CH6_SCRIPT_DIR}/results/$(date -u +%Y%m%dT%H%M%SZ)}"
: "${CH6_WORKLOADS:=bloom_oscar}"
: "${CH6_OPTIMIZERS:=optimizer dp_cedar_optimizer dp_seperate_optimizer dp_optimizer}"
: "${CH6_REPEATS:=3}"
: "${CH6_NUM_EPOCHS:=1}"
: "${CH6_NUM_TOTAL_SAMPLES:=10000}"
: "${CH6_DATA_NUM_TOTAL_SAMPLES:=1000}"
: "${CH6_PROFILE_SAMPLES:=200}"
: "${CH6_USE_RAY:=1}"
: "${CH6_RAY_IP:=}"
: "${CH6_DISABLE_OFFLOAD:=0}"
: "${CH6_ENABLE_LOCAL_PARALLELISM:=1}"
: "${CH6_DISABLE_CACHING:=0}"
: "${CH6_FULL_DATA_RUN:=0}"
: "${CH6_ALLOW_TORCH_PARALLELISM:=0}"
: "${CH6_LOG_LEVEL:=INFO}"
: "${CH6_HOME:=${HOME}}"

export CH6_OUT_DIR CH6_WORKLOADS CH6_OPTIMIZERS CH6_REPEATS CH6_NUM_EPOCHS
export CH6_NUM_TOTAL_SAMPLES CH6_DATA_NUM_TOTAL_SAMPLES CH6_PROFILE_SAMPLES
export CH6_USE_RAY CH6_RAY_IP CH6_DISABLE_OFFLOAD CH6_ENABLE_LOCAL_PARALLELISM
export CH6_DISABLE_CACHING CH6_FULL_DATA_RUN CH6_ALLOW_TORCH_PARALLELISM
export CH6_LOG_LEVEL CH6_HOME

ch6_init() {
  cd "${CH6_REPO_ROOT}"
  # Project convention: activate the repository environment before running code.
  # shellcheck source=/dev/null
  source "${CH6_REPO_ROOT}/env/bin/activate"
  export PYTHONPATH="${CH6_REPO_ROOT}:${PYTHONPATH:-}"
  mkdir -p "${CH6_OUT_DIR}/logs" "${CH6_OUT_DIR}/profiles" \
    "${CH6_OUT_DIR}/raw" "${CH6_OUT_DIR}/summary" "${CH6_OUT_DIR}/plans"
}

ch6_log() {
  printf '[chapter6] %s\n' "$*" >&2
}

ch6_workloads() {
  local workload
  for workload in ${CH6_WORKLOADS}; do
    printf '%s\n' "${workload}"
  done
}

ch6_workload_dataset_file() {
  case "$1" in
    bloom_oscar)
      printf '%s\n' "evaluation/pipelines/bloom_oscar/cedar_dataset.py"
      ;;
    llava_pretrain)
      printf '%s\n' "evaluation/pipelines/llava_pretrain/cedar_dataset.py"
      ;;
    wikitext103)
      printf '%s\n' "evaluation/pipelines/wikitext103/cedar_dataset.py"
      ;;
    simclrv2)
      printf '%s\n' "evaluation/pipelines/simclrv2/cedar_dataset.py"
      ;;
    commonvoice)
      printf '%s\n' "evaluation/pipelines/commonvoice/cedar_dataset.py"
      ;;
    coco)
      printf '%s\n' "evaluation/pipelines/coco/cedar_dataset.py"
      ;;
    *)
      ch6_log "Unknown workload: $1"
      return 2
      ;;
  esac
}

ch6_workload_profile() {
  local default_profile
  case "$1" in
    bloom_oscar)
      default_profile="${CH6_OUT_DIR}/profiles/bloom_oscar_profile.yml"
      printf '%s\n' "${BLOOM_PROFILE_PATH:-${default_profile}}"
      ;;
    llava_pretrain)
      default_profile="${CH6_OUT_DIR}/profiles/llava_pretrain_profile.yml"
      printf '%s\n' "${LLAVA_PROFILE_PATH:-${default_profile}}"
      ;;
    wikitext103)
      printf '%s\n' "${WIKITEXT_PROFILE_PATH:-evaluation/pipelines/wikitext103/stats/cedar.yaml}"
      ;;
    simclrv2)
      printf '%s\n' "${SIMCLR_PROFILE_PATH:-evaluation/pipelines/simclrv2/stats/cedar_stats.yaml}"
      ;;
    commonvoice)
      printf '%s\n' "${COMMONVOICE_PROFILE_PATH:-evaluation/pipelines/commonvoice/stats/cedar.yaml}"
      ;;
    coco)
      printf '%s\n' "${COCO_PROFILE_PATH:-evaluation/pipelines/coco/stats/coco_local_stats.yaml}"
      ;;
    *)
      ch6_log "Unknown workload: $1"
      return 2
      ;;
  esac
}

ch6_workload_kwargs() {
  case "$1" in
    bloom_oscar)
      printf '%s\n' "dataset_path=${BLOOM_DATASET_PATH:-/tmp/redpajama_backup_3gb_for_bloom_oscar.jsonl}"
      ;;
    llava_pretrain)
      printf '%s\n' "dataset_path=${LLAVA_DATASET_PATH:-/tmp/llava_pretrain_cedar_fixture.jsonl},image_root=${LLAVA_IMAGE_ROOT:-/tmp/llava_pretrain_cedar_images}"
      ;;
    wikitext103|simclrv2|commonvoice|coco)
      printf '%s\n' ""
      ;;
    *)
      ch6_log "Unknown workload: $1"
      return 2
      ;;
  esac
}

ch6_workload_samples() {
  case "$1" in
    bloom_oscar)
      printf '%s\n' "${BLOOM_NUM_TOTAL_SAMPLES:-${CH6_NUM_TOTAL_SAMPLES}}"
      ;;
    llava_pretrain)
      printf '%s\n' "${LLAVA_NUM_TOTAL_SAMPLES:-${CH6_NUM_TOTAL_SAMPLES}}"
      ;;
    wikitext103)
      printf '%s\n' "${WIKITEXT_NUM_TOTAL_SAMPLES:-100000}"
      ;;
    simclrv2)
      printf '%s\n' "${SIMCLR_NUM_TOTAL_SAMPLES:-10000}"
      ;;
    commonvoice)
      printf '%s\n' "${COMMONVOICE_NUM_TOTAL_SAMPLES:-10000}"
      ;;
    coco)
      printf '%s\n' "${COCO_NUM_TOTAL_SAMPLES:-10000}"
      ;;
    *)
      ch6_log "Unknown workload: $1"
      return 2
      ;;
  esac
}

ch6_run_and_log() {
  local log_path="$1"
  shift
  mkdir -p "$(dirname "${log_path}")"
  ch6_log "log: ${log_path}"
  {
    printf 'command:'
    printf ' %q' "$@"
    printf '\n'
    HOME="${CH6_HOME}" "$@"
  } 2>&1 | tee "${log_path}"
}

ch6_add_common_compare_flags() {
  local -n _args_ref="$1"
  if [[ "${CH6_USE_RAY}" == "1" ]]; then
    _args_ref+=(--use_ray)
    if [[ -n "${CH6_RAY_IP}" ]]; then
      _args_ref+=(--ray_ip "${CH6_RAY_IP}")
    fi
  fi
  if [[ "${CH6_DISABLE_OFFLOAD}" == "1" ]]; then
    _args_ref+=(--disable_offload)
  fi
  if [[ "${CH6_ENABLE_LOCAL_PARALLELISM}" == "1" ]]; then
    _args_ref+=(--enable_local_parallelism)
  fi
  if [[ "${CH6_DISABLE_CACHING}" == "1" ]]; then
    _args_ref+=(--disable_caching)
  fi
  if [[ "${CH6_FULL_DATA_RUN}" == "1" ]]; then
    _args_ref+=(--full_data_run)
  fi
  if [[ "${CH6_ALLOW_TORCH_PARALLELISM}" == "1" ]]; then
    _args_ref+=(--allow_torch_parallelism)
  fi
}

ch6_run_compare_optimizer_perf() {
  local workload="$1"
  local output_json="$2"
  local log_path="$3"
  shift 3

  local dataset_file profile_path dataset_kwargs num_samples
  dataset_file="$(ch6_workload_dataset_file "${workload}")"
  profile_path="$(ch6_workload_profile "${workload}")"
  dataset_kwargs="$(ch6_workload_kwargs "${workload}")"
  num_samples="$(ch6_workload_samples "${workload}")"

  if [[ ! -f "${profile_path}" ]]; then
    ch6_log "Profile not found for ${workload}: ${profile_path}"
    ch6_log "Run run_profile_workloads.sh first or set the workload profile path."
    return 2
  fi

  local optimizers=()
  read -r -a optimizers <<< "${CH6_OPTIMIZERS}"

  local args=(
    python evaluation/compare_optimizer_perf.py
    --dataset_file "${dataset_file}"
    --profiled_stats "${profile_path}"
    --num_epochs "${CH6_NUM_EPOCHS}"
    --num_total_samples "${num_samples}"
    --data_num_total_samples "${CH6_DATA_NUM_TOTAL_SAMPLES}"
    --num_repeats "${CH6_REPEATS}"
    --results_path "${output_json}"
    --log_level "${CH6_LOG_LEVEL}"
    --optimizers "${optimizers[@]}"
  )
  if [[ -n "${dataset_kwargs}" ]]; then
    args+=(--dataset_kwargs "${dataset_kwargs}")
  fi
  ch6_add_common_compare_flags args
  args+=("$@")

  ch6_run_and_log "${log_path}" "${args[@]}"
}

ch6_profile_workload() {
  local workload="$1"
  local profile_path="$2"
  local num_profile_samples="$3"
  local log_path="$4"

  local dataset_file dataset_kwargs
  dataset_file="$(ch6_workload_dataset_file "${workload}")"
  dataset_kwargs="$(ch6_workload_kwargs "${workload}")"

  local args=(
    python evaluation/eval_cedar.py
    --dataset_file "${dataset_file}"
    --profiled_stats "${profile_path}"
    --run_profiling
    --num_total_samples "${num_profile_samples}"
    --disable_optimizer
    --disable_controller
    --disable_prefetch
    --log_level "${CH6_LOG_LEVEL}"
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
  if [[ "${CH6_DISABLE_OFFLOAD}" == "1" ]]; then
    args+=(--disable_offload)
  fi

  mkdir -p "$(dirname "${profile_path}")"
  ch6_run_and_log "${log_path}" "${args[@]}"
}
