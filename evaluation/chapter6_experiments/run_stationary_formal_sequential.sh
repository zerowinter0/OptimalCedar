#!/usr/bin/env bash
# Profile, validate, and formally evaluate one workload completely before
# moving to the next workload.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULT_ROOT="${RESULT_ROOT:-${REPO_ROOT}/outputs/chapter6_experiments/stationary_formal_sequential_v2}"
WORKLOADS="${WORKLOADS_OVERRIDE:-alpaca_cot,simclrv2_cache,pile_europarl,pile_hackernews,pile_pubmed_abstracts,pile_uspto_backgrounds,redpajama_code,general_video_refine,stackexchange,commonvoice}"
RESUME_COMPLETED_WORKLOADS="${RESUME_COMPLETED_WORKLOADS:-1}"
PROFILE_RUN_LABEL="${PROFILE_RUN_LABEL:-v2}"
RUNNER="${REPO_ROOT}/evaluation/chapter6_experiments/run_unified_formal_seven_all_ten.sh"

cd "${REPO_ROOT}"
# shellcheck source=/dev/null
source env/bin/activate
mkdir -p "${RESULT_ROOT}/workloads"

IFS=',' read -r -a workload_list <<< "${WORKLOADS}"
printf '[%s] START sequential stationary formal experiment: %s\n' \
  "$(date -Is)" "${WORKLOADS}"

for workload in "${workload_list[@]}"; do
  workload_root="${RESULT_ROOT}/workloads/${workload}"
  profile_path="${workload_root}/profiles/${workload}.yaml"
  if [[ "${RESUME_COMPLETED_WORKLOADS}" == "1" && \
        -f "${workload_root}/COMPLETE" ]]; then
    printf '[%s] REUSE completed workload %s\n' "$(date -Is)" "${workload}"
    continue
  fi

  mkdir -p "${workload_root}"
  printf '%s\n' "${workload}" > "${RESULT_ROOT}/CURRENT_WORKLOAD"
  printf '[%s] BEGIN workload %s\n' "$(date -Is)" "${workload}"

  reuse_profile=0
  if [[ -s "${profile_path}" ]]; then
    reuse_profile=1
    printf '[%s] REUSE completed profile for %s: %s\n' \
      "$(date -Is)" "${workload}" "${profile_path}"
  fi

  RESULT_ROOT="${workload_root}" \
  WORKLOADS_OVERRIDE="${workload}" \
  PROFILE_RUN_ID="stationary_sequential_${workload}_${PROFILE_RUN_LABEL}" \
  REUSE_COMPLETED_PROFILES="${reuse_profile}" \
  RESUME_FORMAL_RESULTS=1 \
    bash "${RUNNER}"

  if [[ ! -f "${workload_root}/COMPLETE" ]]; then
    printf '[%s] workload %s returned without COMPLETE marker\n' \
      "$(date -Is)" "${workload}" >&2
    exit 1
  fi
  printf '[%s] END workload %s\n' "$(date -Is)" "${workload}"
done

printf 'complete\n' > "${RESULT_ROOT}/COMPLETE"
printf '[%s] COMPLETE all sequential workloads\n' "$(date -Is)"
