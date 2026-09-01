#!/usr/bin/env bash
# Profile and run the ten-workload max(L, R, S, G) experiment with every
# Cedar Ray actor pinned to a remote-only Ray node. Each workload is fully
# profiled and measured before the runner advances to the next workload.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/chapter6_experiments/remote_ray_resource_family_sum_sequential_formal_v1}"
PROFILE_DIR="${OUTPUT_ROOT}/profiles"
FORMAL_ROOT="${OUTPUT_ROOT}/formal_runs"
RAY_ADDRESS="${RAY_ADDRESS:-172.23.166.105:6379}"
REMOTE_RAY_NODE_IP="${REMOTE_RAY_NODE_IP:-172.23.166.105}"
REMOTE_RAY_RESOURCE="${REMOTE_RAY_RESOURCE:-cedar_remote}"

WORKLOADS=(
  commonvoice
  simclrv2_cache
  alpaca_cot
  pile_europarl
  pile_hackernews
  pile_pubmed_abstracts
  pile_uspto_backgrounds
  redpajama_code
  stackexchange
  general_video_refine
)

cd "${REPO_ROOT}"
source env/bin/activate

mkdir -p "${OUTPUT_ROOT}" "${PROFILE_DIR}" "${FORMAL_ROOT}"
printf '%s\n' "$$" > "${OUTPUT_ROOT}/runner.pid"
printf 'RUNNING\n' > "${OUTPUT_ROOT}/STATUS"

on_error() {
  local exit_code="$?"
  printf 'FAILED exit_code=%s time=%s\n' \
    "${exit_code}" "$(date -Is)" > "${OUTPUT_ROOT}/STATUS"
  exit "${exit_code}"
}
trap on_error ERR

export RAY_ADDRESS REMOTE_RAY_NODE_IP REMOTE_RAY_RESOURCE
export REMOTE_RAY_ONLY=1
export CPU_BUDGET=64
export RAY_CPU_BUDGET=64
export LOCAL_WORKERS=8
export MATRIX_OUTPUT_ROOT="${FORMAL_ROOT}"
export PROFILE_DIR
export OPTIMIZER_SET=quick_model_gate
export REPEATS=3
export TASK_TIMEOUT_SEC=3600
export RESUME_EXISTING=0

for workload in "${WORKLOADS[@]}"; do
  # Remote execution changes serialization, network, and boundary costs, so
  # local-only profiles are intentionally not reused.
  printf 'RUNNING phase=profile workload=%s time=%s\n' \
    "${workload}" "$(date -Is)" > "${OUTPUT_ROOT}/STATUS"
  CH6_RESULT_ROOT="${OUTPUT_ROOT}" \
  CH6_PROFILE_ROOT="${PROFILE_DIR}" \
  CH6_PROFILE_RUN_ID="${workload}_profile" \
  bash evaluation/chapter6_experiments/run_formal_profiles.sh \
    --workloads "${workload}"

  printf 'RUNNING phase=formal workload=%s time=%s\n' \
    "${workload}" "$(date -Is)" > "${OUTPUT_ROOT}/STATUS"
  bash evaluation/chapter6_experiments/run_formal_plan_and_matrix.sh \
    --workloads "${workload}"
done

touch "${OUTPUT_ROOT}/COMPLETE"
printf 'COMPLETE time=%s\n' "$(date -Is)" > "${OUTPUT_ROOT}/STATUS"
