#!/usr/bin/env bash
# Formal remote-Ray evaluation for the separate local/SMP and remote/Ray pools.
# Each workload is fully profiled immediately before its three-round matrix.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/chapter6_experiments/remote_separate_resource_pools_formal_v1}"
PROFILE_DIR="${OUTPUT_ROOT}/profiles"
MATRIX_ROOT="${OUTPUT_ROOT}/formal_runs"
RAY_ADDRESS="${RAY_ADDRESS:-172.23.166.105:6379}"
REMOTE_RAY_NODE_IP="${REMOTE_RAY_NODE_IP:-172.23.166.105}"
REMOTE_RAY_RESOURCE="${REMOTE_RAY_RESOURCE:-cedar_remote}"

# GeneralVideo is deliberately last because it has the highest setup/runtime
# cost and is the only workload that requires the remote GPU.
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
if [[ -n "${WORKLOADS_OVERRIDE:-}" ]]; then
  IFS=',' read -r -a WORKLOADS <<< "${WORKLOADS_OVERRIDE}"
fi

cd "${REPO_ROOT}"
# shellcheck source=/dev/null
source env/bin/activate
mkdir -p "${OUTPUT_ROOT}" "${PROFILE_DIR}" "${MATRIX_ROOT}"
printf '%s\n' "$$" > "${OUTPUT_ROOT}/runner.pid"
printf 'RUNNING phase=setup time=%s\n' "$(date -Is)" > "${OUTPUT_ROOT}/STATUS"

on_error() {
  local exit_code="$?"
  printf 'FAILED exit_code=%s time=%s\n' \
    "${exit_code}" "$(date -Is)" > "${OUTPUT_ROOT}/STATUS"
  exit "${exit_code}"
}
trap on_error ERR

cat > "${OUTPUT_ROOT}/PROTOCOL.md" <<'EOF'
# Formal separate-resource-pool experiment

The local machine contributes a 64-CPU pool shared by eight in-process workers
and SMP processes. The remote Ray node contributes a separate 64-CPU pool and
one GPU. Ray actors request real remote CPUs; Ray and SMP widths are optimized
against their own capacities. For every workload, the complete profile is
freshly measured and immediately followed by the formal matrix.

DJ, Pecan, Simple-DP, and the revised DP use identical profiles, data, switches,
and the configured common repeat count. PICO
models each resource lane additively and minimizes max(L,R,S,G), where R is
the sum of all Ray-stage services and S is the sum of all SMP-stage services.
Optimization plus first execution has a one-hour task deadline. Revised-DP
plan generation additionally
has a strict five-minute internal search deadline and returns the best feasible
incumbent found by then; a six-minute process guard leaves setup/teardown grace.
GeneralVideo runs last.
EOF
{
  printf '\nrepeat_count=%s\n' "${REPEATS:-3}"
  printf 'workloads='
  printf '%s ' "${WORKLOADS[@]}"
  printf '\n'
} >> "${OUTPUT_ROOT}/PROTOCOL.md"

export RAY_ADDRESS REMOTE_RAY_NODE_IP REMOTE_RAY_RESOURCE
export REMOTE_RAY_ONLY=1
export CPU_BUDGET=64
export RAY_CPU_BUDGET=64
export LOCAL_WORKERS=8
export PROFILE_DIR
export MATRIX_OUTPUT_ROOT="${MATRIX_ROOT}"
export OPTIMIZER_SET=quick_model_gate
export REPEATS="${REPEATS:-3}"
export TASK_TIMEOUT_SEC=3600
# The optimizer owns the exact five-minute search deadline and returns its
# best feasible incumbent.  The process guard includes interpreter/Ray setup
# and therefore needs a small grace period so it cannot kill that return path.
export CEDAR_DP_OPTIMIZATION_TIME_LIMIT_SEC=300
export CEDAR_DP_SEARCH_MODE=auto
export DP_PLAN_TIMEOUT_SEC=360
export CH6_PROFILE_TIMEOUT_SEC="${CH6_PROFILE_TIMEOUT_SEC:-10800}"
export RESUME_EXISTING=0
export CEDAR_DP_PARETO_EPSILON=0
export CEDAR_DP_FRONTIER_CAP=0
export CEDAR_DP_MASK_LAYER_WORKERS="${CEDAR_DP_MASK_LAYER_WORKERS:-1}"

for workload in "${WORKLOADS[@]}"; do
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
