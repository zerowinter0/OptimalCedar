#!/usr/bin/env bash
# Full remote profiling followed by a one-round reduced-data execution gate.
# Profiling remains identical to the formal protocol; only measured execution
# sample counts and repetition count are reduced.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/chapter6_experiments/remote_single_budget_exact_quick_gate_v1}"
PROFILE_DIR="${OUTPUT_ROOT}/profiles"
FORMAL_ROOT="${OUTPUT_ROOT}/formal_runs"
RAY_ADDRESS="${RAY_ADDRESS:-172.23.166.105:6379}"
REMOTE_RAY_NODE_IP="${REMOTE_RAY_NODE_IP:-172.23.166.105}"
REMOTE_RAY_RESOURCE="${REMOTE_RAY_RESOURCE:-cedar_remote}"

# Keep GeneralVideo last because it is the slowest and depends on GPU health.
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
mkdir -p "${PROFILE_DIR}" "${FORMAL_ROOT}"
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
# Remote exact-DP quick validation

This run freshly profiles every workload with the complete formal profiling
protocol. Profiling duration, backend coverage, Ray actor/SMP process count,
and boundary calibration are not reduced. After each workload is profiled, all
DJ, Pecan, Simple-DP, and the current DP are measured on the same reduced
execution input for one round. Cedar and both two-stage optimizers are excluded.
Thus the run is a directional validation of the current DP rather than a
replacement for the three-round, full-data paper matrix.

GeneralVideo is deliberately profiled and executed last.
EOF

export RAY_ADDRESS REMOTE_RAY_NODE_IP REMOTE_RAY_RESOURCE
export REMOTE_RAY_ONLY=1
export CPU_BUDGET=64
export RAY_CPU_BUDGET=64
export LOCAL_WORKERS=8
export MATRIX_OUTPUT_ROOT="${FORMAL_ROOT}"
export PROFILE_DIR
export OPTIMIZER_SET=quick_model_gate
export REPEATS=1
export TASK_TIMEOUT_SEC="${TASK_TIMEOUT_SEC:-3600}"
export RESUME_EXISTING=0
export CEDAR_DP_PARETO_EPSILON=0
export CEDAR_DP_FRONTIER_CAP=0
export CEDAR_DP_MASK_LAYER_WORKERS="${CEDAR_DP_MASK_LAYER_WORKERS:-32}"

run_execution() {
  local workload="$1"
  env \
    COMMONVOICE_SAMPLES="${COMMONVOICE_SAMPLES:-5000}" \
    ALPACA_COT_SAMPLES="${ALPACA_COT_SAMPLES:-5000}" \
    STACKEXCHANGE_SAMPLES="${STACKEXCHANGE_SAMPLES:-5000}" \
    PILE_EUROPARL_SAMPLES="${PILE_EUROPARL_SAMPLES:-500}" \
    PILE_HACKERNEWS_SAMPLES="${PILE_HACKERNEWS_SAMPLES:-1000}" \
    PILE_PUBMED_SAMPLES="${PILE_PUBMED_SAMPLES:-1000}" \
    PILE_USPTO_SAMPLES="${PILE_USPTO_SAMPLES:-1000}" \
    REDPAJAMA_CODE_SAMPLES="${REDPAJAMA_CODE_SAMPLES:-1000}" \
    GENERAL_VIDEO_REFINE_SAMPLES="${GENERAL_VIDEO_REFINE_SAMPLES:-10}" \
    bash evaluation/chapter6_experiments/run_formal_plan_and_matrix.sh \
      --workloads "${workload}"
}

for workload in "${WORKLOADS[@]}"; do
  printf 'RUNNING workload=%s phase=profile time=%s\n' \
    "${workload}" "$(date -Is)" > "${OUTPUT_ROOT}/STATUS"

  # COMMONVOICE_SAMPLES is intentionally removed: the full profiler's default
  # (240000) must not inherit the reduced measured-execution sample count.
  env -u COMMONVOICE_SAMPLES \
    CH6_RESULT_ROOT="${OUTPUT_ROOT}" \
    CH6_PROFILE_ROOT="${PROFILE_DIR}" \
    CH6_PROFILE_RUN_ID="${workload}_profile" \
    bash evaluation/chapter6_experiments/run_formal_profiles.sh \
      --workloads "${workload}"

  printf 'RUNNING workload=%s phase=execution time=%s\n' \
    "${workload}" "$(date -Is)" > "${OUTPUT_ROOT}/STATUS"
  run_execution "${workload}"
done

touch "${OUTPUT_ROOT}/COMPLETE"
printf 'COMPLETE time=%s\n' "$(date -Is)" > "${OUTPUT_ROOT}/STATUS"
