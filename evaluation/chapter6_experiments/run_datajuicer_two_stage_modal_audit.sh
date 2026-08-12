#!/usr/bin/env bash
# Add missing original-Cedar and DP two-stage comparators to reused evidence.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_ROOT="${DIVERSE_ROOT:-${REPO_ROOT}/outputs/chapter6_experiments/datajuicer_diverse_workloads}"
AUDIT_ROOT="${BASE_ROOT}/two_stage_modal_audit"
CANONICAL_PROFILES="${REPO_ROOT}/evaluation/chapter6_experiments/formal_results/paper_artifacts/optimizer/profiles"

cd "${REPO_ROOT}"
source env/bin/activate

run_audit() {
  local workload="$1" profile_dir="$2"
  local workload_root="${AUDIT_ROOT}/${workload}"
  local resume=0
  if [[ -f "${workload_root}/nohup.log" ]] &&
     grep -Fq "COMPLETE ${workload}" "${workload_root}/nohup.log"; then
    echo "[$(date -Is)] REUSE completed two-stage audit ${workload}"
    return
  fi
  [[ -e "${workload_root}" ]] && resume=1
  MATRIX_OUTPUT_ROOT="${AUDIT_ROOT}" \
  PROFILE_DIR="${profile_dir}" \
  REPEATS=3 \
  OPTIMIZER_SET=legacy_and_two_stage \
  OPTIMIZER_PLAN_TIMEOUT_SEC=3600 \
  EXECUTION_TIMEOUT_SEC=3600 \
  RESUME_EXISTING="${resume}" \
  bash evaluation/chapter6_experiments/run_formal_plan_and_matrix.sh \
    --workloads "${workload}"
}

run_audit llava_pretrain "${CANONICAL_PROFILES}"

echo "[$(date -Is)] Reused-modal missing-optimizer audit complete"
