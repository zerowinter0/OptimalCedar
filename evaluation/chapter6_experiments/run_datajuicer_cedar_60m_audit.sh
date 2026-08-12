#!/usr/bin/env bash
# Re-audit original Cedar and DP two-stage plans under the current threshold.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_ROOT="${DIVERSE_ROOT:-${REPO_ROOT}/outputs/chapter6_experiments/datajuicer_diverse_workloads}"
AUDIT_ROOT="${BASE_ROOT}/cedar_60m_audit"
PROFILE_DIR="${REPO_ROOT}/evaluation/chapter6_experiments/formal_results/paper_artifacts/optimizer/profiles"

cd "${REPO_ROOT}"
source env/bin/activate
export DJ_CANDIDATE_DIVERSE_PROTOCOL="evaluation/chapter6_experiments/DATA_JUICER_DIVERSE_WORKLOADS.md"

run_audit_batch() {
  local run_id="$1" workloads="$2" outputs="$3"
  local run_root="${AUDIT_ROOT}/datajuicer_candidate_runs/${run_id}"
  if [[ -f "${run_root}/candidate_matrix.log" ]] &&
     grep -Fq "Candidate run complete: ${run_root}" \
       "${run_root}/candidate_matrix.log"; then
    echo "[$(date -Is)] REUSE completed Cedar audit ${run_id}"
    return
  fi
  if [[ -e "${run_root}" ]]; then
    resume=1
    echo "[$(date -Is)] RESUME incomplete optimizer audit ${run_id}"
  else
    resume=0
  fi
  DJ_CANDIDATE_OUTPUT_ROOT="${AUDIT_ROOT}" \
  DJ_CANDIDATE_PROFILE_DIR="${PROFILE_DIR}" \
  DJ_CANDIDATE_RUN_ID="${run_id}" \
  DJ_CANDIDATE_WORKLOADS="${workloads}" \
  DJ_CANDIDATE_OPTIMIZERS=optimizer,dp_two_stage_optimizer \
  DJ_CANDIDATE_REPEATS=3 \
  DJ_CANDIDATE_OUTPUTS="${outputs}" \
  DJ_OPTIMIZER_TIMEOUT_SEC=3600 \
  DJ_EXECUTION_TIMEOUT_SEC=3600 \
  DJ_PROFILE_TIMEOUT_SEC=3600 \
  DJ_REUSE_EXISTING_PROFILES=1 \
  DJ_CANDIDATE_RESUME="${resume}" \
  bash evaluation/chapter6_experiments/run_datajuicer_candidate_matrix.sh
}

run_audit_batch europarl_2500 pile_europarl 2500
run_audit_batch heldout_20000 \
  pile_hackernews,pile_pubmed_abstracts,pile_uspto_backgrounds 20000

bash evaluation/chapter6_experiments/run_datajuicer_two_stage_modal_audit.sh

python evaluation/chapter6_experiments/audit_datajuicer_candidate_scale.py \
  --output "${BASE_ROOT}/data_scale.json"
python evaluation/chapter6_experiments/analyze_datajuicer_diverse_workloads.py \
  --root "${BASE_ROOT}" \
  --json-output "${BASE_ROOT}/final_selection.json" \
  --markdown-output "${BASE_ROOT}/final_selection.md"
python evaluation/chapter6_experiments/plot_datajuicer_diverse_workloads.py \
  --report "${BASE_ROOT}/final_selection.json" \
  --output-dir "${BASE_ROOT}/figures"
python evaluation/chapter6_experiments/archive_datajuicer_diverse_results.py \
  --root "${BASE_ROOT}"

echo "[$(date -Is)] Cedar 60-minute plan audit complete"
