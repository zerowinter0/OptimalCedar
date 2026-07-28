#!/usr/bin/env bash
# Wait for the first candidate batch, then prepare and run the registered
# selectivity-aware extension without overlapping formal measurements.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_DIR="${REPO_ROOT}/evaluation/chapter6_experiments"
FORMAL_ROOT="${BASE_DIR}/formal_results"
CURRENT_PID="${DJ_CURRENT_CANDIDATE_PID:-379823}"
PRIOR_ROOT="${DJ_PRIOR_CANDIDATE_ROOT:-${FORMAL_ROOT}/datajuicer_candidate_runs/20260725T051352Z}"
PRIOR_CODE_DIR="${PRIOR_ROOT}/redpajama_code"
PRIOR_STOP_AFTER_STATUS="${DJ_PRIOR_STOP_AFTER_STATUS:-}"
CHAIN_ID="${DJ_EXTENSION_CHAIN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
EXTENSION_WORKLOADS="pile_hackernews,pile_pubmed_abstracts,pile_freelaw,pile_uspto_backgrounds"
RESUME_AFTER_CODE="${DJ_RESUME_AFTER_CODE:-0}"

cd "${REPO_ROOT}"
# shellcheck source=/dev/null
source env/bin/activate
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

is_current_batch_running() {
  [[ -r "/proc/${CURRENT_PID}/cmdline" ]] || return 1
  tr '\0' ' ' < "/proc/${CURRENT_PID}/cmdline" |
    grep -q "run_datajuicer_candidate_matrix.sh"
}

prior_stop_status_ready() {
  [[ -n "${PRIOR_STOP_AFTER_STATUS}" ]] || return 1
  local status_path
  status_path="${PRIOR_ROOT}/pile_europarl/status/${PRIOR_STOP_AFTER_STATUS}.json"
  [[ -f "${status_path}" ]] || return 1
  python - "${status_path}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    status = json.load(stream).get("status")
if status != "infeasible_timeout":
    raise SystemExit(
        f"Refusing early transition: expected infeasible_timeout, got {status!r}"
    )
PY
}

record_early_transition() {
  local marker="${PRIOR_ROOT}/pile_europarl/status/early_transition.json"
  python - "${marker}" "${PRIOR_STOP_AFTER_STATUS}" <<'PY'
import datetime
import json
import pathlib
import sys

path, trigger = sys.argv[1:]
payload = {
    "status": "candidate_failed_early_transition",
    "trigger": trigger,
    "reason": (
        "Data-Juicer and DP-Cedar each reached the frozen execution limit; "
        "remaining EuroParl cells are missing and cannot count as evidence"
    ),
    "recorded_at_utc": datetime.datetime.now(
        datetime.timezone.utc
    ).isoformat(),
}
pathlib.Path(path).write_text(
    json.dumps(payload, indent=2) + "\n",
    encoding="utf-8",
)
PY
}

stop_legacy_batch_before_code() {
  echo "[$(date -Is)] Stopping legacy batch before its size-only Code branch"
  # The parent may currently be waiting on a timeout child. Signal both the
  # direct children and the exact validated runner PID; the fresh runner
  # below performs a full Ray shutdown before profiling.
  pkill -TERM -P "${CURRENT_PID}" 2>/dev/null || true
  kill -TERM "${CURRENT_PID}" 2>/dev/null || true
  for _ in {1..30}; do
    is_current_batch_running || return 0
    sleep 1
  done
  pkill -KILL -P "${CURRENT_PID}" 2>/dev/null || true
  kill -KILL "${CURRENT_PID}" 2>/dev/null || true
}

if [[ "${RESUME_AFTER_CODE}" != "1" ]]; then
echo "[$(date -Is)] Waiting for first candidate batch pid=${CURRENT_PID}"
while is_current_batch_running; do
  if prior_stop_status_ready; then
    echo "[$(date -Is)] ${PRIOR_STOP_AFTER_STATUS} reached the frozen 3600 s limit"
    echo "[$(date -Is)] Retaining EuroParl as a failed denominator workload"
    record_early_transition
    stop_legacy_batch_before_code
    break
  fi
  if [[ -e "${PRIOR_CODE_DIR}/metadata.txt" ]]; then
    stop_legacy_batch_before_code
    break
  fi
  sleep 5
done
echo "[$(date -Is)] EuroParl batch ended; regenerating Code profile"

CODE_RUN_ID="${CHAIN_ID}_code_selectivity"
env \
  DJ_CANDIDATE_RUN_ID="${CODE_RUN_ID}" \
  DJ_CANDIDATE_WORKLOADS="redpajama_code" \
  bash "${BASE_DIR}/run_datajuicer_candidate_matrix.sh"
else
  CODE_RUN_ID="${CHAIN_ID}_code_selectivity"
  if [[ ! -f "${FORMAL_ROOT}/datajuicer_candidate_runs/${CODE_RUN_ID}/dp_20pct_report.json" ]]; then
    echo "Cannot resume: completed Code run is missing for ${CODE_RUN_ID}" >&2
    exit 2
  fi
fi

env DJ_EXTENSION_PREP_RUN_ID="${CHAIN_ID}" \
  bash "${BASE_DIR}/prepare_datajuicer_extension.sh"

EXTENSION_RUN_ID="${CHAIN_ID}_selectivity"
env \
  DJ_CANDIDATE_RUN_ID="${EXTENSION_RUN_ID}" \
  DJ_CANDIDATE_WORKLOADS="${EXTENSION_WORKLOADS}" \
  DJ_CANDIDATE_RESUME="${RESUME_AFTER_CODE}" \
  DJ_REUSE_PROFILE_RUN_ID="${DJ_REUSE_PROFILE_RUN_ID:-}" \
  bash "${BASE_DIR}/run_datajuicer_candidate_matrix.sh"

CODE_ROOT="${FORMAL_ROOT}/datajuicer_candidate_runs/${CODE_RUN_ID}"
EXTENSION_ROOT="${FORMAL_ROOT}/datajuicer_candidate_runs/${EXTENSION_RUN_ID}"
PRIOR_LEDGER_ROOT="${FORMAL_ROOT}/datajuicer_candidate_runs/${CHAIN_ID}_prior_ledger"
mkdir -p "${PRIOR_LEDGER_ROOT}"
ln -sfn "${PRIOR_ROOT}/pile_europarl" \
  "${PRIOR_LEDGER_ROOT}/pile_europarl"
python "${BASE_DIR}/analyze_dp_20pct_goal.py" \
  --candidate-root "${PRIOR_LEDGER_ROOT}" \
  --candidate-root "${CODE_ROOT}" \
  --candidate-root "${EXTENSION_ROOT}" \
  --require-all-registered \
  --json-output "${FORMAL_ROOT}/dp_20pct_goal_latest.json" \
  --markdown-output "${FORMAL_ROOT}/dp_20pct_goal_latest.md"
echo "[$(date -Is)] Combined goal audit complete"
