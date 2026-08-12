#!/usr/bin/env bash
# Serialize the Cedar timeout audit after the candidate screening matrix.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_ROOT="${DIVERSE_ROOT:-${REPO_ROOT}/outputs/chapter6_experiments/datajuicer_diverse_workloads}"
SCREEN_PID_FILE="${BASE_ROOT}/diverse_screening.pid"
SCREEN_LOG="${BASE_ROOT}/diverse_screening.nohup.log"

cd "${REPO_ROOT}"
source env/bin/activate

if [[ -f "${SCREEN_PID_FILE}" ]]; then
  screen_pid="$(<"${SCREEN_PID_FILE}")"
  while kill -0 "${screen_pid}" 2>/dev/null; do
    grep -Fq "Diverse candidate screening complete" "${SCREEN_LOG}" && break
    process_state="$(awk '{print $3}' "/proc/${screen_pid}/stat" 2>/dev/null || true)"
    [[ "${process_state}" == "Z" || -z "${process_state}" ]] && break
    sleep 60
  done
fi

if ! grep -Fq "Diverse candidate screening complete" "${SCREEN_LOG}"; then
  echo "Candidate screening did not complete successfully; audit not started." >&2
  exit 1
fi

# The first screening process may have started before this queue script was
# updated with conditional three-repeat formalization. Re-entering the
# idempotent driver reuses completed one-run screens, formalizes any new 1.20x
# winner, and always completes the selected Alpaca-CoT matrix.
bash evaluation/chapter6_experiments/run_datajuicer_diverse_screening.sh

bash evaluation/chapter6_experiments/run_datajuicer_cedar_60m_audit.sh
