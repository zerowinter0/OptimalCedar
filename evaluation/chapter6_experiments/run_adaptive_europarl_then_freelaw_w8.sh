#!/usr/bin/env bash
# Adaptively size EuroParl, then give FreeLaw up to three hours to profile.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_DIR="${REPO_ROOT}/evaluation/chapter6_experiments"
OUTPUT_ROOT="${OUTPUT_ROOT:-${BASE_DIR}/paper_candidate_runs}"
PROFILE_DIR="${OUTPUT_ROOT}/profiles"
RUNNER="${BASE_DIR}/run_datajuicer_candidate_matrix.sh"
EUROPARL_START_OUTPUTS="${EUROPARL_START_OUTPUTS:-10000}"
MIN_OUTPUTS="${EUROPARL_MIN_OUTPUTS:-1}"
EXECUTION_TIMEOUT_SEC="${EXECUTION_TIMEOUT_SEC:-3600}"
FREELAW_PROFILE_TIMEOUT_SEC="${FREELAW_PROFILE_TIMEOUT_SEC:-10800}"
PROBE_OPTIMIZERS=(
  dj_optimizer
  dp_cedar_optimizer
  dp_optimizer
  pecan_optimizer
)

for numeric_setting in EUROPARL_START_OUTPUTS MIN_OUTPUTS \
  EXECUTION_TIMEOUT_SEC FREELAW_PROFILE_TIMEOUT_SEC; do
  numeric_value="${!numeric_setting}"
  if [[ ! "${numeric_value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${numeric_setting} must be a positive integer: ${numeric_value}" >&2
    exit 2
  fi
done

cd "${REPO_ROOT}"
# shellcheck source=/dev/null
source env/bin/activate
mkdir -p "${OUTPUT_ROOT}"

run_candidate() {
  local run_id="$1" workload="$2" repeats="$3" outputs="$4"
  local profile_timeout="$5"
  env \
    DJ_CANDIDATE_OUTPUT_ROOT="${OUTPUT_ROOT}" \
    DJ_CANDIDATE_PROFILE_DIR="${PROFILE_DIR}" \
    DJ_CANDIDATE_RUN_ID="${run_id}" \
    DJ_CANDIDATE_WORKLOADS="${workload}" \
    DJ_REUSE_EXISTING_PROFILES=1 \
    DJ_CANDIDATE_REPEATS="${repeats}" \
    DJ_CANDIDATE_OUTPUTS="${outputs}" \
    DJ_EXECUTION_TIMEOUT_SEC="${EXECUTION_TIMEOUT_SEC}" \
    DJ_PROFILE_TIMEOUT_SEC="${profile_timeout}" \
    bash "${RUNNER}"
}

probe_succeeded() {
  local run_id="$1" outputs="$2"
  local workload_root="${OUTPUT_ROOT}/datajuicer_candidate_runs/${run_id}/pile_europarl"
  local optimizer result
  for optimizer in "${PROBE_OPTIMIZERS[@]}"; do
    result="${workload_root}/results/round1__${optimizer}.json"
    [[ -f "${result}" ]] || return 1
    if ! python - "${result}" "${outputs}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    payload = json.load(stream)
expected = int(sys.argv[2])
times = payload.get("epoch_run_times")
samples = payload.get("epoch_num_samples")
if (
    not isinstance(times, list)
    or len(times) != 1
    or not isinstance(samples, list)
    or samples != [expected]
):
    raise SystemExit(1)
PY
    then
      return 1
    fi
  done
}

outputs="${EUROPARL_START_OUTPUTS}"
selected_outputs=
while (( outputs >= MIN_OUTPUTS )); do
  run_id="europarl_probe_n${outputs}"
  echo "[$(date -Is)] EuroParl one-round probe: ${outputs} outputs"
  run_candidate "${run_id}" pile_europarl 1 "${outputs}" 3600
  if probe_succeeded "${run_id}" "${outputs}"; then
    selected_outputs="${outputs}"
    break
  fi
  next_outputs=$((outputs / 2))
  if (( next_outputs < MIN_OUTPUTS || next_outputs == outputs )); then
    break
  fi
  echo "[$(date -Is)] Probe did not complete for every benchmarkable optimizer; halving to ${next_outputs}"
  outputs="${next_outputs}"
done

if [[ -z "${selected_outputs}" ]]; then
  echo "No feasible EuroParl output count found at or above ${MIN_OUTPUTS}." >&2
  exit 1
fi

printf '%s\n' "${selected_outputs}" > "${OUTPUT_ROOT}/europarl_selected_outputs.txt"
echo "[$(date -Is)] EuroParl selected output count: ${selected_outputs}"
run_candidate \
  "europarl_formal_n${selected_outputs}" \
  pile_europarl \
  3 \
  "${selected_outputs}" \
  3600

echo "[$(date -Is)] Starting FreeLaw with a ${FREELAW_PROFILE_TIMEOUT_SEC}s profile limit"
run_candidate \
  freelaw_profile3h_w8 \
  pile_freelaw \
  3 \
  20000 \
  "${FREELAW_PROFILE_TIMEOUT_SEC}"

echo "[$(date -Is)] Adaptive EuroParl and FreeLaw run finished"
