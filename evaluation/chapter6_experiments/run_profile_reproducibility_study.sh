#!/usr/bin/env bash
# Controlled profile-duration and independent-repeat experiment.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STUDY_ROOT="${PROFILE_REPRO_STUDY_ROOT:-${REPO_ROOT}/outputs/chapter6_experiments/profile_reproducibility_study}"
PROFILE_DURATIONS="${PROFILE_REPRO_DURATIONS:-10 60}"
REPEATS="${PROFILE_REPRO_REPEATS:-3}"
WORKLOADS="${PROFILE_REPRO_WORKLOADS:-alpaca_cot,stackexchange,general_video_refine}"
PROFILE_TIMEOUT_SEC="${PROFILE_REPRO_TIMEOUT_SEC:-14400}"

[[ "${REPEATS}" =~ ^[1-9][0-9]*$ ]] || { echo "PROFILE_REPRO_REPEATS must be positive" >&2; exit 2; }
[[ "${PROFILE_TIMEOUT_SEC}" =~ ^[1-9][0-9]*$ ]] || { echo "PROFILE_REPRO_TIMEOUT_SEC must be positive" >&2; exit 2; }

cd "${REPO_ROOT}"
source env/bin/activate
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
mkdir -p "${STUDY_ROOT}"

cat > "${STUDY_ROOT}/PROTOCOL.md" <<EOF
# Profile reproducibility study

- Workloads: Alpaca-CoT, StackExchange, and General Video Refine.
- Per-stage durations: ${PROFILE_DURATIONS} seconds.
- Independent repeats per duration and workload: ${REPEATS}.
- Every cell is a fresh complete Cedar baseline/Ray/SMP profile with one local
  worker and one actor/process per profiled stage; frozen profiles are not reused.
- Acceptance is pre-registered in \`analyze_profile_reproducibility.py\`:
  baseline throughput CV <=5%, backend-cost median CV <=10%, backend-cost P90
  CV <=20%, and at least 90% of backend measurements with RSE <=10%.
EOF

for duration in ${PROFILE_DURATIONS}; do
  [[ "${duration}" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid duration: ${duration}" >&2; exit 2; }
  for ((repeat = 1; repeat <= REPEATS; repeat++)); do
    cell="${STUDY_ROOT}/duration_${duration}/repeat_${repeat}"
    marker="${cell}/COMPLETE"
    if [[ -f "${marker}" ]]; then
      echo "[$(date -Is)] REUSE duration=${duration} repeat=${repeat}"
      continue
    fi
    mkdir -p "${cell}"
    echo "[$(date -Is)] START duration=${duration} repeat=${repeat}"
    CEDAR_PROFILE_TIME_SEC="${duration}" \
    CH6_RESULT_ROOT="${cell}" \
    CH6_PROFILE_ROOT="${cell}/profiles" \
    CH6_PROFILE_RUN_ID="fresh_profile" \
    CH6_PROFILE_TIMEOUT_SEC="${PROFILE_TIMEOUT_SEC}" \
    INCREMENTAL_BACKEND_COMPUTE=0 \
      bash evaluation/chapter6_experiments/run_formal_profiles.sh \
        --workloads "${WORKLOADS}" \
        > "${cell}/cell.log" 2>&1
    printf 'stage_duration_sec=%s\nrepeat=%s\nworkloads=%s\n' \
      "${duration}" "${repeat}" "${WORKLOADS}" > "${cell}/metadata.txt"
    touch "${marker}"
    echo "[$(date -Is)] COMPLETE duration=${duration} repeat=${repeat}"
  done
done

python evaluation/chapter6_experiments/analyze_profile_reproducibility.py \
  --study-root "${STUDY_ROOT}"
touch "${STUDY_ROOT}/STUDY_COMPLETE"
echo "[$(date -Is)] PROFILE REPRODUCIBILITY STUDY COMPLETE"
