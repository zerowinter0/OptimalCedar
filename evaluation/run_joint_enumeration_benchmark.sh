#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT_DIR=${1:-"${REPO_ROOT}/evaluation/chapter6_experiments/formal_results/paper_artifacts/joint_enumeration"}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-3600}
REPEATS=${REPEATS:-3}

mkdir -p "${OUT_DIR}"

for n in $(seq 1 20); do
  for repeat in $(seq 1 "${REPEATS}"); do
    output="${OUT_DIR}/n${n}_repeat${repeat}.json"
    echo "START n=${n} repeat=${repeat} timeout=${TIMEOUT_SECONDS}s"
    set +e
    timeout --signal=TERM "${TIMEOUT_SECONDS}s" \
      python "${REPO_ROOT}/evaluation/benchmark_joint_enumeration.py" \
        --operators "${n}" \
        --repeat "${repeat}" \
        --output "${output}"
    status=$?
    set -e
    if [[ "${status}" -eq 124 ]]; then
      touch "${OUT_DIR}/n${n}_timeout_${TIMEOUT_SECONDS}s"
      echo "TIMEOUT n=${n} repeat=${repeat} after=${TIMEOUT_SECONDS}s"
      exit 0
    fi
    if [[ "${status}" -ne 0 ]]; then
      echo "FAILED n=${n} repeat=${repeat} status=${status}"
      exit "${status}"
    fi
    echo "DONE n=${n} repeat=${repeat}"
  done
done
