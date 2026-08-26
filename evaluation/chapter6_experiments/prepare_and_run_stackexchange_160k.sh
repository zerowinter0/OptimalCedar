#!/usr/bin/env bash
# Materialize the enlarged unique source, then run the 160k formal matrix.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATASET_PATH="${STACKEXCHANGE_DATASET_PATH:-datasets/stackexchange/redpajama-stackexchange-400000.jsonl}"
PARTIAL_PATH="${DATASET_PATH}.partial"
DOWNLOAD_WAIT_SEC="${STACKEXCHANGE_DOWNLOAD_WAIT_SEC:-900}"

cd "${REPO_ROOT}"
# shellcheck source=/dev/null
source env/bin/activate

if [[ ! -s "${DATASET_PATH}" && -e "${PARTIAL_PATH}" ]]; then
  start_time="${SECONDS}"
  while [[ ! -s "${DATASET_PATH}" && -e "${PARTIAL_PATH}" ]]; do
    if ((SECONDS - start_time >= DOWNLOAD_WAIT_SEC)); then
      echo "Timed out waiting for ${PARTIAL_PATH} to be published" >&2
      exit 1
    fi
    sleep 10
  done
fi

if [[ ! -s "${DATASET_PATH}" ]]; then
  python evaluation/pipelines/stackexchange/download_subset.py \
    --output "${DATASET_PATH}" --max-samples 400000
fi

exec bash evaluation/chapter6_experiments/run_stackexchange_160k_formal.sh
