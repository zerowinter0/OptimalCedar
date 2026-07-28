#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ "$#" -lt 1 ]]; then
  echo "usage: run_workload.sh WORKLOAD [OUTPUT_DIR] [DATASET_KWARGS_JSON]" >&2
  exit 2
fi

WORKLOAD="$1"
OUTPUT_DIR="${2:-${REPO_ROOT}/evaluation/plumber/results/${WORKLOAD}}"
DATASET_KWARGS="${3:-{}}"
BASE_WORKLOAD="${WORKLOAD%_cache}"

case "${BASE_WORKLOAD}" in
  coco|commonvoice|simclrv2|wikitext103)
    ;;
  *)
    echo "Plumber does not support workload ${WORKLOAD}" >&2
    exit 2
    ;;
esac

DATASET_FILE="${REPO_ROOT}/evaluation/pipelines/${BASE_WORKLOAD}/tf_dataset.py"
STATS_FILE="${OUTPUT_DIR}/stats.pb"
RESULTS_FILE="${OUTPUT_DIR}/result.json"
mkdir -p "${OUTPUT_DIR}"

python "${SCRIPT_DIR}/profile_pipeline.py" \
  --dataset-file "${DATASET_FILE}" \
  --stats-file "${STATS_FILE}" \
  --dataset-kwargs "${DATASET_KWARGS}" \
  --profile-samples "${PLUMBER_PROFILE_SAMPLES:-1000}" \
  --profile-seconds "${PLUMBER_PROFILE_SECONDS:-10}" \
  --parallelism "${PLUMBER_PROFILE_PARALLELISM:-1}" \
  --threadpool-size "${CPU_BUDGET:-64}"

python "${SCRIPT_DIR}/optimize_pipeline.py" \
  --stats-file "${STATS_FILE}" \
  --results-path "${RESULTS_FILE}" \
  --benchmark-seconds "${PLUMBER_BENCHMARK_SECONDS:-42}"
