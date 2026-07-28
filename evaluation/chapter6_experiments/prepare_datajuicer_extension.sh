#!/usr/bin/env bash
# Download and source-check the complete pre-registered extension batch.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FORMAL_ROOT="${REPO_ROOT}/evaluation/chapter6_experiments/formal_results"
RUN_ID="${DJ_EXTENSION_PREP_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG="${FORMAL_ROOT}/datajuicer_extension_prepare_${RUN_ID}.log"
WORKLOADS=(
  pile_hackernews
  pile_pubmed_abstracts
  pile_freelaw
  pile_uspto_backgrounds
)
FEASIBILITY_TIMEOUT_SEC=3600

cd "${REPO_ROOT}"
# shellcheck source=/dev/null
source env/bin/activate
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
mkdir -p "${FORMAL_ROOT}"
exec > >(tee -a "${LOG}") 2>&1

download_with_retries() {
  local workload="$1"
  local attempt
  for attempt in 1 2 3; do
    if python evaluation/pipelines/download_registered_pile_recipe.py \
      "${workload}"; then
      return 0
    fi
    if (( attempt == 3 )); then
      echo "[$(date -Is)] Download failed after ${attempt} attempts: ${workload}" >&2
      return 1
    fi
    echo "[$(date -Is)] Download attempt ${attempt} failed; retrying ${workload}" >&2
    sleep $((attempt * 10))
  done
}

dataset_path() {
  case "$1" in
    pile_hackernews)
      printf '%s\n' "datasets/pile_hackernews/pile-hackernews-raw-100000.jsonl"
      ;;
    pile_pubmed_abstracts)
      printf '%s\n' "datasets/pile_pubmed_abstracts/pile-pubmed-abstracts-raw-100000.jsonl"
      ;;
    pile_freelaw)
      printf '%s\n' "datasets/pile_freelaw/pile-freelaw-raw-100000.jsonl"
      ;;
    pile_uspto_backgrounds)
      printf '%s\n' "datasets/pile_uspto_backgrounds/pile-uspto-backgrounds-raw-100000.jsonl"
      ;;
    *) return 2 ;;
  esac
}

download_is_verified() {
  local workload="$1" data metadata
  data="$(dataset_path "${workload}")"
  metadata="${data%.jsonl}.metadata.json"
  [[ -f "${data}" && -f "${metadata}" ]] || return 1
  python - "${data}" "${metadata}" "${workload}" <<'PY'
import hashlib
import json
import pathlib
import sys

data_path = pathlib.Path(sys.argv[1])
metadata_path = pathlib.Path(sys.argv[2])
workload = sys.argv[3]
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
if metadata.get("workload") != workload:
    raise SystemExit(1)
if int(metadata.get("records", 0)) != 100_000:
    raise SystemExit(1)
if int(metadata.get("bytes", -1)) != data_path.stat().st_size:
    raise SystemExit(1)
digest = hashlib.sha256()
with data_path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
if digest.hexdigest() != metadata.get("sha256"):
    raise SystemExit(1)
PY
}

feasibility_path() {
  local data
  data="$(dataset_path "$1")"
  printf '%s\n' "${data%.jsonl}.feasibility.json"
}

feasibility_is_recorded() {
  local workload="$1" evidence
  evidence="$(feasibility_path "${workload}")"
  [[ -f "${evidence}" ]] || return 1
  python - "${evidence}" "${workload}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    result = json.load(stream)
if result.get("workload") != sys.argv[2]:
    raise SystemExit(1)
if result.get("status") not in {
    "feasible", "infeasible", "benchmarkable_timeout"
}:
    raise SystemExit(1)
PY
}

echo "[$(date -Is)] Preparing registered Data-Juicer extension batch"
for workload in "${WORKLOADS[@]}"; do
  if download_is_verified "${workload}"; then
    echo "[$(date -Is)] Reusing verified download ${workload}"
  else
    echo "[$(date -Is)] Download ${workload}"
    download_with_retries "${workload}"
  fi
  if feasibility_is_recorded "${workload}"; then
    echo "[$(date -Is)] Reusing recorded feasibility ${workload}"
    continue
  fi
  echo "[$(date -Is)] Source-only feasibility ${workload}"
  started_at="$(date +%s)"
  if timeout --signal=TERM --kill-after=30s "${FEASIBILITY_TIMEOUT_SEC}" \
    python evaluation/pipelines/check_pile_recipe_feasibility.py \
    "${workload}"; then
    echo "[$(date -Is)] Source-feasible ${workload}"
  else
    status=$?
    if [[ "${status}" -eq 124 || "${status}" -eq 137 ]]; then
      elapsed=$(($(date +%s) - started_at))
      python evaluation/pipelines/check_pile_recipe_feasibility.py \
        "${workload}" --record-timeout --elapsed-seconds "${elapsed}"
      echo "[$(date -Is)] Serial feasibility timed out for ${workload}; retaining it for the formal benchmark"
    elif [[ "${status}" -eq 3 ]]; then
      echo "[$(date -Is)] Source-infeasible ${workload}; retaining it in the formal denominator"
    else
      echo "[$(date -Is)] Feasibility check failed unexpectedly for ${workload} (status=${status})" >&2
      exit "${status}"
    fi
  fi
done
echo "[$(date -Is)] Extension preparation complete"
printf '%s\n' "${LOG}" > \
  "${FORMAL_ROOT}/datajuicer_extension_prepare_latest.txt"
