#!/usr/bin/env bash
# Formally rerun EuroParl and FreeLaw without modifying the canonical archive.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_DIR="${REPO_ROOT}/evaluation/chapter6_experiments"
ARCHIVE_ROOT="${BASE_DIR}/formal_results/paper_optimizer_w8"
OUTPUT_ROOT="${OUTPUT_ROOT:-${BASE_DIR}/paper_candidate_runs}"
PROFILE_DIR="${OUTPUT_ROOT}/profiles"
RUN_ID="europarl_freelaw_w8"

cd "${REPO_ROOT}"
source env/bin/activate

mkdir -p "${PROFILE_DIR}"
cp "${ARCHIVE_ROOT}/profiles/pile_europarl.yaml" \
  "${PROFILE_DIR}/pile_europarl.yaml"

DJ_CANDIDATE_OUTPUT_ROOT="${OUTPUT_ROOT}" \
DJ_CANDIDATE_PROFILE_DIR="${PROFILE_DIR}" \
DJ_CANDIDATE_RUN_ID="${RUN_ID}" \
DJ_CANDIDATE_WORKLOADS="pile_europarl,pile_freelaw" \
DJ_REUSE_EXISTING_PROFILES=1 \
DJ_CANDIDATE_RESUME="${DJ_CANDIDATE_RESUME:-0}" \
bash "${BASE_DIR}/run_datajuicer_candidate_matrix.sh"
