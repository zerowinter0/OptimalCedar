#!/usr/bin/env bash
# Reproduce the stable W=8 paper optimizer matrix for workloads whose legacy
# figure cells had only one measured execution.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_DIR="${REPO_ROOT}/evaluation/chapter6_experiments"
OUTPUT_ROOT="${BASE_DIR}/formal_results/paper_optimizer_w8"
SOURCE_PROFILES="${BASE_DIR}/formal_results/cost_model_wall_profiles/20260729T_current/profiles"
WORKLOADS="llava_pretrain,redpajama_c4,stackexchange,simclrv2,simclrv2_cache,wikitext103,wikitext103_cache"

cd "${REPO_ROOT}"
source env/bin/activate

mkdir -p "${OUTPUT_ROOT}/profiles" "${OUTPUT_ROOT}/workloads"
for workload in ${WORKLOADS//,/ }; do
  source_profile="${SOURCE_PROFILES}/${workload}.yaml"
  [[ -f "${source_profile}" ]] || {
    echo "Missing formal profile: ${source_profile}" >&2
    exit 1
  }
  cp -p "${source_profile}" "${OUTPUT_ROOT}/profiles/${workload}.yaml"
done

exec env \
  PROFILE_DIR="${OUTPUT_ROOT}/profiles" \
  MATRIX_OUTPUT_ROOT="${OUTPUT_ROOT}/workloads" \
  CPU_BUDGET=64 \
  LOCAL_WORKERS=8 \
  REPEATS=3 \
  OPTIMIZER_PLAN_TIMEOUT_SEC=300 \
  OPTIMIZER_SET=complete \
  RESUME_EXISTING="${RESUME_EXISTING:-0}" \
  bash "${BASE_DIR}/run_formal_plan_and_matrix.sh" \
    --workloads "${WORKLOADS}"
