#!/usr/bin/env bash
# Incrementally add worker-side compute timings, regenerate only DpOptimizer
# plans, and compare them with the plans that were active before this run.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_DIR="${REPO_ROOT}/evaluation/chapter6_experiments"
RUN_ID="${CH6_INCREMENTAL_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${BASE_DIR}/formal_results/incremental_backend_runs/${RUN_ID}"
BEFORE_DIR="${RUN_ROOT}/plans_before"
AFTER_DIR="${RUN_ROOT}/plans_after"
WORKLOADS="coco,commonvoice,commonvoice_cache,llava_pretrain,redpajama_c4,simclrv2,simclrv2_cache,wikitext103,wikitext103_cache"

cd "${REPO_ROOT}"
source env/bin/activate
mkdir -p "${BEFORE_DIR}" "${AFTER_DIR}"
exec > >(tee -a "${RUN_ROOT}/run.log") 2>&1

echo "[$(date -Is)] Archive current DpOptimizer plans"
IFS=',' read -r -a workload_array <<< "${WORKLOADS}"
for workload in "${workload_array[@]}"; do
  source_plan="${BASE_DIR}/${workload}/plans/dp_optimizer.yaml"
  [[ -f "${source_plan}" ]] || {
    echo "Missing current plan: ${source_plan}" >&2
    exit 1
  }
  cp -p "${source_plan}" "${BEFORE_DIR}/${workload}.yaml"
done

echo "[$(date -Is)] Incrementally profile backend compute"
INCREMENTAL_BACKEND_COMPUTE=1 \
CH6_PROFILE_RUN_ID="incremental_backend_${RUN_ID}" \
bash "${BASE_DIR}/run_formal_profiles.sh" --workloads "${WORKLOADS}"

echo "[$(date -Is)] Remove only stale DpOptimizer plan status files"
for workload in "${workload_array[@]}"; do
  rm -f \
    "${BASE_DIR}/${workload}/plans/dp_optimizer.yaml" \
    "${BASE_DIR}/${workload}/plans/dp_optimizer.unavailable.json"
done

echo "[$(date -Is)] Regenerate only DpOptimizer plans"
PLAN_ONLY=1 RESUME_EXISTING=1 OPTIMIZER_SET=dp_only \
CPU_BUDGET=64 LOCAL_WORKERS=8 REPEATS=1 \
bash "${BASE_DIR}/run_formal_plan_and_matrix.sh" \
  --workloads "${WORKLOADS}"

for workload in "${workload_array[@]}"; do
  regenerated="${BASE_DIR}/${workload}/plans/dp_optimizer.yaml"
  [[ -f "${regenerated}" ]] || {
    echo "Missing regenerated plan: ${regenerated}" >&2
    exit 1
  }
  cp -p "${regenerated}" "${AFTER_DIR}/${workload}.yaml"
done

python evaluation/compare_dp_plan_snapshots.py \
  --before "${BEFORE_DIR}" \
  --after "${AFTER_DIR}" \
  --output "${RUN_ROOT}/comparison.json"
echo "[$(date -Is)] Completed incremental profile and plan comparison: ${RUN_ROOT}"
