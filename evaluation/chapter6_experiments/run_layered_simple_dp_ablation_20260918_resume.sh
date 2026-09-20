#!/usr/bin/env bash
# Resume the layered re-sampling campaign after the 2026-09-17 local-Ray
# OOM/raylet death (node 172.23.166.103):
#   1. resume the matrix: reuse simclrv2 / simclrv2_cache / commonvoice cells,
#      re-profile coco / llava_pretrain / stackexchange, then run their cells;
#   2. verify all six layered profiles exist;
#   3. run the two W-conditioned boundary variants on the new profiles.
#
# Everything else is identical to run_layered_simple_dp_ablation_20260918.sh.
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/OptimalCedar}"
cd "${REPO_ROOT}"
source env/bin/activate
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export RAY_memory_usage_threshold=0.98

MATRIX_ROOT="outputs/simple_dp_layered_ablation_20260918"
WB_ROOT="outputs/simple_dp_layered_workers_boundary_20260918"
MAXWB_ROOT="outputs/simple_dp_layered_max_workers_boundary_20260918"

echo "[$(date -Is)] RESUME start pid=$$"
python -u evaluation/chapter6_experiments/run_simple_dp_ablation_matrix.py \
  --output "${MATRIX_ROOT}" --resume \
  --layered-profile \
  --skip-cedar-workloads llava_pretrain stackexchange \
  --skip-cell simple-dp+width@llava_pretrain \
  --skip-cell simple-dp+width@stackexchange
rc=$?
echo "[$(date -Is)] matrix resume exit=${rc}"
if [ "${rc}" -ne 0 ]; then
  echo "[$(date -Is)] RESUME aborted after matrix"
  exit "${rc}"
fi

missing=0
for workload in simclrv2 simclrv2_cache commonvoice coco llava_pretrain stackexchange; do
  if [ ! -s "${MATRIX_ROOT}/${workload}/profiles/shared.yaml" ]; then
    echo "[$(date -Is)] MISSING PROFILE ${workload}"
    missing=1
  fi
done
if [ "${missing}" -ne 0 ]; then
  echo "[$(date -Is)] RESUME aborted: incomplete profiles"
  exit 3
fi

for root in "${WB_ROOT}" "${MAXWB_ROOT}"; do
  if [ -d "${root}" ] && [ ! -f "${root}/COMPLETE" ]; then
    echo "[$(date -Is)] Removing incomplete run dir ${root}"
    rm -rf "${root}"
  fi
done

echo "[$(date -Is)] PHASE 2/3 simple-dp+W+boundary -> ${WB_ROOT}"
python -u evaluation/chapter6_experiments/run_simple_dp_workers_boundary_reuse.py \
  --output "${WB_ROOT}" --profile-source-root "${MATRIX_ROOT}"
rc=$?
echo "[$(date -Is)] PHASE 2 exit=${rc}"
if [ "${rc}" -ne 0 ]; then
  echo "[$(date -Is)] RESUME aborted after phase 2"
  exit "${rc}"
fi

echo "[$(date -Is)] PHASE 3/3 simple-dp+max-W+boundary -> ${MAXWB_ROOT}"
python -u evaluation/chapter6_experiments/run_simple_dp_workers_boundary_reuse.py \
  --output "${MAXWB_ROOT}" --profile-source-root "${MATRIX_ROOT}" \
  --label "simple-dp+max-W+boundary" \
  --optimizer-internal simple_dp_max_workers_boundary
rc=$?
echo "[$(date -Is)] PHASE 3 exit=${rc}"
if [ "${rc}" -ne 0 ]; then
  echo "[$(date -Is)] RESUME aborted after phase 3"
  exit "${rc}"
fi

echo "[$(date -Is)] RESUME COMPLETE"
