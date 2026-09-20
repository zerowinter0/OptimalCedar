#!/usr/bin/env bash
# Re-sample the six simple-DP ablation workloads with the adaptive layered
# (per-operator x per-backend) profile protocol, then re-run the unchanged
# single-round ablation matrix and the two W-conditioned boundary variants.
#
# Unchanged from the 20260917 campaign: workloads, data volumes (simclrv2 9469,
# simclrv2_cache 9469, commonvoice 15000, coco 5000, llava_pretrain 1000,
# stackexchange 2000), optimizers, 64+64 CPU budgets, free W,
# --match_profile_resources, cache policy, one round.  Cells that timed out in
# that campaign are skipped.
#
# Profiles: CEDAR_LAYERED_ADAPTIVE_PROFILE=1 with a 10 s in-process pilot, a
# per-operator x per-backend fixed-legal-input replay (>=3 s / >=30 obs,
# RSE <= 10 %, <=30 s), measured object boundaries, and width curves
# {1,2,4,8} for the five most expensive operators per backend.  Stage profile
# width stays 1 actor/process so the width-one resource signature is preserved.
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/OptimalCedar}"
cd "${REPO_ROOT}"
source env/bin/activate
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1

MATRIX_ROOT="outputs/simple_dp_layered_ablation_20260918"
WB_ROOT="outputs/simple_dp_layered_workers_boundary_20260918"
MAXWB_ROOT="outputs/simple_dp_layered_max_workers_boundary_20260918"
QUEUE_LOG="${MATRIX_ROOT}.queue.log"

echo "[$(date -Is)] QUEUE start pid=$$"
echo "[$(date -Is)] PHASE 1/3 matrix+profiles -> ${MATRIX_ROOT}"
python -u evaluation/chapter6_experiments/run_simple_dp_ablation_matrix.py \
  --output "${MATRIX_ROOT}" \
  --commonvoice-max-samples 15000 --repeats 1 \
  --layered-profile \
  --skip-cedar-workloads llava_pretrain stackexchange \
  --skip-cell simple-dp+width@llava_pretrain \
  --skip-cell simple-dp+width@stackexchange
rc=$?
echo "[$(date -Is)] PHASE 1 exit=${rc}"
if [ "${rc}" -ne 0 ]; then
  echo "[$(date -Is)] QUEUE aborted after phase 1"
  exit "${rc}"
fi

echo "[$(date -Is)] PHASE 2/3 simple-dp+W+boundary -> ${WB_ROOT}"
python -u evaluation/chapter6_experiments/run_simple_dp_workers_boundary_reuse.py \
  --output "${WB_ROOT}" --profile-source-root "${MATRIX_ROOT}"
rc=$?
echo "[$(date -Is)] PHASE 2 exit=${rc}"
if [ "${rc}" -ne 0 ]; then
  echo "[$(date -Is)] QUEUE aborted after phase 2"
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
  echo "[$(date -Is)] QUEUE aborted after phase 3"
  exit "${rc}"
fi

echo "[$(date -Is)] QUEUE COMPLETE"
echo "logs: ${QUEUE_LOG}"
