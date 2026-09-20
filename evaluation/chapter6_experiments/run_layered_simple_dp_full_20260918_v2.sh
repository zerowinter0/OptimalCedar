#!/usr/bin/env bash
# Fresh one-round six-workload matrix after the mutable-input affine-profile
# fix.  The run creates every profile from scratch and snapshots the current
# Cedar/evaluation implementation into the output directory.
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/OptimalCedar}"
RUN_ROOT="${RUN_ROOT:-outputs/simple_dp_layered_fresh_all_20260918_v7}"
cd "${REPO_ROOT}"
source env/bin/activate
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1

echo "[$(date -Is)] fresh layered matrix start pid=$$ root=${RUN_ROOT}"
python -u evaluation/chapter6_experiments/run_simple_dp_ablation_matrix.py \
  --output "${RUN_ROOT}" --prepared \
  --workloads simclrv2 simclrv2_cache commonvoice coco llava_pretrain stackexchange \
  --commonvoice-max-samples 15000 --repeats 1 \
  --layered-profile \
  --skip-cedar-workloads llava_pretrain stackexchange
rc=$?
echo "[$(date -Is)] fresh layered matrix exit=${rc}"
exit "${rc}"
