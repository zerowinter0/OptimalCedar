#!/usr/bin/env bash
set -euo pipefail

cd /workspace/OptimalCedar
source env/bin/activate

OUT_ROOT="${OUT_ROOT:-/workspace/OptimalCedar/outputs/chapter6_experiments/stackexchange_exact_structural_pruning_gate_v1}"
PROFILE="${PROFILE:-/workspace/OptimalCedar/outputs/chapter6_experiments/remote_single_budget_exact_quick_gate_v1/profiles/stackexchange.yaml}"
RAY_IP="${RAY_IP:-172.23.166.105:6379}"
mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/results"

{
  echo "workload=stackexchange"
  echo "purpose=exact_structural_pruning_runtime_gate"
  echo "profile=${PROFILE}"
  echo "optimizer=dp_optimizer"
  echo "optimizer_timeout_sec=300"
  echo "mask_layer_workers=${CEDAR_DP_MASK_LAYER_WORKERS:-1}"
  echo "pareto_epsilon=0"
  echo "cache=off"
  echo "cpu_budget=64"
  echo "ray_cpu_budget=64"
  echo "local_workers=8"
  echo "ray_ip=${RAY_IP}"
} > "${OUT_ROOT}/PROTOCOL.txt"

export CEDAR_DP_MASK_LAYER_WORKERS="${CEDAR_DP_MASK_LAYER_WORKERS:-1}"
export CEDAR_DP_PARETO_EPSILON=0
export CEDAR_DP_FRONTIER_CAP=0
export CEDAR_DP_CELL_FRONTIER=0

python evaluation/compare_optimizer_perf.py \
  --dataset_file evaluation/pipelines/stackexchange/cedar_dataset.py \
  --profiled_stats "${PROFILE}" \
  --num_total_samples 5000 \
  --num_epochs 1 \
  --num_repeats 1 \
  --warmup_runs 0 \
  --full_data_run \
  --use_ray \
  --ray_ip "${RAY_IP}" \
  --enable_local_parallelism \
  --match_profile_resources \
  --cpu_budget 64 \
  --ray_cpu_budget 64 \
  --fixed_local_workers_ablation 8 \
  --optimizer_time_limit_sec 300 \
  --disable_cedar_runtime_timeout \
  --plan_only \
  --optimizers dp_optimizer \
  --results_path "${OUT_ROOT}/results/plan.json" \
  --disable_caching \
  --dataset_kwargs \
    dataset_path=datasets/stackexchange/redpajama-stackexchange-35000.jsonl \
  > "${OUT_ROOT}/logs/plan.log" 2>&1

touch "${OUT_ROOT}/COMPLETE"
