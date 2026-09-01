#!/usr/bin/env bash
set -euo pipefail

cd /workspace/OptimalCedar
source env/bin/activate

RUN_ID="${RUN_ID:-structural_frontier_v1}"
OUT_ROOT="${OUT_ROOT:-/workspace/OptimalCedar/outputs/chapter6_experiments/dp_planning_pair_${RUN_ID}}"
RAY_IP="${RAY_IP:-172.23.166.105:6379}"
STACK_TIMEOUT_SEC="${STACK_TIMEOUT_SEC:-300}"
SMALL_TIMEOUT_SEC="${SMALL_TIMEOUT_SEC:-120}"
mkdir -p "${OUT_ROOT}"

export CEDAR_DP_MASK_LAYER_WORKERS="${CEDAR_DP_MASK_LAYER_WORKERS:-1}"
export CEDAR_DP_PARETO_EPSILON=0
export CEDAR_DP_FRONTIER_CAP=0
export CEDAR_DP_CELL_FRONTIER=0

run_gate() {
  local workload="$1"
  local dataset_file="$2"
  local profile="$3"
  local samples="$4"
  local timeout_sec="$5"
  local dataset_kwargs="$6"
  local workload_root="${OUT_ROOT}/${workload}"
  mkdir -p "${workload_root}/logs" "${workload_root}/results"

  python evaluation/compare_optimizer_perf.py \
    --dataset_file "${dataset_file}" \
    --profiled_stats "${profile}" \
    --num_total_samples "${samples}" \
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
    --optimizer_time_limit_sec "${timeout_sec}" \
    --disable_cedar_runtime_timeout \
    --plan_only \
    --optimizers dp_optimizer \
    --results_path "${workload_root}/results/plan.json" \
    --disable_caching \
    --dataset_kwargs "${dataset_kwargs}" \
    > "${workload_root}/logs/plan.log" 2>&1
}

# Alpaca-CoT is the fixed small control: eight operators, a complete current
# remote-resource profile, and a historical DP planning time around 5 s.
run_gate \
  alpaca_cot \
  evaluation/pipelines/alpaca_cot/cedar_dataset.py \
  /workspace/OptimalCedar/outputs/chapter6_experiments/remote_separate_resource_pools_formal_v1/profiles/alpaca_cot.yaml \
  2000 \
  "${SMALL_TIMEOUT_SEC}" \
  dataset_path=datasets/alpaca_cot/alpaca-cot-en-cot-data.jsonl

run_gate \
  stackexchange \
  evaluation/pipelines/stackexchange/cedar_dataset.py \
  /workspace/OptimalCedar/outputs/chapter6_experiments/remote_single_budget_exact_quick_gate_v1/profiles/stackexchange.yaml \
  5000 \
  "${STACK_TIMEOUT_SEC}" \
  dataset_path=datasets/stackexchange/redpajama-stackexchange-35000.jsonl

touch "${OUT_ROOT}/COMPLETE"
