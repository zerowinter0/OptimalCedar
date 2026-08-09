#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${REPO_ROOT}/evaluation/chapter6_experiments/formal_results/paper_dp_algorithm_validation"
cd "${REPO_ROOT}"
source env/bin/activate
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
mkdir -p "${OUT}/logs" "${OUT}/data" "${OUT}/figures"

python evaluation/verify_dp_optimizer_optimality.py \
  --num-cases 100 --seed 20260715 --max-dependencies 4 \
  --failed-case "${OUT}/data/failed_case.json" \
  --summary-json "${OUT}/data/optimality_summary.json" \
  > "${OUT}/logs/optimality.log" 2>&1

python evaluation/benchmark_cedar_reorder_time.py \
  --optimizer dp_optimizer --min-ops 2 --max-ops 19 --repeats 3 \
  --timeout-sec 3600 \
  --csv "${OUT}/data/dp_scalability.csv" \
  --figure "${OUT}/figures/dp_scalability_only.pdf" \
  > "${OUT}/logs/dp_scalability.log" 2>&1

python evaluation/benchmark_cedar_reorder_time.py \
  --optimizer cedar --min-ops 2 --max-ops 11 --repeats 3 \
  --timeout-sec 3600 \
  --csv "${OUT}/data/cedar_scalability.csv" \
  --figure "${OUT}/figures/cedar_scalability_only.pdf" \
  > "${OUT}/logs/cedar_scalability.log" 2>&1

python evaluation/chapter6_experiments/plot_dp_scalability.py \
  --dp "${OUT}/data/dp_scalability.csv" \
  --cedar "${OUT}/data/cedar_scalability.csv" \
  --output "${OUT}/figures/dp_vs_cedar_scalability.png"
