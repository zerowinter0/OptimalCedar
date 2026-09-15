#!/bin/bash
# Lean screening loop: four planners per workload, short planning budgets.
#
#   tmp_analysis/explore_workloads.sh <workload> [<workload> ...]
#
# The four planners are the ones that decide the screening question — Plumber
# (width heuristic), Cedar (staged reorder + offload), PICO, and PICO's own
# ablation simple-DP.  The full seven-planner comparison is only worth running
# for workloads that look competitive here.
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
OUT=${OUT:-/tmp/screen}
mkdir -p "$OUT"
for workload in "$@"; do
  echo "########## $workload $(date +%H:%M:%S)"
  SPEC=${SPEC:-150} PICO_PLAN_BUDGET=${PICO_PLAN_BUDGET:-120} OUT="$OUT" \
    OPTIMIZERS="plumber_optimizer optimizer dp_optimizer simple_dp_optimizer" \
    timeout 2700 bash tmp_analysis/screen_workload.sh "$workload" \
    "${SAMPLES:-2000}"
done
echo EXPLORE_DONE
