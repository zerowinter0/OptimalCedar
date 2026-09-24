#!/usr/bin/env bash
# Measure the semantics-preserving plan set (only ``to_float`` moves) with the
# operator capture, three complete runs per plan, interleaved order.
#
# Usage (inside the container):
#   bash scripts/run_semantic_plan_set_20260924.sh [samples] [workers]
set -uo pipefail
cd /workspace/OptimalCedar
source env/bin/activate
SAMPLES=${1:-900}
WORKERS=${2:-4}
PLANS="a_f0 a_f1 a_f2 a_f3 a_f4 a_f5"
for round in 1 2 3; do
  for plan in $PLANS; do
    out="tmp_analysis/capture_${plan}_r${round}"
    if [ -d "$out/capture" ]; then
      echo "SKIP $plan round $round (already captured)"
      continue
    fi
    echo "CAPTURE $plan round $round"
    CEDAR_CAPTURE_BATCH=4 CEDAR_OP_CAPTURE_SAMPLES=4 \
      python -u tmp_analysis/op_capture_run.py \
      "$( [ -f tmp_analysis/semantic_orders/${plan}.yaml ] && echo tmp_analysis/semantic_orders/${plan}.yaml || echo outputs/unopt_order_transfer_repeats_traceall_20260921/plans/${plan}.yaml )" \
      "$out" "$SAMPLES" "$WORKERS" \
      > "$out.log" 2>&1 || echo "FAILED $plan round $round"
  done
done
echo "semantic plan set done"
