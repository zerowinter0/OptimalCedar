#!/usr/bin/env bash
# Complete the diagnosis plan set to three complete runs per plan.
# The four orders were captured once during the diagnosis; this adds rounds 2
# and 3 so every plan has the same repeat budget as the validation set.
#
# Usage (inside the container):
#   bash scripts/run_diagnosis_plan_rounds_20260924.sh [samples] [workers]
set -uo pipefail
cd /workspace/OptimalCedar
source env/bin/activate
SAMPLES=${1:-900}
WORKERS=${2:-4}
declare -A PLANS=(
  [pico]=outputs/unopt_order_transfer_repeats_traceall_20260921/plans/pico.yaml
  [cedar]=outputs/unopt_order_transfer_repeats_traceall_20260921/plans/cedar.yaml
  [old-dp]=outputs/unopt_order_transfer_repeats_traceall_20260921/plans/old-dp.yaml
)
for round in 2 3; do
  for name in "${!PLANS[@]}"; do
    out="tmp_analysis/capture_${name}_r${round}"
    if [ -d "$out/capture" ]; then
      echo "SKIP $name round $round"
      continue
    fi
    echo "CAPTURE $name round $round"
    CEDAR_CAPTURE_BATCH=4 CEDAR_OP_CAPTURE_SAMPLES=4 \
      python -u tmp_analysis/op_capture_run.py \
      "${PLANS[$name]}" "$out" "$SAMPLES" "$WORKERS" \
      > "$out.log" 2>&1 || echo "FAILED $name round $round"
  done
done
echo "diagnosis rounds done"
