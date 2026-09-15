#!/bin/bash
# Re-profile text recipes with the selectivity pass enabled, then screen them.
#
# The old profiles recorded selectivity 1.0 for every filter because the
# timing-bounded pass saw a few dozen records; the selectivity pass measures
# the surviving fraction on real records, which is what lets the joint DP
# order cheap/selective filters before expensive ones.
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
DEV=optimalcedar-torch201-dev
for workload in "$@"; do
  echo "########## $workload $(date +%H:%M:%S)"
  docker exec $DEV bash -lc "rm -f /workspace/OptimalCedar/outputs/pico_drained_20260914/profiles/${workload}_profile.yaml"
  SPEC=300 PICO_PLAN_BUDGET=300 OUT=/tmp/screen \
    OPTIMIZERS="plumber_optimizer optimizer dp_optimizer simple_dp_optimizer" \
    timeout 4500 bash tmp_analysis/screen_workload.sh "$workload" 2000
done
echo TEXT_BATCH_DONE
