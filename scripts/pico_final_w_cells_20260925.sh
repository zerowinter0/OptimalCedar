#!/usr/bin/env bash
# C3: W scaling of the final optimizer.
#
# The fixed-structure arm (execute one materialised plan at several W) is
# blocked by the runner's fixed-plan path: after fixing the CLI flag, the YAML
# reading, the graph key type and the plan-cost reporting, a worker still fails
# with `KeyError: 0` and the driver then aborts in `_shutdown`.  This script runs
# the deployable arm instead: the final PICO is restricted to a single W
# candidate (`CEDAR_WORKER_SEARCH_SET`), so every point is the optimizer's own
# best plan at that W, measured end to end with the normal harness.
#
# Usage (inside the container):
#   bash scripts/pico_final_w_cells_20260925.sh [workload] [samples]
set -uo pipefail
cd /workspace/OptimalCedar
source env/bin/activate

WORKLOAD=${1:-simclrv2}
SAMPLES=${2:-9469}
OUT=${OUT:-outputs/pico_final_w_only_20260924}
PROFILE_DIR=${PROFILE_DIR:-outputs/affine_repr_profile_20260924}
LADDER=${LADDER:-1,4,16,64}
DATA=/workspace/OptimalCedar/evaluation/datasets/imagenette2/imagenette2/train
RAY_IP=172.23.166.105:6379

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CEDAR_RAY_PLACEMENT_RESOURCE=cedar_remote CEDAR_RAY_REQUIRE_REMOTE=1
export CEDAR_WORKER_READY_TIMEOUT_SEC=600
export CEDAR_MATCH_PROFILE_RESOURCES=1
export CEDAR_PROFILE_MATCH_CPU_BUDGET=64
export CEDAR_PROFILE_MATCH_RAY_CPU_BUDGET=64
export CEDAR_DP_WORKER_LADDER=0

work="$OUT/$WORKLOAD"
mkdir -p "$work/logs" "$work/results"
for w in ${LADDER//,/ }; do
  result="$work/results/w_cell_W${w}.json"
  if [ -s "$result" ]; then
    echo "SKIP $WORKLOAD W=$w"
    continue
  fi
  echo "W-CELL $WORKLOAD W=$w $(date -Is)"
  CEDAR_WORKER_SEARCH_SET="$w" python -u "$OUT/entry.py" \
    "$OUT/modules/evaluation/compare_optimizer_perf.py" \
    --dataset_file "$OUT/modules/evaluation/pipelines/simclrv2/cedar_dataset.py" \
    --dataset_kwargs "dataset_path=$DATA" \
    --batch_size 4 --num_epochs 1 --num_total_samples "$SAMPLES" \
    --use_ray --ray_ip "$RAY_IP" \
    --profiled_stats "$PROFILE_DIR/$WORKLOAD/shared.yaml" \
    --full_data_run --enable_local_parallelism --match_profile_resources \
    --cpu_budget 64 --ray_cpu_budget 64 \
    --optimizers pico_final \
    --optimizer_time_limit_sec 2400 --cedar_reorder_timeout_sec 2400 \
    --disable_cedar_runtime_timeout --num_repeats 1 \
    --skip_pico_plan_cost --disable_caching \
    --results_path "$result" > "$work/logs/w_cell_W${w}.log" 2>&1
  echo "W-CELL-DONE $WORKLOAD W=$w exit=$? $(date -Is)"
done
echo "W-CELLS-FINISHED $WORKLOAD $(date -Is)"
