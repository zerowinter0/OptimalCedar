#!/bin/bash
# Screen one workload with every planner on a small data subset.
#
#   SPEC=600 tmp_analysis/screen_workload.sh <workload> [samples]
#
# It (1) builds the subset from the workload's original dataset, (2) profiles
# the pipeline if the workload has no profile yet, (3) runs every planner on
# the subset with a drained full pass, and (4) prints the ranking table.
#
# The protocol mirrors the formal matrix: one shared profile, W chosen by each
# optimizer, 64 local + 64 remote Ray CPUs, `--match_profile_resources`.
# SPEC (seconds) caps each planner's own planning time so screening stays
# interactive; the final table should be re-run with the formal 1800/900.
set -u

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
DEV=optimalcedar-torch201-dev
RAY=172.23.166.105:6379
WORKLOAD=$1
SAMPLES=${2:-2000}
SPEC=${SPEC:-600}
OUT=${OUT:-/tmp/screen}
mkdir -p "$OUT"
# The harness writes results *inside* the container, so the shared repo path is
# the only location both sides can read.
RESULTS_DIR=outputs/screen
docker exec $DEV bash -lc "mkdir -p /workspace/OptimalCedar/$RESULTS_DIR"

# shellcheck disable=SC1091
source tmp_analysis/workload_env.sh "$WORKLOAD" || exit 1

if [ -n "${SOURCE:-}" ]; then
  docker exec $DEV bash -lc "[ -f $DATA ] || head -n $SAMPLES $SOURCE > $DATA; wc -l < $DATA"
fi

PROFILE_ABS="/workspace/OptimalCedar/$PROFILE"
if ! docker exec $DEV bash -lc "[ -f $PROFILE_ABS ]"; then
  echo "[screen] profiling $WORKLOAD -> $PROFILE"
  docker exec -e CEDAR_RAY_PLACEMENT_RESOURCE=cedar_remote \
    -e CEDAR_REUSE_BOUNDARY_MODEL=0 -e CEDAR_LAYERED_ADAPTIVE_PROFILE=1 \
    -e CEDAR_PROFILE_FILTER_SELECTIVITY=1 \
    -e CEDAR_PROFILE_SELECTIVITY_SEC=${SELECTIVITY_SEC:-90} \
    -e CEDAR_PROFILE_TIME_SEC=10 -e CEDAR_CM_SWEEP_TIME_SEC=10 \
    -e CEDAR_PROFILE_SCALING_WIDTHS=1,2,4,8 -e CEDAR_PROFILE_SCALING_TOP_K=5 \
    -e CEDAR_PROFILE_SCALING_MAX_SEC=10 \
    -e CEDAR_ADAPTIVE_PROFILE_MIN_SEC=3 -e CEDAR_ADAPTIVE_PROFILE_MAX_SEC=30 \
    -e CEDAR_PROFILE_MATCH_FIXED_LOCAL_WORKERS=8 \
    $DEV bash -lc "cd /workspace/OptimalCedar && source env/bin/activate && \
      python -u outputs/plumber_bench_20260912/entry.py evaluation/eval_cedar.py \
        --dataset_file $DATASET_FILE --dataset_func ${DATASET_FUNC:-get_dataset} \
        --dataset_kwargs '$DATASET_KWARGS' --batch_size 4 --num_total_samples 0 \
        --run_profiling --profiled_stats $PROFILE --use_ray --ray_ip $RAY \
        --disable_controller --disable_caching" \
    > "$OUT/$WORKLOAD.profile.log" 2>&1
  echo "[screen] profile exit=$? log=$OUT/$WORKLOAD.profile.log"
fi

PLANNERS=${OPTIMIZERS:-"dj_optimizer pecan_optimizer plumber_optimizer raydata_optimizer optimizer dp_optimizer simple_dp_optimizer"}
echo "[screen] planning+executing on $WORKLOAD ($SAMPLES): $PLANNERS"
docker exec -e CEDAR_RAY_PLACEMENT_RESOURCE=cedar_remote \
  -e CEDAR_DP_WORKER_SEARCH_TIME_LIMIT_SEC=${PICO_PLAN_BUDGET:-180} \
  $DEV bash -lc "cd /workspace/OptimalCedar && source env/bin/activate && \
  python -u outputs/plumber_bench_20260912/entry.py evaluation/compare_optimizer_perf.py \
    --dataset_file $DATASET_FILE --dataset_func ${DATASET_FUNC:-get_dataset} \
    --dataset_kwargs '$DATASET_KWARGS' --batch_size 4 --num_total_samples 0 \
    --profiled_stats $PROFILE --use_ray --ray_ip $RAY \
    --enable_local_parallelism --disable_caching --match_profile_resources \
    --cpu_budget 64 --ray_cpu_budget 64 \
    --optimizers $PLANNERS \
    --optimizer_time_limit_sec $SPEC --cedar_reorder_timeout_sec $SPEC \
    --disable_cedar_runtime_timeout --num_repeats 1 \
    --results_path /workspace/OptimalCedar/$RESULTS_DIR/$WORKLOAD.json" \
  > "$OUT/$WORKLOAD.run.log" 2>&1
echo "[screen] run exit=$?"
python3 tmp_analysis/screen_table.py "$RESULTS_DIR/$WORKLOAD.json"
