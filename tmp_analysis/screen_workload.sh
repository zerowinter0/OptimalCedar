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
export SUBSET=${SUBSET:-${SAMPLES}}
# The harness writes results *inside* the container, so the shared repo path is
# the only location both sides can read.  ``RESULTS_DIR`` can be overridden so
# diagnostic probes do not overwrite the recorded screening cells.
RESULTS_DIR=${RESULTS_DIR:-outputs/screen}
docker exec $DEV bash -lc "mkdir -p /workspace/OptimalCedar/$RESULTS_DIR"
# The harness ships ``outputs/plumber_bench_20260912/modules`` to the Ray
# actors as ``py_modules``; that snapshot is what the actors execute, so it
# must be refreshed from the live tree, otherwise actor-side code changes
# silently do not take effect.  Copy only code: the evaluation tree also
# carries the experiment outputs (hundreds of GB) and copying those hangs.
docker exec $DEV bash -lc "cd /workspace/OptimalCedar && \
  mkdir -p outputs/plumber_bench_20260912/modules/cedar outputs/plumber_bench_20260912/modules/evaluation && \
  cp -a cedar/. outputs/plumber_bench_20260912/modules/cedar/ && \
  cp -a evaluation/pipelines/. outputs/plumber_bench_20260912/modules/evaluation/pipelines/ && \
  cp -a evaluation/*.py outputs/plumber_bench_20260912/modules/evaluation/ 2>/dev/null; true"

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
    -e CEDAR_PROFILE_FILTER_SELECTIVITY=${SELECTIVITY_PASS:-1} \
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
docker exec -e CEDAR_RAY_PLACEMENT_RESOURCE=${PLACEMENT:-cedar_remote} \
  -e CEDAR_DP_WORKER_SEARCH_TIME_LIMIT_SEC=${PICO_PLAN_BUDGET:-180} \
  -e CEDAR_DP_SMP_MODE=${SMP_MODE:-lane} \
  ${CEDAR_WORKER_SEARCH_SET:+-e CEDAR_WORKER_SEARCH_SET=$CEDAR_WORKER_SEARCH_SET} \
  ${CEDAR_DP_RUNTIME_CPU_RESERVE_PER_WORKER:+-e CEDAR_DP_RUNTIME_CPU_RESERVE_PER_WORKER=$CEDAR_DP_RUNTIME_CPU_RESERVE_PER_WORKER} \
  ${CEDAR_DP_CHAIN_BEAM:+-e CEDAR_DP_CHAIN_BEAM=$CEDAR_DP_CHAIN_BEAM} \
  ${CEDAR_DP_RAY_MODE:+-e CEDAR_DP_RAY_MODE=$CEDAR_DP_RAY_MODE} \
  ${CEDAR_DP_SEARCH_MODE:+-e CEDAR_DP_SEARCH_MODE=$CEDAR_DP_SEARCH_MODE} \
  -e CEDAR_DATA_JUICER_ROOT=${CEDAR_DATA_JUICER_ROOT:-/workspace/OptimalCedar/data-juicer} \
  -e HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} \
  -e TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1} \
  -e CEDAR_RAY_ACTOR_READY_TIMEOUT_SEC=${ACTOR_READY_TIMEOUT:-240} \
  -e CEDAR_WORKER_READY_TIMEOUT_SEC=${WORKER_READY_TIMEOUT:-600} \
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
