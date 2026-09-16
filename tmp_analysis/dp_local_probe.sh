#!/bin/bash
# Measure the plan PICO's model produces when offload is excluded (Ray), to
# separate "model prefers the wrong backend" from "model builds a bad local plan".
set -u
REPO=/home/xieruiyang/OptimalCedar
cd "$REPO"
DEV=optimalcedar-torch201-dev
RAY=172.23.166.105:6379
W=$1; N=$2; VARIANTS=$3; OUT=${4:-/tmp/screen_local}
mkdir -p "$OUT"
source tmp_analysis/workload_env.sh "$W" || exit 1
docker exec $DEV bash -lc "cd /workspace/OptimalCedar && \
  mkdir -p outputs/plumber_bench_20260912/modules/cedar outputs/plumber_bench_20260912/modules/evaluation && \
  cp -a cedar/. outputs/plumber_bench_20260912/modules/cedar/ && \
  cp -a evaluation/pipelines/. outputs/plumber_bench_20260912/modules/evaluation/pipelines/ && \
  cp -a evaluation/*.py outputs/plumber_bench_20260912/modules/evaluation/ 2>/dev/null; true"
docker exec -e CEDAR_RAY_PLACEMENT_RESOURCE=cedar_remote \
  -e CEDAR_DP_WORKER_SEARCH_TIME_LIMIT_SEC=600 \
  -e CEDAR_DP_SMP_MODE=additive \
  -e CEDAR_DP_FORCE_VARIANTS="$VARIANTS" \
  -e CEDAR_DATA_JUICER_ROOT=/workspace/OptimalCedar/data-juicer \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e CEDAR_RAY_ACTOR_READY_TIMEOUT_SEC=900 \
  -e CEDAR_WORKER_READY_TIMEOUT_SEC=1800 \
  $DEV bash -lc "cd /workspace/OptimalCedar && source env/bin/activate && \
  python -u outputs/plumber_bench_20260912/entry.py evaluation/compare_optimizer_perf.py \
    --dataset_file $DATASET_FILE --dataset_func ${DATASET_FUNC:-get_dataset} \
    --dataset_kwargs '$DATASET_KWARGS' --batch_size 4 --num_total_samples 0 \
    --profiled_stats $PROFILE --use_ray --ray_ip $RAY \
    --enable_local_parallelism --disable_caching --match_profile_resources \
    --cpu_budget 64 --ray_cpu_budget 64 \
    --optimizers dp_optimizer \
    --optimizer_time_limit_sec 600 --cedar_reorder_timeout_sec 600 \
    --disable_cedar_runtime_timeout --num_repeats 1 \
    --results_path /workspace/OptimalCedar/outputs/local_probe/${W}_${VARIANTS//,/_}.json" \
  > "$OUT/${W}_${VARIANTS//,/_}.log" 2>&1
echo "[local_probe] $W variants=$VARIANTS exit=$?"
