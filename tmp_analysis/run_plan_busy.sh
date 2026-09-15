#!/bin/bash
# Measure the drained throughput of one materialized plan with per-operator
# call counters, on a small data subset.
#
#   tmp_analysis/run_plan_busy.sh <outdir> <plan.yaml> [placement] [samples]
#
# Every plan is measured the same way: the harness materializes the recorded
# plan, the counter hook is injected into the local workers and into the Ray
# actors (copied to both hosts), and the run is drained with
# --num_total_samples 0 so that the measured window contains every record.
#
# The workload is selected with environment variables, defaulting to the
# alpaca_cot subset used for the text-workload investigation:
#   DATASET_FILE, DATASET_KWARGS, PROFILE, DATA
#
# Results land in <outdir>/<plan name>.{log,json} together with the raw
# operator-call logs under <outdir>/<plan name>_opcalls/.
set -u

DEV=optimalcedar-torch201-dev
REMOTE_HOST=172.23.166.105
REMOTE_CT=optimalcedar-ray-remote
REPO=/workspace/OptimalCedar

OUT=$1
PLAN=$2
PLACEMENT=${3:-cedar_remote}
SAMPLES=${4:-2000}
NAME=$(basename "$PLAN" .yaml)
DATASET_FILE=${DATASET_FILE:-evaluation/pipelines/alpaca_cot/cedar_dataset.py}
DATASET_FUNC=${DATASET_FUNC:-get_dataset}
DATASET_KWARGS=${DATASET_KWARGS:-dataset_path=${DATA:-/tmp/small/alpaca_2k.jsonl}}
PROFILE=${PROFILE:-outputs/pico_drained_20260914/profiles/alpaca_cot_profile.yaml}

mkdir -p "$OUT"
OPCALLS="$OUT/${NAME}_opcalls"
rm -rf "$OPCALLS"
mkdir -p "$OPCALLS/local" "$OPCALLS/remote"

docker exec $DEV bash -lc "rm -f /tmp/opcalls_*.log"
ssh $REMOTE_HOST "docker exec $REMOTE_CT bash -lc 'rm -f /tmp/opcalls_*.log'" >/dev/null 2>&1

W=$(docker exec $DEV bash -lc "cd $REPO && source env/bin/activate >/dev/null && python3 -c \"import yaml;print(yaml.safe_load(open('$PLAN'))['physical_plan'].get('n_local_workers') or 1)\"")
echo "[run_plan_busy] plan=$NAME workers=$W placement=$PLACEMENT samples=$SAMPLES"

START=$(date +%s)
docker exec \
  -e CEDAR_MATCH_PROFILE_RESOURCES=1 \
  -e CEDAR_PROFILE_MATCH_CPU_BUDGET=64 \
  -e CEDAR_PROFILE_MATCH_RAY_CPU_BUDGET=64 \
  -e CEDAR_PROFILE_MATCH_FIXED_LOCAL_WORKERS="$W" \
  -e CEDAR_RAY_PLACEMENT_RESOURCE="$PLACEMENT" \
  -e PYTHONPATH=/tmp/patchsite \
  -e CEDAR_RAY_ACTOR_PYTHONPATH=/tmp/patchsite \
  $DEV bash -lc "cd $REPO && source env/bin/activate && \
    python -u outputs/plumber_bench_20260912/entry.py evaluation/eval_cedar.py \
      --dataset_file $DATASET_FILE \
      --dataset_func $DATASET_FUNC \
      --dataset_kwargs $DATASET_KWARGS \
      --batch_size 4 --num_total_samples 0 \
      --profiled_stats $PROFILE \
      --master_feature_config $PLAN \
      --use_ray --ray_ip $REMOTE_HOST:6379 \
      --disable_controller --disable_caching" \
  > "$OUT/$NAME.log" 2>&1
WALL=$(( $(date +%s) - START ))

docker exec $DEV bash -lc "cd /tmp && tar cf - \$(ls opcalls_*.log 2>/dev/null)" > "$OPCALLS/local.tar" 2>/dev/null
tar xf "$OPCALLS/local.tar" -C "$OPCALLS/local" 2>/dev/null
ssh $REMOTE_HOST "docker exec $REMOTE_CT bash -lc 'cd /tmp && tar cf - \$(ls opcalls_*.log 2>/dev/null)'" > "$OPCALLS/remote.tar" 2>/dev/null
tar xf "$OPCALLS/remote.tar" -C "$OPCALLS/remote" 2>/dev/null
rm -f "$OPCALLS/local.tar" "$OPCALLS/remote.tar"

echo "[run_plan_busy] plan=$NAME wall=${WALL}s local_logs=$(ls "$OPCALLS/local" | wc -l) remote_logs=$(ls "$OPCALLS/remote" | wc -l)"
python3 "$(dirname "$0")/plan_busy_analyze.py" \
  --label "$NAME" --json "$OUT/$NAME.json" \
  "$OPCALLS/local/*.log" "$OPCALLS/remote/*.log" | tee "$OUT/$NAME.stats"
