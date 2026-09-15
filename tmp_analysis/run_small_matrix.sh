#!/bin/bash
# Small-data plan-and-measure matrix: PICO's DP plan against the recorded
# baseline plans, workload by workload, on the drained busy-window protocol.
#
#   WORKLOADS="blip clip" OUT=/tmp/small_matrix tmp_analysis/run_small_matrix.sh
#
# Per workload it (1) re-plans with dp_optimizer on a small sample count,
# (2) dumps that plan, (3) measures it and every recorded baseline plan with
# tmp_analysis/run_plan_busy.sh, and (4) appends a summary row to
# $OUT/SUMMARY.tsv.
set -u

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
DEV=optimalcedar-torch201-dev
RAY=172.23.166.105:6379
WORKLOADS=${WORKLOADS:-"blip clip dino pile_hackernews pile_pubmed_abstracts pile_uspto_backgrounds bloom_oscar"}
OUT=${OUT:-/tmp/small_matrix}
mkdir -p "$OUT"
SUMMARY="$OUT/SUMMARY.tsv"
[ -f "$SUMMARY" ] || printf 'workload\tplanner\trecords\tbusy_sec\trate\n' > "$SUMMARY"

for workload in $WORKLOADS; do
  # shellcheck disable=SC1091
  source tmp_analysis/workload_env.sh "$workload" || continue
  # The data path may be a host path or a container path; count it inside the
  # container, which is where the measurement will read it.
  SAMPLES=$(docker exec $DEV bash -lc "cd /workspace/OptimalCedar && wc -l < ${DATA:-/dev/null} 2>/dev/null || echo 2000")
  echo "=== $workload samples=$SAMPLES data=$DATA"
  plan_json=/tmp/small/plan_${workload}.json
  plan_dir=/tmp/small/plans_${workload}
  docker exec -e CEDAR_DP_WORKER_SEARCH_TIME_LIMIT_SEC=${WORKER_SEARCH_BUDGET:-420} $DEV bash -lc "cd /workspace/OptimalCedar && source env/bin/activate && \
    export CEDAR_RAY_PLACEMENT_RESOURCE=cedar_remote && \
    python -u outputs/plumber_bench_20260912/entry.py evaluation/compare_optimizer_perf.py \
      --dataset_file $DATASET_FILE --dataset_func ${DATASET_FUNC:-get_dataset} \
      --dataset_kwargs '$DATASET_KWARGS' \
      --batch_size 4 --num_total_samples $SAMPLES --full_data_run \
      --profiled_stats $PROFILE --use_ray --ray_ip $RAY \
      --enable_local_parallelism --disable_caching --match_profile_resources \
      --cpu_budget 64 --ray_cpu_budget 64 --optimizers dp_optimizer \
      --optimizer_time_limit_sec 900 --cedar_reorder_timeout_sec 900 \
      --disable_cedar_runtime_timeout --num_repeats 1 \
      --results_path $plan_json" >> "$OUT/$workload.plan.log" 2>&1
  docker exec $DEV bash -lc "cd /workspace/OptimalCedar && python3 tmp_analysis/dump_run_plans.py $plan_json $plan_dir" >> "$OUT/$workload.plan.log" 2>&1

  specs="pico:$plan_dir/dp_optimizer.yaml"
  for alias in cedar dj pecan plumber raydata simple_dp; do
    if docker exec $DEV bash -lc "cd /workspace/OptimalCedar && python3 tmp_analysis/dump_recorded_plan.py $workload $alias /tmp/small/recorded/${workload}__${alias}.yaml" >/dev/null 2>&1; then
      specs="$specs $alias:/tmp/small/recorded/${workload}__${alias}.yaml"
    fi
  done
  echo "  plans:$specs"
  WORK_OUT="$OUT/$workload"
  for spec in $specs; do
    name=${spec%%:*}
    plan=${spec#*:}
    stem=$(basename "$plan" .yaml)
    DATASET_FILE="$DATASET_FILE" DATASET_KWARGS="$DATASET_KWARGS" \
      DATASET_FUNC="${DATASET_FUNC:-get_dataset}" \
      PROFILE="$PROFILE" DATA="$DATA" \
      ./tmp_analysis/run_plan_busy.sh "$WORK_OUT" "$plan" cedar_remote "$SAMPLES" \
      >> "$OUT/$workload.measure.log" 2>&1
    python3 tmp_analysis/append_summary.py \
      "$SUMMARY" "$workload" "$name" "$WORK_OUT/$stem.json" 2>&1
    grep -h "^$stem " "$WORK_OUT/$stem.stats" 2>/dev/null
  done
done
echo "matrix complete"
