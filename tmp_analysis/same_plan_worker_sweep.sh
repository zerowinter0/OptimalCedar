#!/bin/bash
# Same-plan worker sweep: take one recorded plan, freeze it, and run it at
# several local-worker counts.  This is the calibration the DP's
# worker_contention block expects (plan shape fixed, only W varies).
#   same_plan_worker_sweep.sh <workload> <samples> <plan.json> <planner> <W...>
set -u
REPO=/home/xieruiyang/OptimalCedar
cd "$REPO"
DEV=optimalcedar-torch201-dev
RAY=172.23.166.105:6379
WORKLOAD=$1; N=$2; RESULT=$3; PLANNER=$4; shift 4
OUT=${OUT:-/tmp/screen_spws}
mkdir -p "$OUT"
source tmp_analysis/workload_env.sh "$WORKLOAD" || exit 1
# Materialize the recorded plan as the config the dataset loader expects.
PLAN_REL=outputs/sp_probe/${WORKLOAD}_${PLANNER}_plan.yaml
PLAN_YAML=/tmp/${WORKLOAD}_${PLANNER}_plan.yaml
python3 - "$RESULT" "$PLANNER" "$PLAN_YAML" <<'PY'
import json, sys, yaml
payload = json.loads(open(sys.argv[1]).read())
planner, out = sys.argv[2], sys.argv[3]
for run in payload.get("runs", []):
    if run.get("optimizer") != planner:
        continue
    plans = run.get("physical_plans_by_feature") or {}
    if not plans:
        raise SystemExit(f"no plan recorded for {planner}")
    plan = next(iter(plans.values()))
    yaml.safe_dump({"physical_plan": plan}, open(out, "w"), sort_keys=False)
    print(f"wrote {out} from {planner} (nw={plan.get('n_local_workers')})")
    break
else:
    raise SystemExit(f"planner {planner} not found")
PY
docker cp "$PLAN_YAML" $DEV:/workspace/OptimalCedar/$PLAN_REL >/dev/null
docker exec $DEV bash -lc "cd /workspace/OptimalCedar && \
  mkdir -p outputs/plumber_bench_20260912/modules/cedar outputs/plumber_bench_20260912/modules/evaluation && \
  cp -a cedar/. outputs/plumber_bench_20260912/modules/cedar/ && \
  cp -a evaluation/pipelines/. outputs/plumber_bench_20260912/modules/evaluation/pipelines/ && \
  cp -a evaluation/*.py outputs/plumber_bench_20260912/modules/evaluation/ 2>/dev/null; true"
for W in "$@"; do
  echo "########## $WORKLOAD fixed plan, W=$W $(date +%H:%M:%S)"
  docker exec -e CEDAR_RAY_PLACEMENT_RESOURCE=cedar_remote \
    -e CEDAR_DP_SMP_MODE=lane \
    -e CEDAR_PROFILE_MATCH_FIXED_LOCAL_WORKERS=$W \
    -e CEDAR_MATCH_PROFILE_RESOURCES=1 \
    -e CEDAR_PROFILE_MATCH_CPU_BUDGET=64 \
    -e CEDAR_PROFILE_MATCH_RAY_CPU_BUDGET=64 \
    -e CEDAR_DATA_JUICER_ROOT=/workspace/OptimalCedar/data-juicer \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e CEDAR_RAY_ACTOR_READY_TIMEOUT_SEC=600 \
    -e CEDAR_WORKER_READY_TIMEOUT_SEC=1200 \
    $DEV bash -lc "cd /workspace/OptimalCedar && source env/bin/activate && \
    python -u outputs/plumber_bench_20260912/entry.py evaluation/eval_cedar.py \
      --dataset_file $DATASET_FILE --dataset_func ${DATASET_FUNC:-get_dataset} \
      --dataset_kwargs '$DATASET_KWARGS' --batch_size 4 --num_total_samples $N \
      --master_feature_config /workspace/OptimalCedar/$PLAN_REL \
      --profiled_stats $PROFILE --use_ray --ray_ip $RAY \
      --disable_controller --disable_caching \
      --results_path /workspace/OptimalCedar/outputs/sp_probe/${WORKLOAD}_W${W}.json" \
    > "$OUT/${WORKLOAD}_W${W}.log" 2>&1
  echo "[same-plan] W=$W exit=$?"
done
