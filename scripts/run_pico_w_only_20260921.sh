#!/usr/bin/env bash
# PICO without the stage-width dimension, on the two workloads where the full
# joint W x width search cannot finish.
#
#   llava_pretrain / stackexchange: width ladder forced to {1} so the DP only
#   chooses the worker count, fusion, backend and cache placement.  Everything
#   else (profile, data volumes, harness flags) matches the campaign cells, and
#   the run uses the same fixed snapshot as the completed campaign.
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO" || exit 1
source "$REPO/env/bin/activate"

CAMPAIGN=outputs/ultimate_eight_optimizers_fix_20260921
RUN=outputs/pico_w_only_20260921
ENTRY=$CAMPAIGN/entry.py
MODULES=$CAMPAIGN/modules
RAY_IP=172.23.166.105:6379

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
       CEDAR_RAY_PLACEMENT_RESOURCE=cedar_remote CEDAR_RAY_REQUIRE_REMOTE=1 \
       CEDAR_WORKER_READY_TIMEOUT_SEC=600 \
       CEDAR_MATCH_PROFILE_RESOURCES=1 \
       CEDAR_PROFILE_MATCH_CPU_BUDGET=64 \
       CEDAR_PROFILE_MATCH_RAY_CPU_BUDGET=64 \
       CEDAR_DP_WIDTH_LADDER=1 \
       CEDAR_DP_WORKER_LADDER=0 \
       CEDAR_WORKER_SEARCH_SET="64,32,16" \
       CEDAR_DP_SEARCH_MODE=general

run_cell() {
  local workload=$1 dataset_file=$2 dataset_kwargs=$3
  local budget=${4:-64}
  local work=$RUN/$workload
  mkdir -p "$work"/{cache,logs,plans,results}
  echo "RUN $workload round1__pico_w_only (stage-CPU budget $budget, width ladder {1})"
  CEDAR_PROFILE_MATCH_CPU_BUDGET=$budget CEDAR_PROFILE_MATCH_RAY_CPU_BUDGET=$budget \
  python -u "$ENTRY" "$MODULES/evaluation/compare_optimizer_perf.py" \
    --dataset_file "$MODULES/$dataset_file" \
    --dataset_kwargs "$dataset_kwargs" \
    --batch_size 4 --num_epochs 1 --num_total_samples 0 \
    --use_ray --ray_ip "$RAY_IP" \
    --profiled_stats "$CAMPAIGN/$workload/profiles/shared.yaml" \
    --full_data_run --enable_local_parallelism --match_profile_resources \
    --cpu_budget 64 --ray_cpu_budget 64 \
    --optimizers simple_dp_workers_width_boundary \
    --optimizer_time_limit_sec 7200 --cedar_reorder_timeout_sec 7200 \
    --disable_cedar_runtime_timeout --num_repeats 1 --skip_pico_plan_cost \
    --disable_caching \
    --cache_root "$work/cache" \
    --results_path "$work/results/round1__pico_w_only.json" \
    > "$work/logs/round1__pico_w_only.log" 2>&1
  python - <<PY
import json, pathlib, yaml
work = pathlib.Path("$work")
res = work / "results/round1__pico_w_only.json"
if res.exists():
    payload = json.loads(res.read_text())
    for run in payload.get("runs", []):
        plans = run.get("physical_plans_by_feature", {})
        (work / "plans/pico_w_only.yaml").write_text(yaml.safe_dump(plans))
        print("   %s: %.1f rec/s (%s samples, %.1fs steady)"
              % ("$workload", run.get("throughput_samples_per_sec", 0),
                 run.get("num_samples"), run.get("perf_time_sec", 0)))
else:
    print("   $workload: no result file")
PY
}

run_cell llava_pretrain \
  evaluation/pipelines/llava_pretrain/cedar_dataset.py \
  "dataset_path=$REPO/$CAMPAIGN/inputs/llava_pretrain.jsonl,image_root=/workspace/OptimalCedar/evaluation/datasets/llava_pretrain" \
  8

run_cell stackexchange \
  evaluation/pipelines/stackexchange/cedar_dataset.py \
  "dataset_path=$REPO/$CAMPAIGN/inputs/stackexchange.jsonl" \
  64

echo DONE
