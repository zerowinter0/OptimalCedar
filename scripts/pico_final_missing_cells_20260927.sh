#!/usr/bin/env bash
# Finish the cells that the 7-hour fast campaign left out.
#
# The earlier fast run produced no usable CommonVoice / COCO numbers because
# their profiles were broken (numpy payloads had no representation class; COCO
# counterfactual upscaling blew up on full-resolution images).  Both root
# causes are fixed; this script re-runs exactly the missing cells with the
# already-relaxed fast protocol (1 round, reduced data volume, no unopti, no
# Ray Data on CommonVoice, no Cedar on LLaVA).
#
# Resume-safe: a cell whose result JSON exists and is non-empty is skipped.
#
# Usage (inside the container, under tmux):
#   bash scripts/pico_final_missing_cells_20260927.sh [workloads...]
set -uo pipefail
cd /workspace/OptimalCedar
source env/bin/activate

PROFILE_DIR=${PROFILE_DIR:-outputs/affine_repr_profile_20260924}
OUT=${OUT:-outputs/pico_final_w_only_20260924}
RAY_IP=172.23.166.105:6379
PLAN_TIMEOUT=${PLAN_TIMEOUT:-2400}

ENTRY="$OUT/entry.py"
MODULES="$OUT/modules"
mkdir -p "$OUT/logs" "$OUT/results"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CEDAR_RAY_PLACEMENT_RESOURCE=cedar_remote CEDAR_RAY_REQUIRE_REMOTE=1
export CEDAR_WORKER_READY_TIMEOUT_SEC=600
export CEDAR_MATCH_PROFILE_RESOURCES=1
export CEDAR_PROFILE_MATCH_CPU_BUDGET=64
export CEDAR_PROFILE_MATCH_RAY_CPU_BUDGET=64

declare -A DATASET_FILE DATASET_KWARGS EPOCHS SAMPLES BATCH
DATASET_FILE[commonvoice]="$MODULES/evaluation/pipelines/commonvoice/cedar_dataset.py"
DATASET_KWARGS[commonvoice]="dataset_path=/workspace/OptimalCedar/datasets/commonvoice/cv15_en_train_300000,max_samples=100000"
EPOCHS[commonvoice]=1 SAMPLES[commonvoice]=100000 BATCH[commonvoice]=1
DATASET_FILE[coco]="$MODULES/evaluation/pipelines/coco/cedar_dataset.py"
DATASET_KWARGS[coco]="dataset_path=/workspace/OptimalCedar/evaluation/datasets/coco,split=train2017"
EPOCHS[coco]=1 SAMPLES[coco]=20000 BATCH[coco]=1
DATASET_FILE[wikitext103]="$MODULES/evaluation/pipelines/wikitext103/cedar_dataset.py"
DATASET_KWARGS[wikitext103]="dataset_path=/workspace/OptimalCedar/evaluation/datasets/wikitext103,max_samples=100000"
EPOCHS[wikitext103]=1 SAMPLES[wikitext103]=100000 BATCH[wikitext103]=1

run_cell() {
  local workload=$1 label=$2 optimizers=$3 repeats=$4
  local timeout=${5:-$PLAN_TIMEOUT}
  local work="$OUT/$workload"
  local result="$work/results/${label}.json"
  local log="$work/logs/${label}.log"
  mkdir -p "$work/logs" "$work/results" "$work/plans" "$work/cache"
  if [ -s "$result" ]; then
    echo "SKIP $workload/$label (result exists)"
    return 0
  fi
  local profile="$PROFILE_DIR/${workload}/shared.yaml"
  if [ ! -s "$profile" ]; then
    echo "BLOCKED $workload/$label: no profile at $profile"
    echo "{\"status\":\"blocked_missing_profile\"}" > "$work/results/${label}.failed.json"
    return 0
  fi
  echo "CELL $workload/$label optimizers=$optimizers repeats=$repeats $(date -Is)"
  python -u "$ENTRY" "$MODULES/evaluation/compare_optimizer_perf.py" \
    --dataset_file "${DATASET_FILE[$workload]}" \
    --dataset_kwargs "${DATASET_KWARGS[$workload]}" \
    --batch_size "${BATCH[$workload]}" --num_epochs "${EPOCHS[$workload]}" \
    --num_total_samples "${SAMPLES[$workload]}" \
    --use_ray --ray_ip "$RAY_IP" \
    --profiled_stats "$profile" \
    --full_data_run --enable_local_parallelism --match_profile_resources \
    --cpu_budget 64 --ray_cpu_budget 64 \
    --optimizers ${optimizers//,/ } \
    --optimizer_time_limit_sec "$timeout" \
    --cedar_reorder_timeout_sec "$timeout" \
    --disable_cedar_runtime_timeout --num_repeats "$repeats" \
    --skip_pico_plan_cost --disable_caching \
    --results_path "$result" > "$log" 2>&1
  local code=$?
  echo "CELL-DONE $workload/$label exit=$code $(date -Is)"
  if [ $code -ne 0 ]; then
    echo "{\"status\":\"failed\",\"exit\":$code}" > "$work/results/${label}.failed.json"
  else
    rm -f "$work/results/${label}.failed.json"
  fi
  return 0
}

echo "=== missing cells start $(date -Is) commit=$(git rev-parse --short HEAD) ==="

for workload in "$@"; do
  case "$workload" in
    commonvoice)
      run_cell commonvoice main_fast "pico_final,optimizer,plumber_optimizer" 1
      run_cell commonvoice ablation_fast "pico_final,pico_byte_proportional" 1
      ;;
    coco)
      run_cell coco main_fast "pico_final,optimizer" 1
      ;;
    wikitext103)
      run_cell wikitext103 main_fast "pico_final,optimizer,plumber_optimizer" 1
      ;;
    *)
      echo "unknown workload $workload"
      ;;
  esac
done

echo "=== missing cells done $(date -Is) ==="
