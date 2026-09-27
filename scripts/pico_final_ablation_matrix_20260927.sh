#!/usr/bin/env bash
# Uniform six-arm cost-model ablation on every workload of the final delivery.
#
# Arms (all with the same profile file, remote Ray, 64 CPU budget and the
# fast-protocol data volume of the matching main cell):
#
#   pico_final              39  joint W-only DP + representation-aware affine
#                               compute + boundary          (the final model)
#   optimizer                0  Cedar native staged optimizer (baseline)
#   staged_final            42  Cedar staged search priced by the *final* PICO
#                               model (replaces Cedar's cost function)
#   old_dp_boundary         28  DP search priced by Cedar's legacy profile
#                               entries (baseline latency + whole-pipeline
#                               offload throughput) with Cedar's fixed+bytes
#                               boundary; no W search
#   pico_final_no_boundary  40  same W-only search and element/representation
#                               affine compute, explicit boundary term removed
#                               (operator-level change only)
#   pico_byte_proportional  41  byte-proportional compute (y = x in bytes, i.e.
#                               Cedar's operator layer) with the boundary model
#                               and the same W-only search
#
# LLaVA-Pretrain drops `optimizer` (stable timeout, user-approved).  Wikitext103
# runs in two cells because the three representation-aware arms cannot be priced
# on that profile (4 of 9 operators have no fitted curve) and a failing arm
# aborts its cell.
#
# Usage (inside the container, under tmux):
#   bash scripts/pico_final_ablation_matrix_20260927.sh [workloads...]
set -uo pipefail
cd /workspace/OptimalCedar
source env/bin/activate

PROFILE_DIR=${PROFILE_DIR:-outputs/affine_repr_profile_20260924}
OUT=${OUT:-outputs/pico_final_w_only_20260924}
RAY_IP=172.23.166.105:6379
PLAN_TIMEOUT=${PLAN_TIMEOUT:-2400}
CELL_TIMEOUT=${CELL_TIMEOUT:-7200}

ENTRY="$OUT/entry.py"
MODULES="$OUT/modules"
mkdir -p "$OUT/logs"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CEDAR_RAY_PLACEMENT_RESOURCE=cedar_remote CEDAR_RAY_REQUIRE_REMOTE=1
export CEDAR_WORKER_READY_TIMEOUT_SEC=600
export CEDAR_MATCH_PROFILE_RESOURCES=1
export CEDAR_PROFILE_MATCH_CPU_BUDGET=64
export CEDAR_PROFILE_MATCH_RAY_CPU_BUDGET=64

declare -A DATASET_FILE DATASET_KWARGS EPOCHS SAMPLES BATCH
DATASET_FILE[simclrv2]="$MODULES/evaluation/pipelines/simclrv2/cedar_dataset.py"
DATASET_KWARGS[simclrv2]="dataset_path=/workspace/OptimalCedar/evaluation/datasets/imagenette2/imagenette2/train"
EPOCHS[simclrv2]=8 SAMPLES[simclrv2]=75752 BATCH[simclrv2]=4
DATASET_FILE[simclrv2_cache]="$MODULES/evaluation/pipelines/simclrv2/cedar_cache_dataset.py"
DATASET_KWARGS[simclrv2_cache]="dataset_path=/workspace/OptimalCedar/evaluation/datasets/imagenette2/imagenette2/train"
EPOCHS[simclrv2_cache]=8 SAMPLES[simclrv2_cache]=75752 BATCH[simclrv2_cache]=4
DATASET_FILE[commonvoice]="$MODULES/evaluation/pipelines/commonvoice/cedar_dataset.py"
DATASET_KWARGS[commonvoice]="dataset_path=/workspace/OptimalCedar/datasets/commonvoice/cv15_en_train_300000,max_samples=100000"
EPOCHS[commonvoice]=1 SAMPLES[commonvoice]=100000 BATCH[commonvoice]=1
DATASET_FILE[coco]="$MODULES/evaluation/pipelines/coco/cedar_dataset.py"
DATASET_KWARGS[coco]="dataset_path=/workspace/OptimalCedar/evaluation/datasets/coco,split=train2017"
EPOCHS[coco]=1 SAMPLES[coco]=20000 BATCH[coco]=1
DATASET_FILE[llava_pretrain]="$MODULES/evaluation/pipelines/llava_pretrain/cedar_dataset.py"
DATASET_KWARGS[llava_pretrain]="dataset_path=/workspace/OptimalCedar/outputs/ultimate_eight_optimizers_fix_20260921/inputs/llava_pretrain.jsonl,image_root=/workspace/OptimalCedar/evaluation/datasets/llava_pretrain"
EPOCHS[llava_pretrain]=1 SAMPLES[llava_pretrain]=10000 BATCH[llava_pretrain]=1
DATASET_FILE[wikitext103]="$MODULES/evaluation/pipelines/wikitext103/cedar_dataset.py"
DATASET_KWARGS[wikitext103]="dataset_path=/workspace/OptimalCedar/evaluation/datasets/wikitext103,max_samples=100000"
EPOCHS[wikitext103]=1 SAMPLES[wikitext103]=100000 BATCH[wikitext103]=1

run_cell() {
  local workload=$1 label=$2 optimizers=$3 repeats=$4
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
    echo "BLOCKED $workload/$label: no profile"
    echo "{\"status\":\"blocked_missing_profile\"}" > "$work/results/${label}.failed.json"
    return 0
  fi
  echo "CELL $workload/$label optimizers=$optimizers $(date -Is)"
  timeout "$CELL_TIMEOUT" python -u "$ENTRY" "$MODULES/evaluation/compare_optimizer_perf.py" \
    --dataset_file "${DATASET_FILE[$workload]}" \
    --dataset_kwargs "${DATASET_KWARGS[$workload]}" \
    --batch_size "${BATCH[$workload]}" --num_epochs "${EPOCHS[$workload]}" \
    --num_total_samples "${SAMPLES[$workload]}" \
    --use_ray --ray_ip "$RAY_IP" \
    --profiled_stats "$profile" \
    --full_data_run --enable_local_parallelism --match_profile_resources \
    --cpu_budget 64 --ray_cpu_budget 64 \
    --optimizers ${optimizers//,/ } \
    --optimizer_time_limit_sec "$PLAN_TIMEOUT" \
    --cedar_reorder_timeout_sec "$PLAN_TIMEOUT" \
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

echo "=== ablation matrix start $(date -Is) commit=$(git rev-parse --short HEAD) ==="

for workload in "$@"; do
  case "$workload" in
    simclrv2|simclrv2_cache|commonvoice|coco)
      run_cell "$workload" ablation_matrix \
        "pico_final,optimizer,staged_final,old_dp_boundary,pico_final_no_boundary,pico_byte_proportional" 1
      ;;
    llava_pretrain)
      # Cedar's staged optimizer times out here (user-approved exclusion).
      run_cell "$workload" ablation_matrix \
        "pico_final,staged_final,old_dp_boundary,pico_final_no_boundary,pico_byte_proportional" 1
      ;;
    wikitext103)
      # Representation-aware arms cannot be priced on this profile; run them in
      # a separate cell so their failure does not abort the byte-model arms.
      run_cell "$workload" ablation_matrix_byte \
        "old_dp_boundary,pico_byte_proportional,optimizer" 1
      run_cell "$workload" ablation_matrix_repr \
        "pico_final,staged_final,pico_final_no_boundary" 1
      ;;
    *)
      echo "unknown workload $workload"
      ;;
  esac
done

echo "=== ablation matrix done $(date -Is) ==="
