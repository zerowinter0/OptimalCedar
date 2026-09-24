#!/usr/bin/env bash
# Final PICO (W-only, representation-aware compute) campaign.
#
# Runs, per workload, the five-way main comparison plus the model ablation and
# the W-scaling cells, with resume, round-robin repeats and per-cell logs.
# Every cell goes through the same harness the previous campaigns used, so the
# results are directly comparable and every raw artefact is kept.
#
# Usage (inside the container, via tmux or nohup):
#   bash scripts/pico_final_w_only_campaign.sh [workloads...]
set -uo pipefail
cd /workspace/OptimalCedar
source env/bin/activate

PROFILE_DIR=${PROFILE_DIR:-outputs/affine_repr_profile_20260924}
OUT=${OUT:-outputs/pico_final_w_only_20260924}
# entry.py resolves ``modules`` relative to itself, and the driver must import
# the dataset module from the *same* tree that is shipped to the Ray workers.
ENTRY="$OUT/entry.py"
MODULES="$OUT/modules"
RAY_IP=172.23.166.105:6379
ROUNDS=${ROUNDS:-3}
PLAN_TIMEOUT=${PLAN_TIMEOUT:-3600}
CELL_TIMEOUT=${CELL_TIMEOUT:-7200}

mkdir -p "$OUT" "$OUT/logs" "$OUT/results" "$OUT/plans" "$OUT/profiles"
cp -f outputs/ultimate_eight_optimizers_fix_20260921/entry.py "$OUT/entry.py"
if [ ! -d "$OUT/modules" ]; then
  mkdir -p "$OUT/modules"
  cp -r cedar "$OUT/modules"/cedar
  mkdir -p "$OUT/modules/evaluation"
  find evaluation -maxdepth 1 -type f -name '*.py' -exec cp {} "$OUT/modules/evaluation/" \;
  cp -r evaluation/pipelines "$OUT/modules/evaluation/pipelines"
  find "$OUT/modules" -name '__pycache__' -type d -prune -exec rm -rf {} +
fi

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CEDAR_RAY_PLACEMENT_RESOURCE=cedar_remote CEDAR_RAY_REQUIRE_REMOTE=1
export CEDAR_WORKER_READY_TIMEOUT_SEC=600
export CEDAR_MATCH_PROFILE_RESOURCES=1
export CEDAR_PROFILE_MATCH_CPU_BUDGET=64
export CEDAR_PROFILE_MATCH_RAY_CPU_BUDGET=64

declare -A DATASET_FILE DATASET_KWARGS EPOCHS SAMPLES
DATASET_FILE[simclrv2]="$MODULES/evaluation/pipelines/simclrv2/cedar_dataset.py"
DATASET_KWARGS[simclrv2]="dataset_path=/workspace/OptimalCedar/evaluation/datasets/imagenette2/imagenette2/train"
EPOCHS[simclrv2]=${SIMCLRV2_EPOCHS:-20} SAMPLES[simclrv2]=${SIMCLRV2_SAMPLES:-189380}
DATASET_FILE[simclrv2_cache]="$MODULES/evaluation/pipelines/simclrv2/cedar_cache_dataset.py"
DATASET_KWARGS[simclrv2_cache]="dataset_path=/workspace/OptimalCedar/evaluation/datasets/imagenette2/imagenette2/train"
EPOCHS[simclrv2_cache]=20 SAMPLES[simclrv2_cache]=189380
DATASET_FILE[commonvoice]="$MODULES/evaluation/pipelines/commonvoice/cedar_dataset.py"
DATASET_KWARGS[commonvoice]="dataset_path=/workspace/OptimalCedar/datasets/commonvoice/cv-corpus-15.0-delta-2023-09-08/en/clips,max_samples=300000"
EPOCHS[commonvoice]=1 SAMPLES[commonvoice]=300000
DATASET_FILE[coco]="$MODULES/evaluation/pipelines/coco/cedar_dataset.py"
DATASET_KWARGS[coco]="dataset_path=/workspace/OptimalCedar/evaluation/datasets/coco,split=train2017"
EPOCHS[coco]=1 SAMPLES[coco]=50000
DATASET_FILE[llava_pretrain]="$MODULES/evaluation/pipelines/multimodal_running_example/cedar_dataset.py"
DATASET_KWARGS[llava_pretrain]="dataset_path=/workspace/OptimalCedar/outputs/ultimate_eight_optimizers_fix_20260921/inputs/llava_pretrain.jsonl,image_root=/workspace/OptimalCedar/evaluation/datasets/llava_pretrain"
EPOCHS[llava_pretrain]=1 SAMPLES[llava_pretrain]=50000
DATASET_FILE[wikitext103]="$MODULES/evaluation/pipelines/wikitext103/cedar_dataset.py"
DATASET_KWARGS[wikitext103]="dataset_path=/workspace/OptimalCedar/datasets/wikitext103"
EPOCHS[wikitext103]=1 SAMPLES[wikitext103]=0

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
  echo "CELL $workload/$label optimizers=$optimizers repeats=$repeats $(date -Is)"
  python -u "$OUT/entry.py" "$OUT/modules/evaluation/compare_optimizer_perf.py" \
    --dataset_file "${DATASET_FILE[$workload]}" \
    --dataset_kwargs "${DATASET_KWARGS[$workload]}" \
    --batch_size 4 --num_epochs "${EPOCHS[$workload]}" \
    --num_total_samples "${SAMPLES[$workload]}" \
    --use_ray --ray_ip "$RAY_IP" \
    --profiled_stats "$PROFILE_DIR/${workload}/shared.yaml" \
    --full_data_run --enable_local_parallelism --match_profile_resources \
    --cpu_budget 64 --ray_cpu_budget 64 \
    --optimizers ${optimizers//,/ } \
    --optimizer_time_limit_sec "$PLAN_TIMEOUT" \
    --cedar_reorder_timeout_sec "$PLAN_TIMEOUT" \
    --disable_cedar_runtime_timeout --num_repeats "$repeats" \
    --skip_pico_plan_cost --disable_caching \
    --results_path "$result" \
    > "$log" 2>&1
  local code=$?
  echo "CELL-DONE $workload/$label exit=$code $(date -Is)"
  if [ $code -ne 0 ]; then
    echo "{\"status\":\"failed\",\"exit\":$code,\"log\":\"$log\"}" \
      > "$work/results/${label}.failed.json"
  fi
}

WORKLOADS=("$@")
if [ ${#WORKLOADS[@]} -eq 0 ]; then
  WORKLOADS=(simclrv2 simclrv2_cache commonvoice coco llava_pretrain)
fi

for workload in "${WORKLOADS[@]}"; do
  profile="$PROFILE_DIR/${workload}/shared.yaml"
  if [ ! -f "$profile" ]; then
    echo "NO PROFILE for $workload ($profile); skipping"
    continue
  fi
  # B1/B2: main comparison (final PICO + native baselines + unoptimized).
  run_cell "$workload" main_mean \
    "pico_final,optimizer,plumber_optimizer,raydata_optimizer,unopti" "$ROUNDS"
  # C1: model ablation under the same W-only search.
  run_cell "$workload" ablation_model \
    "pico_final,pico_final_no_boundary,pico_byte_proportional,simple_dp_workers_boundary" "$ROUNDS"
  # C2: staged versus joint for the final model.
  run_cell "$workload" staged_joint \
    "pico_final,staged_final,staged_workers_boundary_affine" "$ROUNDS"
done

echo "CAMPAIGN-FINISHED $(date -Is)"
