#!/usr/bin/env bash
# 7-hour budget version of the final W-only campaign.
#
# Deviations from the frozen three-round protocol (user-approved relaxation):
#   * unoptimized (unopti) dropped on every workload;
#   * Ray Data dropped on CommonVoice, and CommonVoice runs at 150k records;
#   * one complete round for the slow native baselines, two for the DP pair;
#   * model ablation on SimCLRv2 + SimCLRv2-cache + CommonVoice (one round);
#   * staged-vs-joint and the W sweep on SimCLRv2 only;
#   * only COCO needs a new profile; its compute-curve window is shortened.
# Every cell is still a complete run of the full workload subset, round-robin
# where a cell has more than one repeat, and every deviation is recorded in the
# delivery README/protocol.
#
# Usage (inside the container, under tmux):
#   bash scripts/pico_final_fast_20260925.sh
set -uo pipefail
cd /workspace/OptimalCedar
source env/bin/activate

PROFILE_DIR=${PROFILE_DIR:-outputs/affine_repr_profile_20260924}
OUT=${OUT:-outputs/pico_final_w_only_20260924}
RAY_IP=172.23.166.105:6379
PLAN_TIMEOUT=${PLAN_TIMEOUT:-2400}

mkdir -p "$OUT/logs" "$OUT/results"
cp -f outputs/ultimate_eight_optimizers_fix_20260921/entry.py "$OUT/entry.py"
if [ ! -d "$OUT/modules/cedar" ]; then
  rm -rf "$OUT/modules"
  mkdir -p "$OUT/modules/evaluation"
  cp -r cedar "$OUT/modules"/cedar
  find evaluation -maxdepth 1 -type f -name '*.py' -exec cp {} "$OUT/modules/evaluation/" \;
  cp -r evaluation/pipelines "$OUT/modules/evaluation/pipelines"
  find "$OUT/modules" -name '__pycache__' -type d -prune -exec rm -rf {} +
fi
MODULES="$OUT/modules"

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
EPOCHS[simclrv2]=20 SAMPLES[simclrv2]=189380
DATASET_FILE[simclrv2_cache]="$MODULES/evaluation/pipelines/simclrv2/cedar_cache_dataset.py"
DATASET_KWARGS[simclrv2_cache]="dataset_path=/workspace/OptimalCedar/evaluation/datasets/imagenette2/imagenette2/train"
EPOCHS[simclrv2_cache]=20 SAMPLES[simclrv2_cache]=189380
DATASET_FILE[commonvoice]="$MODULES/evaluation/pipelines/commonvoice/cedar_dataset.py"
DATASET_KWARGS[commonvoice]="dataset_path=/workspace/OptimalCedar/datasets/commonvoice/cv-corpus-15.0-delta-2023-09-08/en/clips,max_samples=150000"
EPOCHS[commonvoice]=1 SAMPLES[commonvoice]=150000
DATASET_FILE[coco]="$MODULES/evaluation/pipelines/coco/cedar_dataset.py"
DATASET_KWARGS[coco]="dataset_path=/workspace/OptimalCedar/evaluation/datasets/coco,split=train2017"
EPOCHS[coco]=1 SAMPLES[coco]=20000
DATASET_FILE[wikitext103]="$MODULES/evaluation/pipelines/wikitext103/cedar_dataset.py"
DATASET_KWARGS[wikitext103]="dataset_path=/workspace/OptimalCedar/evaluation/datasets/wikitext103,max_samples=20000"
EPOCHS[wikitext103]=1 SAMPLES[wikitext103]=20000

run_cell() {
  local workload=$1 label=$2 optimizers=$3 repeats=$4
  local work="$OUT/$workload"
  local result="$work/results/${label}.json"
  local log="$work/logs/${label}.log"
  mkdir -p "$work/logs" "$work/results" "$work/plans" "$work/cache"
  if [ -s "$result" ]; then
    echo "SKIP $workload/$label"
    return 0
  fi
  echo "CELL $workload/$label optimizers=$optimizers repeats=$repeats $(date -Is)"
  python -u "$OUT/entry.py" "$MODULES/evaluation/compare_optimizer_perf.py" \
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
    --results_path "$result" > "$log" 2>&1
  local code=$?
  echo "CELL-DONE $workload/$label exit=$code $(date -Is)"
  [ $code -ne 0 ] && echo "{\"status\":\"failed\",\"exit\":$code}" > "$work/results/${label}.failed.json"
  return 0
}

echo "=== fast campaign start $(date -Is) ==="

# COCO needs a fresh profile (its earlier attempt died on a non-spatial tensor,
# now guarded).  Shorten the compute-curve window to keep it inside the budget.
if [ ! -s "$PROFILE_DIR/coco/shared.yaml" ]; then
  echo "PROFILE coco (fast budget) $(date -Is)"
  OUT="$PROFILE_DIR" CEDAR_PROFILE_COMPUTE_TARGET_SEC=1.5 \
    CEDAR_ADAPTIVE_PROFILE_MIN_SEC=2 CEDAR_ADAPTIVE_PROFILE_MAX_SEC=10 \
    bash scripts/pico_final_profiles_20260924.sh coco
fi

# SimCLRv2: main comparison (1 round, no unopti) + one extra DP repeat,
# model ablation, staged-vs-joint, W sweep.
run_cell simclrv2 main_fast "pico_final,optimizer,plumber_optimizer,raydata_optimizer" 1
run_cell simclrv2 dp_repeat "pico_final,optimizer" 2
run_cell simclrv2 ablation_fast "pico_final,pico_final_no_boundary,pico_byte_proportional,simple_dp_workers_boundary" 1
run_cell simclrv2 staged_fast "pico_final,staged_final" 1
plan=$(ls -t "$OUT"/simclrv2/plans/*.yaml 2>/dev/null | head -1 || true)
if [ -n "${plan:-}" ]; then
  OUT="$OUT" ROUNDS=1 SAMPLES=9469 bash scripts/pico_final_w_scaling_20260924.sh simclrv2 "$plan" || true
fi

# SimCLRv2-cache: main comparison + ablation.
run_cell simclrv2_cache main_fast "pico_final,optimizer,plumber_optimizer,raydata_optimizer" 1
run_cell simclrv2_cache ablation_fast "pico_final,pico_final_no_boundary,pico_byte_proportional,simple_dp_workers_boundary" 1

# CommonVoice: main comparison without Ray Data, at 150k records, plus ablation.
run_cell commonvoice main_fast "pico_final,optimizer,plumber_optimizer" 1
run_cell commonvoice ablation_fast "pico_final,pico_byte_proportional" 1

# COCO (detection; has non-spatial payloads): DP pair only.
run_cell coco main_fast "pico_final,optimizer" 1

echo "=== assembly $(date -Is) ==="
python -u tmp_analysis/assemble_final_delivery.py || true
echo "=== FAST CAMPAIGN DONE $(date -Is) ==="
