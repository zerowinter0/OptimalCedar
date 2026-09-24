#!/usr/bin/env bash
# Stage B: swap only the operator compute model inside PICO.
#
# Both cells read the *same* newly generated profile, so the comparison isolates
# the compute model (byte affine -> representation-aware affine); everything
# else (data, threads, CPU budget, harness) is the campaign configuration.
#
# Usage (inside the container):
#   bash scripts/run_stage_b_pico_repr_20260924.sh <profile.yaml> <out_dir> [repeats]
set -uo pipefail
cd /workspace/OptimalCedar
source env/bin/activate

PROFILE=${1:?profile path required}
OUT=${2:?output dir required}
REPEATS=${3:-3}
DATA=/workspace/OptimalCedar/evaluation/datasets/imagenette2/imagenette2/train
# Ship a snapshot of the *current* source to the Ray workers; the campaign
# snapshot predates the representation-aware optimizers.  entry.py resolves
# ``modules`` relative to itself, so the copy must live in $OUT.
MODULES="$OUT/modules"
ENTRY="$OUT/entry.py"
RAY_IP=172.23.166.105:6379

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CEDAR_RAY_PLACEMENT_RESOURCE=cedar_remote CEDAR_RAY_REQUIRE_REMOTE=1
export CEDAR_WORKER_READY_TIMEOUT_SEC=600
export CEDAR_MATCH_PROFILE_RESOURCES=1
export CEDAR_PROFILE_MATCH_CPU_BUDGET=64
export CEDAR_PROFILE_MATCH_RAY_CPU_BUDGET=64
mkdir -p "$OUT"
cp -f outputs/ultimate_eight_optimizers_fix_20260921/entry.py "$OUT/entry.py"
rm -rf "$MODULES"
mkdir -p "$MODULES"
cp -r cedar "$MODULES"/cedar
mkdir -p "$MODULES/evaluation"
find evaluation -maxdepth 1 -type f -name '*.py' -exec cp {} "$MODULES/evaluation/" \;
cp -r evaluation/pipelines "$MODULES/evaluation/pipelines"
find "$MODULES" -name '__pycache__' -type d -prune -exec rm -rf {} +

run_pair() {
  local label=$1 optimizers=$2 repeats=$3
  local epochs=${4:-1}
  echo "RUN $label repeats=$repeats optimizers=$optimizers"
  python -u "$ENTRY" "$MODULES/evaluation/compare_optimizer_perf.py" \
    --dataset_file "$MODULES/evaluation/pipelines/simclrv2/cedar_dataset.py" \
    --dataset_kwargs "dataset_path=$DATA" \
    --batch_size 4 --num_epochs "$epochs" --num_total_samples 9469 \
    --use_ray --ray_ip "$RAY_IP" \
    --profiled_stats "$PROFILE" \
    --full_data_run --enable_local_parallelism --match_profile_resources \
    --cpu_budget 64 --ray_cpu_budget 64 \
    --optimizers ${optimizers//,/ } \
    --optimizer_time_limit_sec 7200 --cedar_reorder_timeout_sec 7200 \
    --disable_cedar_runtime_timeout --num_repeats "$repeats" \
    --skip_pico_plan_cost --disable_caching \
    --results_path "$OUT/results_$label.json" \
    > "$OUT/${label}.log" 2>&1
  echo "DONE $label"
}

# Cheap pair: the same DP search with and without the representation-aware
# compute model (byte affine -> element/representation affine).
run_pair cheap simple_dp_boundary,simple_dp_repr_affine "$REPEATS" 1
# Full PICO pair: workers x width search, only the compute model differs.
run_pair pico simple_dp_workers_width_boundary,simple_dp_workers_width_repr_affine 3 1
