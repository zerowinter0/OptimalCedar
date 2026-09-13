#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
source /workspace/OptimalCedar/env/bin/activate

OUT="outputs/motivation_multimodal/smoke"
DATASET=".multimodal_smoke_input.jsonl"
THRESHOLD=".multimodal_smoke_threshold.json"
IMAGE_ROOT="/workspace/OptimalCedar/datasets/coco/val2017"
PROFILE="$OUT/profile.yml"
OPTIMIZERS=(
  staged_rfo staged_rof staged_fro staged_for staged_orf staged_ofr joint
)

export W=1
export CPU_BUDGET=64
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p "$OUT/plans" "$OUT/results" "$OUT/logs"
rm -f "$OUT/SUCCESS" "$OUT/FAILED"
echo "$$" > "$OUT/workflow.pid"
trap 'status=$?; if [[ $status -ne 0 ]]; then echo "$status" > "$OUT/FAILED"; fi' EXIT

progress() {
  printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*"
}

for optimizer in "${OPTIMIZERS[@]}"; do
  progress "generating $optimizer"
  python -m evaluation.motivation_multimodal.runner plan \
    --optimizer "$optimizer" \
    --dataset "$DATASET" \
    --threshold "$THRESHOLD" \
    --image-root "$IMAGE_ROOT" \
    --profile "$PROFILE" \
    --plan "$OUT/plans/$optimizer.yml" \
    --num-samples 2 \
    --result "$OUT/results/plan_$optimizer.json" \
    > "$OUT/logs/plan_$optimizer.log" 2>&1
  progress "generated $optimizer"
done

score_args=()
for optimizer in "${OPTIMIZERS[@]}"; do
  score_args+=(--named-plan "$optimizer=$OUT/plans/$optimizer.yml")
done
progress "scoring all plans with Cedar calculate_cost"
python -m evaluation.motivation_multimodal.runner score \
  --dataset "$DATASET" \
  --threshold "$THRESHOLD" \
  --image-root "$IMAGE_ROOT" \
  --profile "$PROFILE" \
  --num-samples 2 \
  "${score_args[@]}" \
  --result "$OUT/results/cedar_costs.json" \
  > "$OUT/logs/cedar_costs.log" 2>&1
progress "scored all plans"

for optimizer in "${OPTIMIZERS[@]}"; do
  progress "executing $optimizer"
  python -m evaluation.motivation_multimodal.runner execute \
    --dataset "$DATASET" \
    --threshold "$THRESHOLD" \
    --image-root "$IMAGE_ROOT" \
    --plan "$OUT/plans/$optimizer.yml" \
    --num-samples 2 \
    --result "$OUT/results/run_$optimizer.json" \
    > "$OUT/logs/run_$optimizer.log" 2>&1
  progress "executed $optimizer"
done

echo success > "$OUT/SUCCESS"
progress "smoke workflow completed"
