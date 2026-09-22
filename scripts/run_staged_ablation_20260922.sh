#!/usr/bin/env bash
# Staged-vs-DP ablation on the formal campaign's five workloads.
#
# Cedar's staged search (reorder -> offload/fusion -> TF fusion -> stage
# widths) is kept as-is; only its plan evaluator is replaced by the matching
# PICO cost model, so each staged variant and its DP counterpart are priced by
# bit-identical code:
#
#   staged-boundary          <-> old_dp_boundary            (boundary + Cedar compute)
#   staged-boundary-affine   <-> simple_dp_boundary         (+ per-operator kx+b)
#   staged-boundary-affine-W <-> simple_dp_workers_width_boundary  (+ W)
#
# The profiles are the campaign's own shared profiles (copied, so the staged
# plans and the already-measured DP plans are modelled from one profile).
#
# Usage (inside the container):
#   MODE=validation bash scripts/run_staged_ablation_20260922.sh
#   MODE=formal     bash scripts/run_staged_ablation_20260922.sh
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO" || exit 1
source "$REPO/env/bin/activate"

ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
export CEDAR_CACHE_WARMUP_GRACE_SEC=${CEDAR_CACHE_WARMUP_GRACE_SEC:-900}

MODE=${MODE:-validation}
RESUME=${RESUME:-0}
WORKLOADS=${WORKLOADS:-"simclrv2 simclrv2_cache commonvoice coco llava_pretrain"}
METHODS="staged-boundary staged-boundary-affine staged-boundary-affine-W"
# The DP counterparts are re-run in the same campaign so every staged/DP pair
# shares one profile, one day and one harness configuration; the cost model of
# a pair is identical, so any difference is the search.
METHODS="$METHODS old_dp_boundary simple_dp_boundary simple_dp_workers_boundary"
PROFILE_SOURCES="outputs/ultimate_eight_optimizers_fix_20260921"

if [ "$MODE" = validation ]; then
  RUN=outputs/staged_ablation_validation_20260922
  EXTRA=(--cell-timeout-sec 1800 --simclrv2-epochs 1
         --commonvoice-max-samples 200
         --record-override coco=200
         --llava-samples 200)
else
  RUN=outputs/staged_ablation_20260922
  EXTRA=(--cell-timeout-sec 7200 --simclrv2-epochs 20
         --commonvoice-max-samples 300000
         --commonvoice-dataset-path /workspace/OptimalCedar/datasets/commonvoice/cv15_en_train_300000
         --llava-samples 50000)
fi

nohup python -u evaluation/chapter6_experiments/run_simple_dp_ablation_matrix.py \
    --output "$RUN" \
    --workloads $WORKLOADS \
    --methods $METHODS \
    --repeats 1 \
    --layered-profile --smp-aggregate-profile \
    --profile-from $PROFILE_SOURCES \
    --coco-split train2017 \
    --llava-source /workspace/OptimalCedar/evaluation/datasets/llava_pretrain/blip_laion_cc_sbu_558k.jsonl \
    "${EXTRA[@]}" \
    $( [ "$RESUME" = 1 ] && echo --resume ) \
    > "$RUN.nohup.log" 2>&1 &
echo "launched mode=$MODE pid=$! -> $RUN.nohup.log"
