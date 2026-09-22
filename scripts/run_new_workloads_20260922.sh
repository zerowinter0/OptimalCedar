#!/usr/bin/env bash
# Add Cedar's own workloads to the formal campaign:
#   WikiText-103 (plain / cache / TF), CommonVoice-cache, SimCLR-v2-TF, COCO-TF.
#
# MODE=validation  (default) runs every cell on a few hundred records with a
#                  30-minute cell limit, to prove the pipelines, the layered
#                  profile and all nine optimizers work on the new workloads.
# MODE=formal      runs the same matrix at the campaign's data volumes with the
#                  two-hour cell limit.
#
# Usage (inside the container):
#   MODE=validation bash scripts/run_new_workloads_20260922.sh
#   MODE=formal     bash scripts/run_new_workloads_20260922.sh
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO" || exit 1
source "$REPO/env/bin/activate"

# Cache-enabled workloads run 64 local workers that each keep a shard of
# open cache files; the container's default soft limit (1024) is exhausted and
# workers die with "OSError: [Errno 24] Too many open files".
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true

# The budget-limited cache warmup can stop while workers are still draining
# their shard; give the slowest shard more time to commit before the warmup is
# declared incomplete (see docs/experiments.md §3.5; 900 s did not rescue
# old_dp_boundary / simple_dp_workers_width_boundary on commonvoice_cache).
export CEDAR_CACHE_WARMUP_GRACE_SEC=${CEDAR_CACHE_WARMUP_GRACE_SEC:-900}

MODE=${MODE:-validation}
RESUME=${RESUME:-0}
# The TF variants stay opt-in: their remote actors need the HF/tokenizer files
# that only exist on the driver host, see docs/experiments.md §3.6.
NEW_WORKLOADS=${NEW_WORKLOADS:-"wikitext103 wikitext103_cache commonvoice_cache"}
METHODS="cedar-opt plumber-opt ray-opt unopti old_dp_boundary simple_dp_boundary \
         simple_dp_workers_width_boundary simple-dp-opt old-dp-opt"

if [ "$MODE" = validation ]; then
  RUN=outputs/new_workloads_validation_20260922
  EXTRA=(--cell-timeout-sec 1800 --simclrv2-epochs 1 --commonvoice-max-samples 200
         --record-override wikitext103=200
         --record-override wikitext103_cache=200
         --record-override wikitext103_tf=200
         --record-override commonvoice_cache=200
         --record-override simclrv2_tf=32
         --record-override coco_tf=200)
else
  RUN=outputs/ultimate_new_workloads_20260922
  EXTRA=(--cell-timeout-sec 7200 --simclrv2-epochs 20 --commonvoice-max-samples 300000
         --commonvoice-dataset-path /workspace/OptimalCedar/datasets/commonvoice/cv15_en_train_300000)
fi

nohup python -u evaluation/chapter6_experiments/run_simple_dp_ablation_matrix.py \
    --output "$RUN" \
    --workloads $NEW_WORKLOADS \
    --methods $METHODS \
    --repeats 1 \
    --layered-profile --smp-aggregate-profile \
    "${EXTRA[@]}" \
    $( [ "$RESUME" = 1 ] && echo --resume ) \
    > "$RUN.nohup.log" 2>&1 &
echo "launched mode=$MODE pid=$! -> $RUN.nohup.log"
