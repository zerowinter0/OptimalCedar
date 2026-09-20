#!/usr/bin/env bash
# End-to-end final campaign: fetch ImageNet-1k, extract it, then profile and
# run every optimizer on the six workloads with the scaled data volumes.
#
# Stages (all resumable, everything is logged):
#   1. ImageNet-1k download (segmented, restarts safely)
#   2. extraction to train/<wnid>/ + flat val/
#   3. profile + six-method matrix through run_simple_dp_ablation_matrix.py
#      with 8 optimizers, 2 h per cell, and the agreed skip list
set -uo pipefail

# Launch this script detached inside the code container:
#   docker exec -d optimalcedar-torch201-dev bash -lc \
#     'nohup /workspace/OptimalCedar/scripts/run_ultimate_experiment_20260920.sh &'
# A host-side background process is reaped when the tool session ends, a
# detached container process is not.
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO" || exit 1
source "$REPO/env/bin/activate"
RUN=outputs/ultimate_eight_optimizers_20260920
IMAGENET=$REPO/datasets/imagenet/ILSVRC2012
IMAGENET_CONTAINER=/workspace/OptimalCedar/datasets/imagenet/ILSVRC2012
LOG=$IMAGENET/pipeline.log

mkdir -p "$IMAGENET"
exec >>"$LOG" 2>&1
log() { echo "[$(date '+%F %T')] $*"; }
log "=== ultimate experiment pipeline start ==="

# ---------------------------------------------------------------- 1. download
# The campaign needs 189,380 ILSVRC2012 training images (20x the imagenette2
# training split). Whole classes are fetched straight out of the remote tar
# with HTTP range requests, which costs ~22 GB instead of the full 147.9 GB.
log "stage 1: fetching the ImageNet-1k class subset (189,380 images)"
python "$REPO/scripts/download_imagenet1k_subset.py" \
    --dest "$IMAGENET" --images 189380 \
    || { log "FATAL: subset download failed"; exit 1; }

# ---------------------------------------------------------------- 2. extract
log "stage 2: extracting archives"
bash "$REPO/scripts/extract_imagenet1k.sh"
train_images=$(find "$IMAGENET/train" -type f 2>/dev/null | wc -l)
log "stage 2: $train_images train images available"
if [ "$train_images" -lt 189380 ]; then
    log "FATAL: fewer train images than the 189,380 the campaign needs"
    exit 1
fi

# --------------------------------------------------- 3. profile + experiment
log "stage 3: launching the eight-optimizer matrix (profile included)"
nohup python -u evaluation/chapter6_experiments/run_simple_dp_ablation_matrix.py \
    --output $RUN \
    --workloads simclrv2 simclrv2_cache commonvoice coco llava_pretrain stackexchange \
    --methods cedar-opt plumber-opt ray-opt unopti old_dp_boundary simple_dp_boundary \
              simple_dp_workers_width_boundary simple-dp-opt old-dp-opt \
    --repeats 1 --cell-timeout-sec 7200 \
    --layered-profile --smp-aggregate-profile \
    --simclrv2-dataset $IMAGENET_CONTAINER/train \
    --coco-split train2017 \
    --commonvoice-max-samples 300000 \
    --commonvoice-dataset-path /workspace/OptimalCedar/datasets/commonvoice/cv15_en_train_300000 \
    --llava-samples 50000 --stackexchange-samples 20000 \
    --llava-source /workspace/OptimalCedar/evaluation/datasets/llava_pretrain/blip_laion_cc_sbu_558k.jsonl \
    --stackexchange-source /workspace/OptimalCedar/datasets/stackexchange/redpajama-stackexchange-400000.jsonl \
    --skip-cedar-workloads llava_pretrain stackexchange \
    --skip-cell simple_dp_workers_width_boundary@llava_pretrain \
    --skip-cell simple_dp_workers_width_boundary@stackexchange \
    > $RUN.nohup.log 2>&1 &
log "stage 3: runner launched; follow $REPO/$RUN.nohup.log and $REPO/$RUN/status.json"
log "=== pipeline hand-off complete ==="
