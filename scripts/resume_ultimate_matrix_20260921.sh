#!/usr/bin/env bash
# Resume the scaled campaign on the fixed snapshot (teardown fix in
# cedar/client/dataset.py).  The new root was prepared with --prepare-only and
# then seeded with the completed cells of the paused campaign:
#   simclrv2, simclrv2_cache, commonvoice  -> reused as-is
#   coco  -> cedar-opt/plumber/ray/affine-W-width reused, the two pseudo
#            timeouts (old_dp_boundary, simple_dp_boundary) and the interrupted
#            simple-dp-opt cell re-run, unopti stays a genuine timeout (skipped)
#   llava_pretrain, stackexchange -> run from scratch
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO" || exit 1
source "$REPO/env/bin/activate"

RUN=outputs/ultimate_eight_optimizers_fix_20260921
PROFILE_SOURCES="outputs/six_workload_profile_v3_20260919 outputs/llava_profile_check_20260919"

nohup python -u evaluation/chapter6_experiments/run_simple_dp_ablation_matrix.py \
    --output "$RUN" --resume \
    --workloads simclrv2 simclrv2_cache commonvoice coco llava_pretrain stackexchange \
    --methods cedar-opt plumber-opt ray-opt unopti old_dp_boundary simple_dp_boundary \
              simple_dp_workers_width_boundary simple-dp-opt old-dp-opt \
    --repeats 1 --cell-timeout-sec 7200 \
    --layered-profile --smp-aggregate-profile \
    --profile-from $PROFILE_SOURCES \
    --simclrv2-epochs 20 \
    --coco-split train2017 \
    --commonvoice-max-samples 300000 \
    --commonvoice-dataset-path /workspace/OptimalCedar/datasets/commonvoice/cv15_en_train_300000 \
    --llava-samples 50000 --stackexchange-samples 20000 \
    --llava-source /workspace/OptimalCedar/evaluation/datasets/llava_pretrain/blip_laion_cc_sbu_558k.jsonl \
    --stackexchange-source /workspace/OptimalCedar/datasets/stackexchange/redpajama-stackexchange-400000.jsonl \
    --skip-cedar-workloads llava_pretrain stackexchange \
    --skip-cell simple_dp_workers_width_boundary@llava_pretrain \
    --skip-cell simple_dp_workers_width_boundary@stackexchange \
    > "$RUN.nohup.log" 2>&1 &
echo "launched pid=$! -> $RUN.nohup.log"
