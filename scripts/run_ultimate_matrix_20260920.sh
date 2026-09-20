#!/usr/bin/env bash
# Final campaign, matrix only: the profiles already exist for these exact
# workload definitions (local imagenette2, COCO train2017, the 558k LLaVA and
# 400k StackExchange corpora, 300k CommonVoice clips), so the run reuses them
# and spends its time on the 9 optimizers x 6 workloads instead.
#
# simclrv2 and simclrv2_cache replay the local 9,469-image training split 20
# times (189,380 records) instead of downloading ImageNet-1k, per instruction.
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO" || exit 1
source "$REPO/env/bin/activate"

RUN=outputs/ultimate_eight_optimizers_20260920
PROFILE_SOURCES="outputs/six_workload_profile_v3_20260919 outputs/llava_profile_check_20260919"

nohup python -u evaluation/chapter6_experiments/run_simple_dp_ablation_matrix.py \
    --output "$RUN" \
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
