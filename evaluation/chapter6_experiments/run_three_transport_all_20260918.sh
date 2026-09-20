#!/usr/bin/env bash
set -euo pipefail
cd /workspace/OptimalCedar
source env/bin/activate
export PYTHONDONTWRITEBYTECODE=1
python -u evaluation/chapter6_experiments/run_simple_dp_ablation_matrix.py \
  --output outputs/simple_dp_three_transport_all_20260918 \
  --workloads simclrv2 simclrv2_cache commonvoice coco llava_pretrain stackexchange \
  --methods simple_dp_workers_boundary simple_dp_workers_width_boundary simple_dp_boundary \
  --repeats 1 --cell-timeout-sec 7200 --layered-profile --smp-aggregate-profile \
  --commonvoice-max-samples 15000 \
  --commonvoice-dataset-path /workspace/OptimalCedar/datasets/commonvoice/cv15_en_train_300000
