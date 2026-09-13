#!/usr/bin/env bash
set -euo pipefail

cd /workspace/OptimalCedar
source env/bin/activate

readonly OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/OptimalCedar/outputs/motivation_multimodal/simclrv2_multimodal_modality_scaling_formal_20260909}"
readonly FIXTURE="/workspace/OptimalCedar/outputs/motivation_multimodal/fixture_pilot2000/scaling.jsonl"
readonly IMAGE_ROOT="/workspace/OptimalCedar/datasets/coco/val2017"

mkdir -p "${OUTPUT_ROOT}"
rm -f "${OUTPUT_ROOT}/COMPLETE" "${OUTPUT_ROOT}/FAILED"
trap 'touch "${OUTPUT_ROOT}/FAILED"' ERR

python -m evaluation.pipelines.simclrv2_multimodal.modality_scaling \
    --fixture "${FIXTURE}" \
    --image-root "${IMAGE_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --records "${RECORDS:-128}" \
    --repetitions "${REPETITIONS:-7}"

touch "${OUTPUT_ROOT}/COMPLETE"
rm -f "${OUTPUT_ROOT}/FAILED"
