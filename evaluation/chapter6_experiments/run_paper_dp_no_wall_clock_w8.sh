#!/usr/bin/env bash
# Re-measure the DP optimizer workloads affected by removal of the wall-clock
# cost correction, then generate comparison figures in an isolated run folder.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_DIR="${REPO_ROOT}/evaluation/chapter6_experiments"
FORMAL_DIR="${BASE_DIR}/formal_results"
ARCHIVE_ROOT="${FORMAL_DIR}/paper_optimizer_w8"
OUTPUT_ROOT="${OUTPUT_ROOT:-${BASE_DIR}/paper_dp_no_wall_clock_w8_run}"
PROFILE_DIR="${OUTPUT_ROOT}/profiles"
WORKLOAD_ROOT="${OUTPUT_ROOT}/workloads"
FIGURE_DIR="${OUTPUT_ROOT}/figures"

cd "${REPO_ROOT}"
source env/bin/activate

mkdir -p "${PROFILE_DIR}"
cp "${ARCHIVE_ROOT}/profiles/coco.yaml" "${PROFILE_DIR}/coco.yaml"
cp "${ARCHIVE_ROOT}/profiles/simclrv2.yaml" \
  "${PROFILE_DIR}/simclrv2.yaml"
cp "${ARCHIVE_ROOT}/profiles/simclrv2_cache.yaml" \
  "${PROFILE_DIR}/simclrv2_cache.yaml"

PROFILE_DIR="${PROFILE_DIR}" \
MATRIX_OUTPUT_ROOT="${WORKLOAD_ROOT}" \
CPU_BUDGET=64 \
LOCAL_WORKERS=8 \
REPEATS=3 \
OPTIMIZER_PLAN_TIMEOUT_SEC=300 \
OPTIMIZER_SET=dp_only \
RESUME_EXISTING="${RESUME_EXISTING:-0}" \
COCO_SAMPLES=50000 \
COCO_DATASET_KWARGS=split=train2017 \
bash "${BASE_DIR}/run_formal_plan_and_matrix.sh" \
  --workloads coco,simclrv2,simclrv2_cache

python "${BASE_DIR}/plot_latest_optimizer_dp_cedar_baseline.py" \
  --candidate-report "${ARCHIVE_ROOT}/data/data_pipeline_matrix.json" \
  --scaled-run "${ARCHIVE_ROOT}/data/enlarged_core" \
  --paper-matrix "${ARCHIVE_ROOT}/data/standard_core" \
  --dp-replacement-matrix "${WORKLOAD_ROOT}" \
  --output-dir "${FIGURE_DIR}"

echo "[$(date -Is)] DP no-wall-clock experiment and figures complete"
