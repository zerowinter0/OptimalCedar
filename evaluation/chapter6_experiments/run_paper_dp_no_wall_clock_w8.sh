#!/usr/bin/env bash
# Re-measure the DP optimizer workloads affected by removal of the wall-clock
# cost correction, then regenerate the canonical optimizer figures.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_DIR="${REPO_ROOT}/evaluation/chapter6_experiments"
FORMAL_DIR="${BASE_DIR}/formal_results"
OUTPUT_ROOT="${FORMAL_DIR}/paper_dp_no_wall_clock_w8"
PROFILE_DIR="${OUTPUT_ROOT}/profiles"
WORKLOAD_ROOT="${OUTPUT_ROOT}/workloads"
FIGURE_DIR="${FORMAL_DIR}/paper_figures_optimizer_w8_no_wall_clock"

cd "${REPO_ROOT}"
source env/bin/activate

mkdir -p "${PROFILE_DIR}"
cp "${FORMAL_DIR}/profiles/coco.yaml" "${PROFILE_DIR}/coco.yaml"
cp "${FORMAL_DIR}/paper_optimizer_w8/profiles/simclrv2.yaml" \
  "${PROFILE_DIR}/simclrv2.yaml"
cp "${FORMAL_DIR}/paper_optimizer_w8/profiles/simclrv2_cache.yaml" \
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
  --candidate-report "${FORMAL_DIR}/dp_20pct_goal_latest.json" \
  --scaled-run \
    "${FORMAL_DIR}/scaled_reuse_plan_runs/coco_cv_enlarged_w8_formal_20260727" \
  --paper-matrix "${FORMAL_DIR}/paper_optimizer_w8" \
  --dp-replacement-matrix "${WORKLOAD_ROOT}" \
  --output-dir "${FIGURE_DIR}"

echo "[$(date -Is)] DP no-wall-clock experiment and figures complete"
