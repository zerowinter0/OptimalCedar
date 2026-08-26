#!/usr/bin/env bash
# Formal GPU-aware DP evaluation: one GPU workload plus two Cedar workloads.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNNER="${REPO_ROOT}/evaluation/chapter6_experiments/run_formal_plan_and_matrix.sh"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/chapter6_experiments/gpu_aware_formal_matrix}"

export MATRIX_OUTPUT_ROOT="${OUTPUT_ROOT}"
export OPTIMIZER_SET="formal_seven"
export REPEATS="3"
export TASK_TIMEOUT_SEC="3600"
export CPU_BUDGET="64"
export LOCAL_WORKERS="8"
export GENERAL_VIDEO_REFINE_SAMPLES="5000"
export RESUME_EXISTING="${RESUME_EXISTING:-0}"

# GenerateVideo uses the latest operator-aware offline profile. The workload
# is run separately because the two Cedar workloads use the archived formal
# profile directory below.
PROFILE_DIR="${REPO_ROOT}/outputs/chapter6_experiments/operator_scaling_two_stage_study/profiles" \
  bash "${RUNNER}" --workloads general_video_refine

PROFILE_DIR="${REPO_ROOT}/evaluation/chapter6_experiments/formal_results/paper_artifacts/optimizer/profiles" \
  bash "${RUNNER}" --workloads commonvoice,simclrv2_cache

printf 'GPU-aware formal matrix complete: %s\n' "${OUTPUT_ROOT}"
