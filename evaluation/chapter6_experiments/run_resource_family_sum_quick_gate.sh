#!/usr/bin/env bash
# One-round screening gate for the max(L, R, S, G) DP objective.
#
# This intentionally reuses the newest compatible stationary/adaptive profile
# for every workload: the objective changes plan scoring, not measurements.
# Execution data sizes are reduced to obtain a directional result quickly.
# Cedar and the exhaustive policy-two-stage baselines are excluded because
# their search time, rather than execution time, dominates this screening
# experiment on complex workloads.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/chapter6_experiments/resource_family_sum_quick_gate_5x}"
PROFILE_DIR="${OUTPUT_ROOT}/profiles"
FORMAL_ROOT="${OUTPUT_ROOT}/formal_runs"

cd "${REPO_ROOT}"
source env/bin/activate

mkdir -p "${PROFILE_DIR}" "${FORMAL_ROOT}"
printf '%s\n' "$$" > "${OUTPUT_ROOT}/runner.pid"

copy_profile() {
  local workload="$1" source="$2"
  if [[ ! -f "${source}" ]]; then
    echo "Missing source profile for ${workload}: ${source}" >&2
    exit 1
  fi
  cp -p "${source}" "${PROFILE_DIR}/${workload}.yaml"
  printf '%s\t%s\t%s\n' \
    "${workload}" "${source#${REPO_ROOT}/}" "$(sha256sum "${source}" | cut -d' ' -f1)"
}

{
  printf 'workload\tsource_profile\tsha256\n'
  copy_profile general_video_refine \
    "${REPO_ROOT}/outputs/chapter6_experiments/joint_actor_budget_formal_all_ten_v1/profiles/general_video_refine.yaml"
  copy_profile pile_hackernews \
    "${REPO_ROOT}/outputs/chapter6_experiments/stationary_formal_runtime_floor_30m_v1/workloads/pile_hackernews/profiles/pile_hackernews.yaml"
  copy_profile pile_europarl \
    "${REPO_ROOT}/outputs/chapter6_experiments/stationary_formal_runtime_floor_30m_v1/workloads/pile_europarl/profiles/pile_europarl.yaml"
  copy_profile stackexchange \
    "${REPO_ROOT}/outputs/chapter6_experiments/stationary_formal_sequential_remaining_nine_v2/workloads/stackexchange/profiles/stackexchange.yaml"
  copy_profile pile_pubmed_abstracts \
    "${REPO_ROOT}/outputs/chapter6_experiments/joint_actor_budget_formal_all_ten_v1/profiles/pile_pubmed_abstracts.yaml"
  copy_profile pile_uspto_backgrounds \
    "${REPO_ROOT}/outputs/chapter6_experiments/joint_actor_budget_formal_all_ten_v1/profiles/pile_uspto_backgrounds.yaml"
  copy_profile alpaca_cot \
    "${REPO_ROOT}/outputs/chapter6_experiments/stationary_formal_runtime_floor_30m_v1/workloads/alpaca_cot/profiles/alpaca_cot.yaml"
  copy_profile simclrv2_cache \
    "${REPO_ROOT}/outputs/chapter6_experiments/stationary_formal_runtime_floor_30m_v1/workloads/simclrv2_cache/profiles/simclrv2_cache.yaml"
  copy_profile commonvoice \
    "${REPO_ROOT}/outputs/chapter6_experiments/commonvoice_stationary_profile_v2/profiles/commonvoice.yaml"
  copy_profile redpajama_code \
    "${REPO_ROOT}/outputs/chapter6_experiments/joint_actor_budget_formal_all_ten_v1/profiles/redpajama_code.yaml"
} > "${OUTPUT_ROOT}/PROFILE_PROVENANCE.tsv"

export MATRIX_OUTPUT_ROOT="${FORMAL_ROOT}"
export PROFILE_DIR
export OPTIMIZER_SET=quick_model_gate
export REPEATS=1
export TASK_TIMEOUT_SEC="${TASK_TIMEOUT_SEC:-900}"
export RESUME_EXISTING="${RESUME_EXISTING:-0}"

# Screening-only sizes. SimCLRv2-cache remains at its bounded 9,469 records in
# the shared matrix runner because that complete source is already small.
export GENERAL_VIDEO_REFINE_SAMPLES="${GENERAL_VIDEO_REFINE_SAMPLES:-50}"
export COMMONVOICE_SAMPLES="${COMMONVOICE_SAMPLES:-10000}"
export ALPACA_COT_SAMPLES="${ALPACA_COT_SAMPLES:-10000}"
export STACKEXCHANGE_SAMPLES="${STACKEXCHANGE_SAMPLES:-10000}"
export PILE_EUROPARL_SAMPLES="${PILE_EUROPARL_SAMPLES:-10000}"
export PILE_HACKERNEWS_SAMPLES="${PILE_HACKERNEWS_SAMPLES:-10000}"
export PILE_PUBMED_SAMPLES="${PILE_PUBMED_SAMPLES:-10000}"
export PILE_USPTO_SAMPLES="${PILE_USPTO_SAMPLES:-10000}"
export REDPAJAMA_CODE_SAMPLES="${REDPAJAMA_CODE_SAMPLES:-10000}"

bash evaluation/chapter6_experiments/run_formal_plan_and_matrix.sh \
  --workloads pile_hackernews,pile_europarl,stackexchange,pile_pubmed_abstracts,pile_uspto_backgrounds,alpaca_cot,simclrv2_cache,commonvoice,redpajama_code

# Keep the video workload last so its comparatively expensive decoding and
# GPU inference cannot delay directional results for the other workloads.
bash evaluation/chapter6_experiments/run_formal_plan_and_matrix.sh \
  --workloads general_video_refine

touch "${OUTPUT_ROOT}/COMPLETE"
