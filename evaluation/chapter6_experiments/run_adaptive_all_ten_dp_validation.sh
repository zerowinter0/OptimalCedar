#!/usr/bin/env bash
# Build fresh adaptive layered profiles for all ten current workloads, then
# regenerate and execute DP plans for three formal rounds with those profiles.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULT_ROOT="${RESULT_ROOT:-${REPO_ROOT}/outputs/chapter6_experiments/adaptive_all_ten_dp_validation}"
PROFILE_ROOT="${RESULT_ROOT}/profiles"
MATRIX_ROOT="${RESULT_ROOT}/formal_runs"
PROFILE_RUN_ID="adaptive_all_ten"
WORKLOADS="general_video_refine,pile_hackernews,pile_europarl,stackexchange,pile_pubmed_abstracts,pile_uspto_backgrounds,alpaca_cot,simclrv2_cache,commonvoice,redpajama_code"
MIN_AVAILABLE_GIB="${CEDAR_PROFILE_MIN_AVAILABLE_GIB:-64}"

cd "${REPO_ROOT}"
source env/bin/activate
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
mkdir -p "${RESULT_ROOT}" "${PROFILE_ROOT}" "${MATRIX_ROOT}"

cat > "${RESULT_ROOT}/PROTOCOL.md" <<'EOF'
# Adaptive profiles and DP formal validation

- Workloads: GeneralVideoRefine, HackerNews, EuroParl, StackExchange, PubMed,
  USPTO, Alpaca-CoT, SimCLR-cache, CommonVoice, and RP-Code.
- Every workload receives a new adaptive layered profile; no numeric profile
  is copied from an earlier experiment.
- One 10-second in-process pilot captures baseline statistics and fixed legal
  inputs at every operator boundary.
- Every legal Ray/SMP operator replays the same input pool at width 1 for at
  least 3 seconds and 30 observations. It stops at RSE <= 10%, or at 30 seconds.
- The two highest-cost operators per backend receive an additional width-8
  calibration of at most 10 seconds. Ray/SMP boundary costs are independently
  calibrated.
- Profile resources are one local worker and one actor/process per stage.
- Formal execution uses W=8, CPU_BUDGET=64, the revised DP optimizer only,
  three rounds, and the workload's standard cache setting. One unified
  optimization-plus-first-execution deadline of 3600 seconds applies; a first
  round timeout skips later rounds.
EOF

available_kib="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
if (( available_kib < MIN_AVAILABLE_GIB * 1024 * 1024 )); then
  printf '[%s] Insufficient memory: available_gib=%s required_gib=%s\n' \
    "$(date -Is)" "$((available_kib / 1024 / 1024))" "${MIN_AVAILABLE_GIB}"
  exit 1
fi

printf '[%s] PHASE adaptive profiles\n' "$(date -Is)"
export CEDAR_LAYERED_ADAPTIVE_PROFILE=1
export CEDAR_PROFILE_TIME_SEC=10
export CEDAR_ADAPTIVE_PROFILE_MIN_SEC=3
export CEDAR_ADAPTIVE_PROFILE_MAX_SEC=30
export CEDAR_ADAPTIVE_PROFILE_TARGET_RSE=0.10
export CEDAR_ADAPTIVE_PROFILE_MIN_OBS=30
export CEDAR_PROFILE_POOL_SAMPLES=64
export CEDAR_PROFILE_POOL_BYTES_PER_PIPE=$((64 * 1024 * 1024))
export CEDAR_PROFILE_POOL_BYTES_TOTAL=$((512 * 1024 * 1024))
export CEDAR_PROFILE_SCALING_TOP_K=2
export CEDAR_PROFILE_SCALING_WIDTH=8
export CEDAR_PROFILE_BOUNDARY_MODEL=1
export CEDAR_PROFILE_INFER_COMPUTE_SCALING=1
export CEDAR_PROFILE_RAY_ACTORS=1
export CEDAR_PROFILE_SMP_PROCS=1
export CH6_RESULT_ROOT="${RESULT_ROOT}"
export CH6_PROFILE_ROOT="${PROFILE_ROOT}"
export CH6_PROFILE_RUN_ID="${PROFILE_RUN_ID}"
export CH6_PROFILE_TIMEOUT_SEC=10800
export INCREMENTAL_BACKEND_COMPUTE=0

if [[ ! -f "${RESULT_ROOT}/PROFILES_COMPLETE" ]]; then
  bash evaluation/chapter6_experiments/run_formal_profiles.sh \
    --workloads "${WORKLOADS}"
  python evaluation/chapter6_experiments/validate_adaptive_layered_profiles.py \
    --profile-root "${PROFILE_ROOT}" \
    --workloads "${WORKLOADS}" \
    --output "${RESULT_ROOT}/profile_validation.json"
  printf 'complete\n' > "${RESULT_ROOT}/PROFILES_COMPLETE"
else
  printf '[%s] REUSE validated adaptive profiles\n' "$(date -Is)"
fi

printf '[%s] PHASE DP formal matrix\n' "$(date -Is)"
unset CEDAR_LAYERED_ADAPTIVE_PROFILE
PROFILE_DIR="${PROFILE_ROOT}" \
MATRIX_OUTPUT_ROOT="${MATRIX_ROOT}" \
OPTIMIZER_SET=dp_only \
PLAN_ONLY=0 \
REPEATS=3 \
RESUME_EXISTING=1 \
TASK_TIMEOUT_SEC=3600 \
  bash evaluation/chapter6_experiments/run_formal_plan_and_matrix.sh \
    --workloads "${WORKLOADS}"

python - "${MATRIX_ROOT}" "${RESULT_ROOT}/execution_summary.json" \
  "${WORKLOADS}" <<'PY'
import json
import sys
from pathlib import Path

matrix_root = Path(sys.argv[1])
output = Path(sys.argv[2])
workloads = sys.argv[3].split(",")
summary = {}
for workload in workloads:
    results = matrix_root / workload / "results"
    summary[workload] = {
        "successful_rounds": len(list(results.glob("round*__dp_optimizer.json"))),
        "timeouts": len(list(results.glob("round*__dp_optimizer.timeout.json"))),
        "skipped": len(list(results.glob("round*__dp_optimizer.skipped.json"))),
    }
output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
PY
printf 'complete\n' > "${RESULT_ROOT}/COMPLETE"
printf '[%s] COMPLETE adaptive all-ten DP validation\n' "$(date -Is)"
