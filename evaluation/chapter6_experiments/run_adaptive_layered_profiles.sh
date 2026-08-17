#!/usr/bin/env bash
# Formal bounded-time profile for the three cost-model study workloads.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULT_ROOT="${REPO_ROOT}/outputs/chapter6_experiments/adaptive_layered_profiles"
PROFILE_ROOT="${RESULT_ROOT}/profiles"
LOG_FILE="${RESULT_ROOT}/adaptive_layered_profiles.log"
WORKLOADS="alpaca_cot,stackexchange,general_video_refine"
MIN_AVAILABLE_GIB="${CEDAR_PROFILE_MIN_AVAILABLE_GIB:-64}"

cd "${REPO_ROOT}"
source env/bin/activate
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
mkdir -p "${RESULT_ROOT}" "${PROFILE_ROOT}"

cat > "${RESULT_ROOT}/PROTOCOL.md" <<'EOF'
# Adaptive layered profiling protocol

This profile separates measurements that the old whole-pipeline offload pass
mixed together:

1. A single 10-second in-process pilot measures baseline throughput,
   selectivity, record/data scaling, and captures immutable legal outputs at
   every operator boundary.
2. Every Ray/SMP operator replays the same predecessor-output pool. Worker-side
   compute is measured for at least 3 seconds and 30 observations, stopping when
   RSE is at most 10%; only unstable stages continue, up to 30 seconds.
3. Serialization/stage-boundary cost is calibrated independently. The two
   highest-cost stages per backend are additionally measured at width 8 to
   calibrate parallel scaling without multiplying the entire profile matrix.
4. Contention and fusion are intentionally not charged to isolated operators;
   they are deferred to short end-to-end validation of optimizer-selected plans.

All workloads use one baseline worker, identical fixed replay inputs for Ray and
SMP, and the formal W=8/CPU_BUDGET=64 resource target. The legacy offload schema
is retained through documented component-substitution throughput so existing
optimizers can parse the same profile.
EOF

exec >>"${LOG_FILE}" 2>&1
echo "[$(date -Is)] Waiting for at least ${MIN_AVAILABLE_GIB} GiB MemAvailable"
while true; do
  available_kib="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
  if (( available_kib >= MIN_AVAILABLE_GIB * 1024 * 1024 )); then
    break
  fi
  echo "[$(date -Is)] WAIT memory_available_gib=$((available_kib / 1024 / 1024))"
  sleep 300
done

echo "[$(date -Is)] Starting adaptive layered profiles"
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
export CH6_RESULT_ROOT="${RESULT_ROOT}"
export CH6_PROFILE_ROOT="${PROFILE_ROOT}"
export CH6_PROFILE_RUN_ID="adaptive_layered"
export CH6_PROFILE_TIMEOUT_SEC=10800
export INCREMENTAL_BACKEND_COMPUTE=0

bash evaluation/chapter6_experiments/run_formal_profiles.sh \
  --workloads "${WORKLOADS}"

python evaluation/chapter6_experiments/validate_adaptive_layered_profiles.py \
  --profile-root "${PROFILE_ROOT}" \
  --workloads "${WORKLOADS}" \
  --output "${RESULT_ROOT}/validation_summary.json"
touch "${RESULT_ROOT}/COMPLETE"
echo "[$(date -Is)] Adaptive layered profiles completed"
