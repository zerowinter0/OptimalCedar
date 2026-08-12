#!/usr/bin/env bash
# Screen the official video self-evolution recipe and formalize only a 1.20x win.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_ROOT="${DIVERSE_ROOT:-${REPO_ROOT}/outputs/chapter6_experiments/datajuicer_diverse_workloads}"
PROFILE="${BASE_ROOT}/profiles/video_self_evolution.yaml"
SCREEN_ROOT="${BASE_ROOT}/video_self_evolution_screening"
FORMAL_ROOT="${BASE_ROOT}/video_self_evolution_formal"
STATUS_ROOT="${BASE_ROOT}/video_self_evolution_status"
GENERAL_FORMAL_ROOT="${BASE_ROOT}/general_video_refine_formal_7500"
GENERAL_FORMAL_SAMPLES=7500

cd "${REPO_ROOT}"
source env/bin/activate
mkdir -p "${STATUS_ROOT}"

valid_profile() {
  local profile_path="$1"
  python - "${profile_path}" <<'PY'
import sys
import yaml

with open(sys.argv[1], encoding="utf-8") as stream:
    profile = yaml.safe_load(stream)
expected = {
    "schema_version": 1,
    "profile_scope": "single_local_worker",
    "profile_local_workers": 1,
    "actors_per_stage": 1,
    "ray_actors_per_stage": 1,
    "smp_procs_per_stage": 1,
}
assert profile.get("resource_config") == expected
assert {"RAY", "SMP"} <= set(profile.get("offloads", {}))
assert "selectivities" in profile.get("baseline", {})
PY
}

run_general_video_fallback() {
  local general_profile="${BASE_ROOT}/profiles/general_video_refine.yaml"
  local resume=0
  echo "[$(date -Is)] FORMALIZE general-video fallback at ${GENERAL_FORMAL_SAMPLES} outputs"
  if [[ ! -f "${general_profile}" ]] || ! valid_profile "${general_profile}"; then
    CH6_RESULT_ROOT="${BASE_ROOT}" \
    CH6_PROFILE_ROOT="${BASE_ROOT}/profiles" \
    CH6_PROFILE_TIMEOUT_SEC=3600 \
    CH6_PROFILE_RUN_ID=general_video_refine_fallback_profile \
    bash evaluation/chapter6_experiments/run_formal_profiles.sh \
      --workloads general_video_refine
    valid_profile "${general_profile}"
  fi
  if [[ -f "${GENERAL_FORMAL_ROOT}/general_video_refine/nohup.log" ]] &&
     grep -Fq "COMPLETE general_video_refine" \
       "${GENERAL_FORMAL_ROOT}/general_video_refine/nohup.log"; then
    echo "[$(date -Is)] REUSE completed general-video fallback matrix"
    return
  fi
  [[ -e "${GENERAL_FORMAL_ROOT}" ]] && resume=1
  MATRIX_OUTPUT_ROOT="${GENERAL_FORMAL_ROOT}" \
  PROFILE_DIR="${BASE_ROOT}/profiles" \
  REPEATS=3 \
  OPTIMIZER_SET=complete \
  OPTIMIZER_PLAN_TIMEOUT_SEC=3600 \
  EXECUTION_TIMEOUT_SEC=3600 \
  GENERAL_VIDEO_REFINE_SAMPLES="${GENERAL_FORMAL_SAMPLES}" \
  RESUME_EXISTING="${resume}" \
  bash evaluation/chapter6_experiments/run_formal_plan_and_matrix.sh \
    --workloads general_video_refine
}

formal_video_self_is_win() {
  python - "${FORMAL_ROOT}/video_self_evolution/results" <<'PY'
import sys
from pathlib import Path

from evaluation.chapter6_experiments.analyze_datajuicer_diverse_workloads import (
    OPTIMIZERS,
    _summarize_matrix,
)

root = Path(sys.argv[1])
summary = _summarize_matrix(
    "video_self_evolution",
    5000,
    {optimizer: [root] for optimizer in OPTIMIZERS},
    str(root.parent),
)
raise SystemExit(
    0
    if summary["valid"] and summary["dp_at_least_20pct_faster"]
    else 1
)
PY
}

if [[ -f "${PROFILE}" ]] && valid_profile "${PROFILE}"; then
  echo "[$(date -Is)] REUSE video-self-evolution profile"
else
  set +e
  CH6_RESULT_ROOT="${BASE_ROOT}" \
  CH6_PROFILE_ROOT="${BASE_ROOT}/profiles" \
  CH6_PROFILE_TIMEOUT_SEC=3600 \
  CH6_PROFILE_RUN_ID=video_self_evolution_profile \
  bash evaluation/chapter6_experiments/run_formal_profiles.sh \
    --workloads video_self_evolution
  profile_status=$?
  set -e
  if [[ "${profile_status}" -ne 0 ]] || ! valid_profile "${PROFILE}"; then
    printf '{\n  "status": "profile_unavailable",\n  "profile_timeout_sec": 3600,\n  "selected": false\n}\n' \
      > "${STATUS_ROOT}/decision.json"
    echo "[$(date -Is)] video-self-evolution profile unavailable; retain evidence and continue"
    run_general_video_fallback
    exit 0
  fi
fi

if [[ -f "${SCREEN_ROOT}/video_self_evolution/nohup.log" ]] &&
   grep -Fq "COMPLETE video_self_evolution" \
     "${SCREEN_ROOT}/video_self_evolution/nohup.log"; then
  echo "[$(date -Is)] REUSE video-self-evolution screen"
elif [[ -e "${SCREEN_ROOT}" ]]; then
  echo "[$(date -Is)] RESUME incomplete video-self-evolution screen"
  set +e
  MATRIX_OUTPUT_ROOT="${SCREEN_ROOT}" \
  PROFILE_DIR="${BASE_ROOT}/profiles" \
  REPEATS=1 \
  OPTIMIZER_SET=complete \
  OPTIMIZER_PLAN_TIMEOUT_SEC=3600 \
  EXECUTION_TIMEOUT_SEC=3600 \
  VIDEO_SELF_EVOLUTION_SAMPLES=2000 \
  RESUME_EXISTING=1 \
  bash evaluation/chapter6_experiments/run_formal_plan_and_matrix.sh \
    --workloads video_self_evolution
  screen_status=$?
  set -e
  if [[ "${screen_status}" -ne 0 ]]; then
    printf '{\n  "status": "screening_infeasible_or_failed",\n  "requested_outputs": 2000,\n  "selected": false\n}\n' \
      > "${STATUS_ROOT}/decision.json"
    echo "[$(date -Is)] video-self-evolution resumed screen infeasible; retain evidence and continue"
    run_general_video_fallback
    exit 0
  fi
else
  set +e
  MATRIX_OUTPUT_ROOT="${SCREEN_ROOT}" \
  PROFILE_DIR="${BASE_ROOT}/profiles" \
  REPEATS=1 \
  OPTIMIZER_SET=complete \
  OPTIMIZER_PLAN_TIMEOUT_SEC=3600 \
  EXECUTION_TIMEOUT_SEC=3600 \
  VIDEO_SELF_EVOLUTION_SAMPLES=2000 \
  bash evaluation/chapter6_experiments/run_formal_plan_and_matrix.sh \
    --workloads video_self_evolution
  screen_status=$?
  set -e
  if [[ "${screen_status}" -ne 0 ]]; then
    printf '{\n  "status": "screening_infeasible_or_failed",\n  "requested_outputs": 2000,\n  "selected": false\n}\n' \
      > "${STATUS_ROOT}/decision.json"
    echo "[$(date -Is)] video-self-evolution screen infeasible; retain evidence and continue"
    run_general_video_fallback
    exit 0
  fi
fi

if python - "${SCREEN_ROOT}/video_self_evolution/results" <<'PY'
import json
import sys
from pathlib import Path

times = {}
for path in Path(sys.argv[1]).glob("round1__*.json"):
    optimizer = path.stem.split("__", 1)[1]
    with path.open(encoding="utf-8") as stream:
        result = json.load(stream)
    if result.get("epoch_num_samples") == [2000]:
        values = result.get("epoch_run_times", [])
        if len(values) == 1:
            times[optimizer] = float(values[0])
dp = times.get("dp_optimizer")
others = [value for name, value in times.items() if name != "dp_optimizer"]
raise SystemExit(0 if dp and others and min(others) / dp >= 1.20 else 1)
PY
then
  if [[ -f "${FORMAL_ROOT}/video_self_evolution/nohup.log" ]] &&
     grep -Fq "COMPLETE video_self_evolution" \
       "${FORMAL_ROOT}/video_self_evolution/nohup.log"; then
    echo "[$(date -Is)] REUSE video-self-evolution formal matrix"
  elif [[ -e "${FORMAL_ROOT}" ]]; then
    echo "[$(date -Is)] RESUME incomplete video-self-evolution formal matrix"
    set +e
    MATRIX_OUTPUT_ROOT="${FORMAL_ROOT}" \
    PROFILE_DIR="${BASE_ROOT}/profiles" \
    REPEATS=3 \
    OPTIMIZER_SET=complete \
    OPTIMIZER_PLAN_TIMEOUT_SEC=3600 \
    EXECUTION_TIMEOUT_SEC=3600 \
    VIDEO_SELF_EVOLUTION_SAMPLES=5000 \
    RESUME_EXISTING=1 \
    bash evaluation/chapter6_experiments/run_formal_plan_and_matrix.sh \
      --workloads video_self_evolution
    formal_status=$?
    set -e
  else
    set +e
    MATRIX_OUTPUT_ROOT="${FORMAL_ROOT}" \
    PROFILE_DIR="${BASE_ROOT}/profiles" \
    REPEATS=3 \
    OPTIMIZER_SET=complete \
    OPTIMIZER_PLAN_TIMEOUT_SEC=3600 \
    EXECUTION_TIMEOUT_SEC=3600 \
    VIDEO_SELF_EVOLUTION_SAMPLES=5000 \
    bash evaluation/chapter6_experiments/run_formal_plan_and_matrix.sh \
      --workloads video_self_evolution
    formal_status=$?
    set -e
  fi
  formal_status="${formal_status:-0}"
  if [[ "${formal_status}" -eq 0 ]] && formal_video_self_is_win; then
    printf '{\n  "status": "formalized_screening_win",\n  "screening_threshold": 1.2,\n  "formal_outputs": 5000,\n  "selected_if_formal_gate_passes": true\n}\n' \
      > "${STATUS_ROOT}/decision.json"
  else
    printf '{\n  "status": "formal_matrix_failed_or_below_1.20x",\n  "screening_threshold": 1.2,\n  "selected": false\n}\n' \
      > "${STATUS_ROOT}/decision.json"
    run_general_video_fallback
  fi
else
  printf '{\n  "status": "screening_below_1.20x",\n  "screening_threshold": 1.2,\n  "selected": false\n}\n' \
    > "${STATUS_ROOT}/decision.json"
  run_general_video_fallback
fi
