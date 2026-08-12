#!/usr/bin/env bash
# Screen the two remaining Data-Juicer candidates without overlapping runs.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_ROOT="${DIVERSE_ROOT:-${REPO_ROOT}/outputs/chapter6_experiments/datajuicer_diverse_workloads}"
CODE_ROOT="${BASE_ROOT}/code_screening"
CODE_RUN_ROOT="${CODE_ROOT}/datajuicer_candidate_runs/code_additive_screen"
CODE_FORMAL_ROOT="${BASE_ROOT}/code_formal"
CODE_FORMAL_RUN_ROOT="${CODE_FORMAL_ROOT}/datajuicer_candidate_runs/code_additive_formal"
ARXIV_ROOT="${BASE_ROOT}/arxiv_scaled_screening"
ARXIV_FORMAL_ROOT="${BASE_ROOT}/arxiv_formal"
ALPACA_ROOT="${BASE_ROOT}/alpaca_formal"

cd "${REPO_ROOT}"
source env/bin/activate

complete_candidate_report() {
  local report="$1" repeats="$2"
  [[ -f "${report}" ]] || return 1
  python - "${report}" "${repeats}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    report = json.load(stream)
expected = {
    "optimizer",
    "dj_optimizer",
    "dp_cedar_optimizer",
    "dp_optimizer",
    "dp_two_stage_optimizer",
    "pecan_optimizer",
}
protocol = report.get("protocol", {})
assert set(protocol.get("candidate_optimizer_set", [])) == expected
assert protocol.get("candidate_repeats") == int(sys.argv[2])
PY
}

mkdir -p "${CODE_ROOT}"
if complete_candidate_report "${CODE_RUN_ROOT}/dp_20pct_report.json" 1; then
  echo "[$(date -Is)] REUSE completed RP-Code screening"
elif [[ -e "${CODE_RUN_ROOT}" ]]; then
  echo "[$(date -Is)] RESUME incomplete RP-Code screening"
  DJ_CANDIDATE_OUTPUT_ROOT="${CODE_ROOT}" \
  DJ_CANDIDATE_PROFILE_DIR="${REPO_ROOT}/evaluation/chapter6_experiments/formal_results/paper_artifacts/optimizer/profiles" \
  DJ_CANDIDATE_RUN_ID=code_additive_screen \
  DJ_CANDIDATE_WORKLOADS=redpajama_code \
  DJ_CANDIDATE_REPEATS=1 \
  DJ_CANDIDATE_OUTPUTS=20000 \
  DJ_OPTIMIZER_TIMEOUT_SEC=3600 \
  DJ_EXECUTION_TIMEOUT_SEC=3600 \
  DJ_PROFILE_TIMEOUT_SEC=3600 \
  DJ_REUSE_EXISTING_PROFILES=1 \
  DJ_CANDIDATE_RESUME=1 \
  bash evaluation/chapter6_experiments/run_datajuicer_candidate_matrix.sh
else
  DJ_CANDIDATE_OUTPUT_ROOT="${CODE_ROOT}" \
  DJ_CANDIDATE_PROFILE_DIR="${REPO_ROOT}/evaluation/chapter6_experiments/formal_results/paper_artifacts/optimizer/profiles" \
  DJ_CANDIDATE_RUN_ID=code_additive_screen \
  DJ_CANDIDATE_WORKLOADS=redpajama_code \
  DJ_CANDIDATE_REPEATS=1 \
  DJ_CANDIDATE_OUTPUTS=20000 \
  DJ_OPTIMIZER_TIMEOUT_SEC=3600 \
  DJ_EXECUTION_TIMEOUT_SEC=3600 \
  DJ_PROFILE_TIMEOUT_SEC=3600 \
  DJ_REUSE_EXISTING_PROFILES=1 \
  bash evaluation/chapter6_experiments/run_datajuicer_candidate_matrix.sh
fi

if python - "${CODE_RUN_ROOT}/dp_20pct_report.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    report = json.load(stream)
raise SystemExit(
    0
    if report["candidates"]["redpajama_code"][
        "dp_at_least_20pct_faster"
    ]
    else 1
)
PY
then
  if complete_candidate_report \
       "${CODE_FORMAL_RUN_ROOT}/dp_20pct_report.json" 3; then
    echo "[$(date -Is)] REUSE completed RP-Code formal matrix"
  elif [[ -e "${CODE_FORMAL_RUN_ROOT}" ]]; then
    echo "[$(date -Is)] RESUME incomplete RP-Code formal matrix"
    DJ_CANDIDATE_OUTPUT_ROOT="${CODE_FORMAL_ROOT}" \
    DJ_CANDIDATE_PROFILE_DIR="${REPO_ROOT}/evaluation/chapter6_experiments/formal_results/paper_artifacts/optimizer/profiles" \
    DJ_CANDIDATE_RUN_ID=code_additive_formal \
    DJ_CANDIDATE_WORKLOADS=redpajama_code \
    DJ_CANDIDATE_REPEATS=3 \
    DJ_CANDIDATE_OUTPUTS=20000 \
    DJ_OPTIMIZER_TIMEOUT_SEC=3600 \
    DJ_EXECUTION_TIMEOUT_SEC=3600 \
    DJ_PROFILE_TIMEOUT_SEC=3600 \
    DJ_REUSE_EXISTING_PROFILES=1 \
    DJ_CANDIDATE_RESUME=1 \
    bash evaluation/chapter6_experiments/run_datajuicer_candidate_matrix.sh
  else
    DJ_CANDIDATE_OUTPUT_ROOT="${CODE_FORMAL_ROOT}" \
    DJ_CANDIDATE_PROFILE_DIR="${REPO_ROOT}/evaluation/chapter6_experiments/formal_results/paper_artifacts/optimizer/profiles" \
    DJ_CANDIDATE_RUN_ID=code_additive_formal \
    DJ_CANDIDATE_WORKLOADS=redpajama_code \
    DJ_CANDIDATE_REPEATS=3 \
    DJ_CANDIDATE_OUTPUTS=20000 \
    DJ_OPTIMIZER_TIMEOUT_SEC=3600 \
    DJ_EXECUTION_TIMEOUT_SEC=3600 \
    DJ_PROFILE_TIMEOUT_SEC=3600 \
    DJ_REUSE_EXISTING_PROFILES=1 \
    bash evaluation/chapter6_experiments/run_datajuicer_candidate_matrix.sh
  fi
else
  echo "[$(date -Is)] RP-Code is below 1.20x; one-run evidence retained"
fi

# The 20,000-output ArXiv attempt established that this scale exceeds the
# one-hour execution limit (3,367 outputs for DJ in 3,600 s).  A 2,500-output
# run still processes roughly 140 MiB of long scientific text while leaving
# enough headroom for every viable plan to finish within the formal threshold.
if [[ -f "${ARXIV_ROOT}/redpajama_arxiv/results/round1__dp_optimizer.json" &&
      -f "${ARXIV_ROOT}/redpajama_arxiv/nohup.log" ]] &&
   grep -Fq "COMPLETE redpajama_arxiv" \
     "${ARXIV_ROOT}/redpajama_arxiv/nohup.log"; then
  echo "[$(date -Is)] REUSE completed scaled RP-ArXiv screening"
elif [[ -e "${ARXIV_ROOT}" ]]; then
  echo "[$(date -Is)] RESUME incomplete scaled RP-ArXiv screening"
  MATRIX_OUTPUT_ROOT="${ARXIV_ROOT}" \
  PROFILE_DIR="${BASE_ROOT}/profiles" \
  REPEATS=1 \
  OPTIMIZER_SET=complete \
  OPTIMIZER_PLAN_TIMEOUT_SEC=3600 \
  EXECUTION_TIMEOUT_SEC=3600 \
  REDPAJAMA_ARXIV_SAMPLES=2500 \
  RESUME_EXISTING=1 \
  bash evaluation/chapter6_experiments/run_formal_plan_and_matrix.sh \
    --workloads redpajama_arxiv
else
  MATRIX_OUTPUT_ROOT="${ARXIV_ROOT}" \
  PROFILE_DIR="${BASE_ROOT}/profiles" \
  REPEATS=1 \
  OPTIMIZER_SET=complete \
  OPTIMIZER_PLAN_TIMEOUT_SEC=3600 \
  EXECUTION_TIMEOUT_SEC=3600 \
  REDPAJAMA_ARXIV_SAMPLES=2500 \
  bash evaluation/chapter6_experiments/run_formal_plan_and_matrix.sh \
    --workloads redpajama_arxiv
fi

if python - "${ARXIV_ROOT}/redpajama_arxiv/results" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
times = {}
for path in root.glob("round1__*.json"):
    optimizer = path.stem.split("__", 1)[1]
    with path.open(encoding="utf-8") as stream:
        result = json.load(stream)
    values = result.get("epoch_run_times", [])
    samples = result.get("epoch_num_samples", [])
    if len(values) == 1 and samples == [2500]:
        times[optimizer] = float(values[0])
dp = times.get("dp_optimizer")
competitors = [value for name, value in times.items() if name != "dp_optimizer"]
passes = dp is not None and competitors and min(competitors) / dp >= 1.20
raise SystemExit(0 if passes else 1)
PY
then
  if [[ -f "${ARXIV_FORMAL_ROOT}/redpajama_arxiv/results/round3__dp_optimizer.json" &&
        -f "${ARXIV_FORMAL_ROOT}/redpajama_arxiv/nohup.log" ]] &&
     grep -Fq "COMPLETE redpajama_arxiv" \
       "${ARXIV_FORMAL_ROOT}/redpajama_arxiv/nohup.log"; then
    echo "[$(date -Is)] REUSE completed RP-ArXiv formal matrix"
  elif [[ -e "${ARXIV_FORMAL_ROOT}" ]]; then
    echo "[$(date -Is)] RESUME incomplete RP-ArXiv formal matrix"
    MATRIX_OUTPUT_ROOT="${ARXIV_FORMAL_ROOT}" \
    PROFILE_DIR="${BASE_ROOT}/profiles" \
    REPEATS=3 \
    OPTIMIZER_SET=complete \
    OPTIMIZER_PLAN_TIMEOUT_SEC=3600 \
    EXECUTION_TIMEOUT_SEC=3600 \
    REDPAJAMA_ARXIV_SAMPLES=2500 \
    RESUME_EXISTING=1 \
    bash evaluation/chapter6_experiments/run_formal_plan_and_matrix.sh \
      --workloads redpajama_arxiv
  else
    MATRIX_OUTPUT_ROOT="${ARXIV_FORMAL_ROOT}" \
    PROFILE_DIR="${BASE_ROOT}/profiles" \
    REPEATS=3 \
    OPTIMIZER_SET=complete \
    OPTIMIZER_PLAN_TIMEOUT_SEC=3600 \
    EXECUTION_TIMEOUT_SEC=3600 \
    REDPAJAMA_ARXIV_SAMPLES=2500 \
    bash evaluation/chapter6_experiments/run_formal_plan_and_matrix.sh \
      --workloads redpajama_arxiv
  fi
else
  echo "[$(date -Is)] RP-ArXiv is below 1.20x; one-run evidence retained"
fi

# Alpaca-CoT is retained as the short instruction-tuning representative even
# when its screening result is neutral; formalize it with the required three
# round-robin repetitions rather than relying on the one-run screen.
if [[ -f "${ALPACA_ROOT}/alpaca_cot/results/round3__dp_optimizer.json" &&
      -f "${ALPACA_ROOT}/alpaca_cot/nohup.log" ]] &&
   grep -Fq "COMPLETE alpaca_cot" "${ALPACA_ROOT}/alpaca_cot/nohup.log"; then
  echo "[$(date -Is)] REUSE completed Alpaca-CoT formal matrix"
elif [[ -e "${ALPACA_ROOT}" ]]; then
  echo "[$(date -Is)] RESUME incomplete Alpaca-CoT formal matrix"
  MATRIX_OUTPUT_ROOT="${ALPACA_ROOT}" \
  PROFILE_DIR="${BASE_ROOT}/profiles" \
  REPEATS=3 \
  OPTIMIZER_SET=complete \
  OPTIMIZER_PLAN_TIMEOUT_SEC=3600 \
  EXECUTION_TIMEOUT_SEC=3600 \
  ALPACA_COT_SAMPLES=20000 \
  RESUME_EXISTING=1 \
  bash evaluation/chapter6_experiments/run_formal_plan_and_matrix.sh \
    --workloads alpaca_cot
else
  MATRIX_OUTPUT_ROOT="${ALPACA_ROOT}" \
  PROFILE_DIR="${BASE_ROOT}/profiles" \
  REPEATS=3 \
  OPTIMIZER_SET=complete \
  OPTIMIZER_PLAN_TIMEOUT_SEC=3600 \
  EXECUTION_TIMEOUT_SEC=3600 \
  ALPACA_COT_SAMPLES=20000 \
  bash evaluation/chapter6_experiments/run_formal_plan_and_matrix.sh \
    --workloads alpaca_cot
fi

bash evaluation/chapter6_experiments/run_video_self_evolution_candidate.sh

echo "[$(date -Is)] Diverse candidate screening complete"
