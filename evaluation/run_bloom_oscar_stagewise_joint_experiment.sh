#!/usr/bin/env bash
set -euo pipefail

# One-stop reproduction for the motivation experiment:
# 1. profile the real BLOOM/OSCAR workload, including both Ray and SMP variants;
# 2. compare Cedar staged optimization with PICO/joint DP by modeled cost;
# 3. execute the generated physical plans on a finite real-data subset.
#
# Usage:
#   bash evaluation/run_bloom_oscar_stagewise_joint_experiment.sh
#
# Optional environment overrides:
#   DATASET_PATH=/path/to/redpajama.jsonl
#   OUT_DIR=/tmp/bloom_oscar_stagewise_joint_repro
#   NUM_PROFILE_SAMPLES=50
#   NUM_TIMING_INPUT_LINES=10
#   TIMING_TIMEOUT_SEC=600
#   RUN_TIMING=0

cd "$(dirname "$0")/.."
source env/bin/activate

DATASET_PATH="${DATASET_PATH:-/tmp/redpajama_backup_3gb_for_bloom_oscar.jsonl}"
OUT_DIR="${OUT_DIR:-/tmp/bloom_oscar_stagewise_joint_repro}"
NUM_PROFILE_SAMPLES="${NUM_PROFILE_SAMPLES:-50}"
NUM_TIMING_INPUT_LINES="${NUM_TIMING_INPUT_LINES:-10}"
TIMING_TIMEOUT_SEC="${TIMING_TIMEOUT_SEC:-600}"
RUN_TIMING="${RUN_TIMING:-1}"

PROFILE_PATH="${PROFILE_PATH:-${OUT_DIR}/bloom_oscar_profile_with_smp.yml}"
COST_LOG="${OUT_DIR}/cost_compare.log"
TIMING_LOG="${OUT_DIR}/timing.log"

mkdir -p "${OUT_DIR}"

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "Dataset not found: ${DATASET_PATH}" >&2
  echo "Set DATASET_PATH to the real BLOOM/OSCAR input JSONL and rerun." >&2
  exit 1
fi

echo "== BLOOM/OSCAR staged-vs-joint experiment =="
echo "dataset: ${DATASET_PATH}"
echo "profile: ${PROFILE_PATH}"
echo "out_dir: ${OUT_DIR}"
echo "num_profile_samples: ${NUM_PROFILE_SAMPLES}"
echo "num_timing_input_lines: ${NUM_TIMING_INPUT_LINES}"
echo

echo "== Step 1/2: profile workload with Ray + SMP, then compare modeled cost =="
HOME=/tmp python evaluation/compare_stagewise_joint_bloom_oscar.py \
  --dataset_path "${DATASET_PATH}" \
  --profile_path "${PROFILE_PATH}" \
  --num_profile_samples "${NUM_PROFILE_SAMPLES}" \
  2>&1 | tee "${COST_LOG}"

echo
echo "Cost comparison log written to ${COST_LOG}"

echo
echo "== Profile offload coverage =="
python - "${PROFILE_PATH}" <<'PY'
import sys
import yaml

profile_path = sys.argv[1]
with open(profile_path, "r", encoding="utf-8") as f:
    stats = yaml.safe_load(f)
counts = {k: len(v) for k, v in stats.get("offloads", {}).items()}
print(counts)
if counts.get("SMP", 0) == 0:
    raise SystemExit("SMP profile is empty; this is not a rigorous SMP-inclusive run.")
PY

if [[ "${RUN_TIMING}" == "0" ]]; then
  echo
  echo "RUN_TIMING=0, skipping real plan execution."
  exit 0
fi

echo
echo "== Step 2/2: execute the two physical plans on a real-data subset =="
echo "Note: this step starts local Ray/SMP workers and may need unrestricted local sockets."
HOME=/tmp python evaluation/time_stagewise_joint_bloom_oscar.py \
  --dataset_path "${DATASET_PATH}" \
  --profile_path "${PROFILE_PATH}" \
  --out_dir "${OUT_DIR}/timing" \
  --num_input_lines "${NUM_TIMING_INPUT_LINES}" \
  --timeout_sec "${TIMING_TIMEOUT_SEC}" \
  2>&1 | tee "${TIMING_LOG}"

echo
echo "Timing log written to ${TIMING_LOG}"
echo "Timing YAML: ${OUT_DIR}/timing/timing_results.yml"
