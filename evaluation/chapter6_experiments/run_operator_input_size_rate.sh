#!/usr/bin/env bash
# Reproducible single-node operator-size microbenchmark for the paper.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/chapter6_experiments/operator_input_size_rate}"
MAX_SOURCE_RECORDS="${MAX_SOURCE_RECORDS:-5000}"

cd "${REPO_ROOT}"
source env/bin/activate
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=2

if [[ -e "${OUTPUT_ROOT}/results" ]]; then
  echo "Refusing to overwrite existing results: ${OUTPUT_ROOT}/results" >&2
  exit 1
fi
mkdir -p "${OUTPUT_ROOT}"
cp evaluation/chapter6_experiments/OPERATOR_INPUT_SIZE_RATE_PROTOCOL.md \
  "${OUTPUT_ROOT}/PROTOCOL.md"

python evaluation/chapter6_experiments/benchmark_operator_input_size_rate.py \
  --dataset datasets/stackexchange/redpajama-stackexchange-35000.jsonl \
  --output-dir "${OUTPUT_ROOT}/results" \
  --max-source-records "${MAX_SOURCE_RECORDS}"
