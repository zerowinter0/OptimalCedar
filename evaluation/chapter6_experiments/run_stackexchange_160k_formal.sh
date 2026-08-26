#!/usr/bin/env bash
# Run the enlarged StackExchange matrix without overwriting the 10k evidence.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULT_ROOT="${RESULT_ROOT:-${REPO_ROOT}/outputs/chapter6_experiments/stackexchange_160k_scale_v2/formal}"
DATASET_PATH="${STACKEXCHANGE_DATASET_PATH:-datasets/stackexchange/redpajama-stackexchange-400000.jsonl}"
SOURCE_ROOT="${STACKEXCHANGE_SOURCE_ROOT:-${REPO_ROOT}/outputs/chapter6_experiments/stationary_formal_sequential_remaining_nine_v2/workloads/stackexchange}"
SOURCE_MATRIX="${SOURCE_ROOT}/formal_runs/stackexchange"
SOURCE_PROFILE="${SOURCE_ROOT}/profiles/stackexchange.yaml"
EXPECTED_SOURCE_RECORDS=400000
EXPECTED_OUTPUT_RECORDS=160000

cd "${REPO_ROOT}"
# shellcheck source=/dev/null
source env/bin/activate

python - "${DATASET_PATH}" "${EXPECTED_SOURCE_RECORDS}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected = int(sys.argv[2])
count = 0
digest = hashlib.sha256()
with path.open("rb") as stream:
    for line in stream:
        json.loads(line)
        digest.update(line)
        count += 1
if count != expected:
    raise SystemExit(f"expected {expected} source records in {path}, found {count}")
print(f"validated {count} records; sha256={digest.hexdigest()}")
PY

[[ -s "${SOURCE_PROFILE}" ]] || {
  echo "Missing reusable profile: ${SOURCE_PROFILE}" >&2
  exit 1
}
mkdir -p "${RESULT_ROOT}/profiles" \
  "${RESULT_ROOT}/formal_runs/stackexchange/plans" \
  "${RESULT_ROOT}/formal_runs/stackexchange/warmup_results"
cp -p "${SOURCE_PROFILE}" "${RESULT_ROOT}/profiles/stackexchange.yaml"

# Cache is disabled, so changing only the finite source/output cardinality does
# not alter these physical plans. Preserve all prior one-hour unavailability
# decisions as part of the same optimizer task protocol.
for plan in "${SOURCE_MATRIX}"/plans/*.yaml \
            "${SOURCE_MATRIX}"/plans/*.unavailable.json; do
  [[ -e "${plan}" ]] || continue
  destination="${RESULT_ROOT}/formal_runs/stackexchange/plans/$(basename "${plan}")"
  [[ -e "${destination}" ]] || cp -p "${plan}" "${destination}"
done

PROFILE_DIR="${RESULT_ROOT}/profiles" \
MATRIX_OUTPUT_ROOT="${RESULT_ROOT}/formal_runs" \
OPTIMIZER_SET=formal_seven \
PLAN_ONLY=0 \
REPEATS=3 \
RESUME_EXISTING=1 \
TASK_TIMEOUT_SEC=3600 \
STACKEXCHANGE_SAMPLES="${EXPECTED_OUTPUT_RECORDS}" \
STACKEXCHANGE_DATASET_PATH="${DATASET_PATH}" \
  bash evaluation/chapter6_experiments/run_formal_plan_and_matrix.sh \
    --workloads stackexchange

printf 'complete\n' > "${RESULT_ROOT}/COMPLETE"
