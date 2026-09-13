#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="outputs/motivation_multimodal/formal"
PILOT_ROOT="outputs/motivation_multimodal/pilot"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --pilot-root) PILOT_ROOT="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"
source /workspace/OptimalCedar/env/bin/activate
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUTPUT_ROOT"
PID_FILE="$OUTPUT_ROOT/run.pid"
LOG_FILE="$OUTPUT_ROOT/run.log"
if [[ -f "$PID_FILE" ]] && kill -0 "$(<"$PID_FILE")" 2>/dev/null; then
  echo "Formal comparison is already running with PID $(<"$PID_FILE")" >&2
  exit 1
fi

export W=1
export CPU_BUDGET=64
export CEDAR_PROFILE_TIME_SEC=10
export CEDAR_PROFILE_RAY_ACTORS=1
export CEDAR_PROFILE_SMP_PROCS=1
export CEDAR_PROFILE_FILTER_SELECTIVITY=1
export CEDAR_MATCH_PROFILE_RESOURCES=1
export CEDAR_PROFILE_MATCH_CPU_BUDGET="$CPU_BUDGET"
export CEDAR_PROFILE_MATCH_RAY_CPU_BUDGET="$CPU_BUDGET"
export CEDAR_PROFILE_MATCH_FIXED_LOCAL_WORKERS="$W"
{
  echo "W=$W"
  echo "CPU_BUDGET=$CPU_BUDGET"
  echo "pilot_root=$PILOT_ROOT"
  date --iso-8601=seconds
} > "$OUTPUT_ROOT/metadata.txt"

python -m evaluation.motivation_multimodal.formal \
  --fixture-root "$REPO_ROOT/outputs/motivation_multimodal/fixture" \
  --image-root /workspace/OptimalCedar/datasets/coco/val2017 \
  --pilot-root "$PILOT_ROOT" \
  --output-root "$OUTPUT_ROOT" > "$LOG_FILE" 2>&1 &
CHILD_PID=$!
echo "$CHILD_PID" > "$PID_FILE"
if wait "$CHILD_PID"; then
  echo success > "$OUTPUT_ROOT/SUCCESS"
else
  STATUS=$?
  echo "$STATUS" > "$OUTPUT_ROOT/FAILED"
  exit "$STATUS"
fi
