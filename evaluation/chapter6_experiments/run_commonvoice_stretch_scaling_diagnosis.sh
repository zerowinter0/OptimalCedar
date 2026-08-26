#!/usr/bin/env bash
# Repeat the CommonVoice Stretch scaling points under controlled Ray batches.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULT_ROOT="${RESULT_ROOT:-${REPO_ROOT}/outputs/chapter6_experiments/commonvoice_stretch_scaling_diagnosis_v1}"

cd "${REPO_ROOT}"
source env/bin/activate
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
mkdir -p "${RESULT_ROOT}"

vmstat 2 > "${RESULT_ROOT}/vmstat.log" &
monitor_pid=$!
cleanup() {
  kill "${monitor_pid}" 2>/dev/null || true
}
trap cleanup EXIT

for batch_size in 1 10; do
  for repeat in 1 2 3; do
    run_name="batch${batch_size}_repeat${repeat}"
    run_root="${RESULT_ROOT}/${run_name}"
    printf '[%s] START %s\n' "$(date -Is)" "${run_name}"
    env \
      CEDAR_LAYERED_ADAPTIVE_PROFILE=1 \
      CEDAR_PROFILE_TIME_SEC=10 \
      CEDAR_ADAPTIVE_PROFILE_MIN_SEC=3 \
      CEDAR_ADAPTIVE_PROFILE_MAX_SEC=30 \
      CEDAR_ADAPTIVE_PROFILE_TARGET_RSE=0.10 \
      CEDAR_ADAPTIVE_PROFILE_MIN_OBS=30 \
      CEDAR_PROFILE_POOL_SAMPLES=64 \
      CEDAR_PROFILE_POOL_BYTES_PER_PIPE=$((64 * 1024 * 1024)) \
      CEDAR_PROFILE_POOL_BYTES_TOTAL=$((512 * 1024 * 1024)) \
      CEDAR_PROFILE_SCALING_TOP_K=1 \
      CEDAR_PROFILE_SCALING_WIDTHS="32,40,48" \
      CEDAR_PROFILE_SCALING_MAX_SEC=30 \
      CEDAR_PROFILE_SCALING_RAY_BATCH_SIZE="${batch_size}" \
      CEDAR_PROFILE_SCALING_MIN_RECORDS_PER_WORKER=100 \
      CEDAR_PROFILE_BOUNDARY_MODEL=1 \
      CEDAR_PROFILE_INFER_COMPUTE_SCALING=1 \
      CEDAR_PROFILE_RAY_ACTORS=1 \
      CEDAR_PROFILE_SMP_PROCS=1 \
      CH6_RESULT_ROOT="${run_root}" \
      CH6_PROFILE_ROOT="${run_root}/profiles" \
      CH6_PROFILE_RUN_ID="${run_name}" \
      CH6_PROFILE_TIMEOUT_SEC=10800 \
      INCREMENTAL_BACKEND_COMPUTE=0 \
      bash evaluation/chapter6_experiments/run_formal_profiles.sh \
        --workloads commonvoice
    printf '[%s] COMPLETE %s\n' "$(date -Is)" "${run_name}"
  done
done

python - "${RESULT_ROOT}" <<'PY'
import json
import statistics
import sys
from pathlib import Path

import yaml

root = Path(sys.argv[1])
raw = {}
for batch_size in (1, 10):
    raw[str(batch_size)] = {}
    for repeat in (1, 2, 3):
        name = f"batch{batch_size}_repeat{repeat}"
        path = root / name / "profiles/commonvoice.yaml"
        profile = yaml.safe_load(path.read_text(encoding="utf-8"))
        ray = profile["physical_model"]["scaling"]["RAY"]
        # top-k=1 is selected independently from measured isolated cost.
        if len(ray) != 1:
            raise RuntimeError(f"Expected one Ray scaling operator in {path}")
        pipe_id, entry = next(iter(ray.items()))
        points = {}
        for width, timing in entry["widths"].items():
            points[str(width)] = {
                "mean_ms_per_sample": timing["mean_ms_per_sample"],
                "stderr_ms_per_sample": timing["stderr_ms_per_sample"],
                "count": timing["count"],
                "measured_input_records": timing["measured_input_records"],
                "adaptive_profile": timing["adaptive_profile"],
            }
        raw[str(batch_size)][str(repeat)] = {
            "pipe_id": int(pipe_id),
            "points": points,
        }

summary = {}
for batch_size, repeats in raw.items():
    summary[batch_size] = {}
    widths = sorted(
        {width for run in repeats.values() for width in run["points"]},
        key=int,
    )
    for width in widths:
        values = [
            float(run["points"][width]["mean_ms_per_sample"])
            for run in repeats.values()
        ]
        summary[batch_size][width] = {
            "values_ms": values,
            "mean_ms": statistics.mean(values),
            "cv": statistics.stdev(values) / statistics.mean(values),
        }

(root / "summary.json").write_text(
    json.dumps({"raw": raw, "summary": summary}, indent=2, sort_keys=True)
    + "\n",
    encoding="utf-8",
)
PY

printf '[%s] ALL DIAGNOSTICS COMPLETE\n' "$(date -Is)"
