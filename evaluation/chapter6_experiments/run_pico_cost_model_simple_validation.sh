#!/usr/bin/env bash
# Fresh object-aware profiles and a three-round seven-optimizer gate on the two
# short workloads. Complex-workload execution is intentionally separate.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULT_ROOT="${RESULT_ROOT:-${REPO_ROOT}/outputs/chapter6_experiments/pico_width_aware_simple_validation_v3}"
PROFILE_ROOT="${RESULT_ROOT}/profiles"
MATRIX_ROOT="${RESULT_ROOT}/formal_runs"
WORKLOADS="alpaca_cot,commonvoice"

cd "${REPO_ROOT}"
source env/bin/activate
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
mkdir -p "${RESULT_ROOT}" "${PROFILE_ROOT}" "${MATRIX_ROOT}"

cat > "${RESULT_ROOT}/PROTOCOL.md" <<'EOF'
# PICO cost-model simple-workload validation

- Workloads: Alpaca-CoT and CommonVoice.
- A fresh common adaptive layered profile is collected for each workload.
- Legal prefix objects are replayed through real one-actor/process Ray/SMP
  identity stages. This directly measures queueing, task submission,
  serialization, transport, and result delivery for each physical boundary.
- Profile resources are one local worker and one actor/process per stage;
  targeted backend scaling is measured at W=8. Each scaling epoch contains
  at least two complete submit batches per worker, so every actor/process is
  exercised.
- PICO consumes converged W=8 worker compute costs. Multi-Ray-stage plans cap
  submit batch size so ideal finite-workload fill/drain overhead is at most
  10%.
- PICO no longer retains parallel total work as a Pareto coordinate.
- PICO and all six comparison optimizers use identical profiles, W=8, CPU budget 64, workload
  size, cache policy, and three-round round-robin execution.
- Complex workloads are not started here. Inspect this gate first.
EOF

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
export CEDAR_PROFILE_SCALING_MAX_SEC=20
export CEDAR_PROFILE_BOUNDARY_MODEL=1
export CEDAR_PROFILE_INFER_COMPUTE_SCALING=1
export CEDAR_PROFILE_RAY_ACTORS=1
export CEDAR_PROFILE_SMP_PROCS=1
export CH6_RESULT_ROOT="${RESULT_ROOT}"
export CH6_PROFILE_ROOT="${PROFILE_ROOT}"
export CH6_PROFILE_RUN_ID="pico_width_aware_gate"
export CH6_PROFILE_TIMEOUT_SEC=10800
export INCREMENTAL_BACKEND_COMPUTE=0

printf '[%s] PHASE fresh profiles\n' "$(date -Is)"
bash evaluation/chapter6_experiments/run_formal_profiles.sh \
  --workloads "${WORKLOADS}"

python - "${PROFILE_ROOT}" <<'PY'
import sys
from pathlib import Path
import yaml

root = Path(sys.argv[1])
for workload in ("alpaca_cot", "commonvoice"):
    path = root / f"{workload}.yaml"
    profile = yaml.safe_load(path.read_text())
    objects = profile.get("physical_model", {}).get("object_boundary", {})
    scaling = profile.get("physical_model", {}).get("scaling", {})
    for backend in ("RAY", "SMP"):
        operators = objects.get(backend, {}).get("operators", {})
        if not operators:
            raise RuntimeError(
                f"{path} has no real-object {backend} boundary measurements"
            )
        if not all(
            isinstance(entry.get("input_identity_stage"), dict)
            and isinstance(entry.get("output_identity_stage"), dict)
            and entry["input_identity_stage"].get("mean_ms_per_sample")
            is not None
            and entry["output_identity_stage"].get("mean_ms_per_sample")
            is not None
            for entry in operators.values()
        ):
            raise RuntimeError(
                f"{path} has incomplete real identity-stage {backend} data"
            )
        width_entries = scaling.get(backend, {})
        if not width_entries:
            raise RuntimeError(f"{path} has no width profile for {backend}")
        for pipe_id, timing in width_entries.items():
            adaptive = timing.get("adaptive_profile", {})
            if adaptive.get("converged") is not True:
                raise RuntimeError(
                    f"{path} {backend}:{pipe_id} width profile did not converge"
                )
            if adaptive.get("epoch_records", 0) < adaptive.get(
                "minimum_parallel_epoch_records", 1
            ):
                raise RuntimeError(
                    f"{path} {backend}:{pipe_id} did not exercise all workers"
                )
print("Validated real boundaries and converged width-aware profiles")
PY

unset CEDAR_LAYERED_ADAPTIVE_PROFILE
printf '[%s] PHASE round-robin plan and execution gate\n' "$(date -Is)"
PROFILE_DIR="${PROFILE_ROOT}" \
MATRIX_OUTPUT_ROOT="${MATRIX_ROOT}" \
OPTIMIZER_SET=formal_seven \
PLAN_ONLY=0 \
REPEATS=3 \
RESUME_EXISTING=0 \
TASK_TIMEOUT_SEC=3600 \
  bash evaluation/chapter6_experiments/run_formal_plan_and_matrix.sh \
    --workloads "${WORKLOADS}"

python - "${MATRIX_ROOT}" "${RESULT_ROOT}/gate_summary.json" <<'PY'
import hashlib
import json
import statistics
import sys
from pathlib import Path
import yaml

root = Path(sys.argv[1])
output = Path(sys.argv[2])
summary = {}
for workload in ("alpaca_cot", "commonvoice"):
    work = root / workload
    row = {"optimizers": {}}
    for optimizer in ("optimizer", "dj_optimizer", "pecan_optimizer", "dj_two_stage_optimizer", "pecan_two_stage_optimizer", "simple_dp_optimizer", "dp_optimizer"):
        durations = []
        pattern = f"round*__{optimizer}.json"
        for result in sorted((work / "results").glob(pattern)):
            payload = json.loads(result.read_text())
            durations.extend(float(x) for x in payload["epoch_run_times"])
        plan_path = work / "plans" / f"{optimizer}.yaml"
        plan_hash = None
        if plan_path.exists():
            physical = yaml.safe_load(plan_path.read_text())["physical_plan"]
            canonical = json.dumps(
                physical, sort_keys=True, separators=(",", ":")
            )
            plan_hash = hashlib.sha256(canonical.encode()).hexdigest()
        row["optimizers"][optimizer] = {
            "rounds": len(durations),
            "times_sec": durations,
            "mean_sec": statistics.mean(durations) if durations else None,
            "median_sec": statistics.median(durations) if durations else None,
            "physical_plan_sha256": plan_hash,
        }
    means = {
        name: value["mean_sec"]
        for name, value in row["optimizers"].items()
        if value["mean_sec"] is not None
    }
    row["winner"] = min(means, key=means.get) if means else None
    summary[workload] = row
output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

printf 'complete\n' > "${RESULT_ROOT}/COMPLETE"
printf '[%s] COMPLETE simple gate; inspect before complex workloads\n' \
  "$(date -Is)"
