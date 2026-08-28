#!/usr/bin/env bash
# Reprofile all ten paper workloads with multi-width actor curves, then run
# the seven formal optimizers round-robin under one common CPU budget.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULT_ROOT="${RESULT_ROOT:-${REPO_ROOT}/outputs/chapter6_experiments/joint_actor_budget_formal_all_ten_v1}"
PROFILE_ROOT="${RESULT_ROOT}/profiles"
MATRIX_ROOT="${RESULT_ROOT}/formal_runs"
WORKLOADS="${WORKLOADS_OVERRIDE:-general_video_refine,pile_hackernews,pile_europarl,stackexchange,pile_pubmed_abstracts,pile_uspto_backgrounds,alpaca_cot,simclrv2_cache,commonvoice,redpajama_code}"
PROFILE_RUN_ID="${PROFILE_RUN_ID:-joint_actor_budget_formal_all_ten_v1}"
MIN_AVAILABLE_GIB="${CEDAR_PROFILE_MIN_AVAILABLE_GIB:-64}"
MIN_FORMAL_RUNTIME_SEC="${MIN_FORMAL_RUNTIME_SEC:-1800}"
RUNTIME_FLOOR_WORKLOADS="${RUNTIME_FLOOR_WORKLOADS:-commonvoice,stackexchange,general_video_refine,pile_europarl,pile_hackernews,pile_pubmed_abstracts,pile_uspto_backgrounds,redpajama_code}"

cd "${REPO_ROOT}"
source env/bin/activate
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
mkdir -p "${RESULT_ROOT}" "${PROFILE_ROOT}" "${MATRIX_ROOT}"

cat > "${RESULT_ROOT}/PROTOCOL.md" <<EOF
# Joint-actor seven-optimizer, selected-workload formal experiment

## Workloads

Workload identifiers: ${WORKLOADS}.

## Optimizers

Cedar, DJ, Pecan, DJ-TwoStage, Pecan-TwoStage, Simple-DP, and PICO (DP).
The temporary PICO-Add ablation is excluded.

## Protocol

- Every workload receives a fresh adaptive layered profile from the same run.
- Baseline profiling lasts 10 seconds and captures legal inputs at every
  operator boundary.
- Every legal Ray/SMP operator is profiled with real operator inputs at global
  concurrency widths 1, 8, 16, 24, 32, 40, and 48, matching W=8 and each
  legal per-worker stage width 1..6. Width 1 runs for at least 3 seconds and 30
  observations, stopping at RSE <= 10% or 30 seconds; additional widths use
  the same confidence target with a 20-second cap. Scaling curves use Ray
  batch size 1 and epochs of at least four records per worker. Cheap operators
  repeat epochs until confidence converges; expensive operators therefore
  exercise every worker without allowing a fixed 100-record floor to override
  the per-width duration budget.
- Real legal objects provide Ray/SMP marshalling and boundary measurements.
- Width points that reach the confidence target are eligible for PICO's cost
  interpolation. Time-capped points remain in the profile with their RSE for
  auditability but are excluded from interpolation; PICO conservatively uses
  the nearest converged width when necessary.
- Runtime uses W=8 and CPU_BUDGET=64 for every optimizer.
- One CPU per worker is reserved for local execution and one for runtime
  coordination, leaving six remote actor/process slots per worker.
- Cedar, DJ, Pecan, their two-stage variants, and Simple-DP retain their
  selected stage topology and divide those six slots deterministically and
  equally among remote stages. PICO jointly chooses each remote stage width
  from 1..6 while respecting the same six-slot budget.
- Non-cache workloads disable cache. SimCLR-cache is independently warmed
  before formal measurement.
- Each optimizer runs three times in round-robin order.
- For every workload with sufficient source capacity, the median execution
  time of the slowest optimizer that completes all three rounds must be at
  least ${MIN_FORMAL_RUNTIME_SEC} seconds. The run is rejected rather than
  marked complete when this condition is not met.
- Alpaca-CoT and SimCLR-cache are capacity-limited exceptions: their frozen
  sources cannot provide enough distinct records to reach the duration floor.
- Optimization and first execution share one 3600-second deadline. A timeout
  skips that optimizer's remaining two rounds for the workload.
- PICO uses its current throughput objective without a total-work Pareto
  coordinate; CPU remote-stage service time is measured per actor at the
  chosen width and divided by that stage's actor count.
EOF

available_kib="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
if (( available_kib < MIN_AVAILABLE_GIB * 1024 * 1024 )); then
  printf '[%s] Insufficient memory: available_gib=%s required_gib=%s\n' \
    "$(date -Is)" "$((available_kib / 1024 / 1024))" "${MIN_AVAILABLE_GIB}"
  exit 1
fi

export CEDAR_LAYERED_ADAPTIVE_PROFILE=1
export CEDAR_PROFILE_TIME_SEC=10
export CEDAR_ADAPTIVE_PROFILE_MIN_SEC=3
export CEDAR_ADAPTIVE_PROFILE_MAX_SEC=30
export CEDAR_ADAPTIVE_PROFILE_TARGET_RSE=0.10
export CEDAR_ADAPTIVE_PROFILE_MIN_OBS=30
export CEDAR_PROFILE_POOL_SAMPLES=64
export CEDAR_PROFILE_POOL_BYTES_PER_PIPE=$((64 * 1024 * 1024))
export CEDAR_PROFILE_POOL_BYTES_TOTAL=$((512 * 1024 * 1024))
export CEDAR_PROFILE_SCALING_TOP_K=0
export CEDAR_PROFILE_SCALING_WIDTHS="1,8,16,24,32,40,48"
export CEDAR_PROFILE_SCALING_MAX_SEC=20
export CEDAR_PROFILE_SCALING_RAY_BATCH_SIZE=1
export CEDAR_PROFILE_SCALING_MIN_RECORDS_PER_WORKER=4
export CEDAR_PROFILE_BOUNDARY_MODEL=1
export CEDAR_PROFILE_INFER_COMPUTE_SCALING=1
export CEDAR_PROFILE_RAY_ACTORS=1
export CEDAR_PROFILE_SMP_PROCS=1
export CH6_RESULT_ROOT="${RESULT_ROOT}"
export CH6_PROFILE_ROOT="${PROFILE_ROOT}"
export CH6_PROFILE_RUN_ID="${PROFILE_RUN_ID}"
export CH6_PROFILE_TIMEOUT_SEC=10800
export INCREMENTAL_BACKEND_COMPUTE=0

if [[ "${REUSE_COMPLETED_PROFILES:-0}" == "1" ]]; then
  printf '[%s] PHASE reuse completed unified profiles\n' "$(date -Is)"
else
  printf '[%s] PHASE fresh unified profiles\n' "$(date -Is)"
  bash evaluation/chapter6_experiments/run_formal_profiles.sh \
    --workloads "${WORKLOADS}"
fi

python - "${PROFILE_ROOT}" "${WORKLOADS}" \
  "${RESULT_ROOT}/profile_validation.json" <<'PY'
import json
import math
import sys
from pathlib import Path
import yaml

root = Path(sys.argv[1])
workloads = sys.argv[2].split(",")
output = Path(sys.argv[3])
report = {}
required_widths = {1, 8, 16, 24, 32, 40, 48}
for workload in workloads:
    path = root / f"{workload}.yaml"
    profile = yaml.safe_load(path.read_text())
    physical = profile.get("physical_model", {})
    object_boundaries = physical.get("object_boundary", {})
    backend_counts = {}
    for backend in ("RAY", "SMP"):
        count = len(object_boundaries.get(backend, {}).get("operators", {}))
        if count < 1:
            raise RuntimeError(
                f"{path} has no real-object {backend} boundary measurements"
            )
        backend_counts[backend] = count
    offloads = profile.get("offloads", {})
    scaling = physical.get("scaling", {})
    expected_by_family = {
        "RAY": set(offloads.get("RAY", {})) | set(offloads.get("TF_RAY", {})),
        "SMP": set(offloads.get("SMP", {})),
    }
    scaling_counts = {}
    convergence_counts = {}
    for family, expected_ids in expected_by_family.items():
        entries = scaling.get(family, {})
        normalized_entries = {str(key): value for key, value in entries.items()}
        normalized_expected = {str(key) for key in expected_ids}
        if set(normalized_entries) != normalized_expected:
            raise RuntimeError(
                f"{path} {family} scaling/operator mismatch: "
                f"expected={sorted(normalized_expected)}, "
                f"actual={sorted(normalized_entries)}"
            )
        converged_points = 0
        capped_points = 0
        for pipe_id, entry in normalized_entries.items():
            widths = {
                int(key): value for key, value in entry.get("widths", {}).items()
            }
            if set(widths) != required_widths:
                raise RuntimeError(
                    f"{path} {family} pipe {pipe_id} widths: "
                    f"expected={sorted(required_widths)}, actual={sorted(widths)}"
                )
            for width, timing in widths.items():
                adaptive = timing.get("adaptive_profile", {})
                try:
                    mean = float(timing["mean_ms_per_sample"])
                    observations = int(timing["count"])
                    measured_records = int(timing["measured_input_records"])
                    minimum_epoch = int(
                        adaptive["minimum_parallel_epoch_records"]
                    )
                    observed_rse = float(adaptive["observed_rse"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"{path} {family} pipe {pipe_id} width {width} "
                        "has incomplete timing metadata"
                    ) from exc
                if not math.isfinite(mean) or mean < 0 or observations < 1:
                    raise RuntimeError(
                        f"{path} {family} pipe {pipe_id} width {width} "
                        "has no finite worker timing observations"
                    )
                if measured_records < minimum_epoch:
                    raise RuntimeError(
                        f"{path} {family} pipe {pipe_id} width {width} "
                        "did not exercise every worker"
                    )
                if not math.isfinite(observed_rse) or observed_rse < 0:
                    raise RuntimeError(
                        f"{path} {family} pipe {pipe_id} width {width} "
                        "has invalid uncertainty metadata"
                    )
                if adaptive.get("converged") is True:
                    if observations < int(adaptive["min_observations"]):
                        raise RuntimeError(
                            f"{path} {family} pipe {pipe_id} width {width} "
                            "claims convergence with too few observations"
                        )
                    converged_points += 1
                elif adaptive.get("stop_reason") == "max_duration":
                    capped_points += 1
                else:
                    raise RuntimeError(
                        f"{path} {family} pipe {pipe_id} width {width} "
                        "has an invalid adaptive stop state"
                    )
        scaling_counts[family] = len(normalized_entries)
        convergence_counts[family] = {
            "converged_points": converged_points,
            "time_capped_points": capped_points,
        }
    report[workload] = {
        "profile": str(path),
        "object_boundary_operator_counts": backend_counts,
        "multiwidth_scaling_operator_counts": scaling_counts,
        "multiwidth_confidence_counts": convergence_counts,
        "required_actor_widths": sorted(required_widths),
        "resource_config": profile.get("resource_config"),
    }
output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print("Validated all unified profiles")
PY
printf 'complete\n' > "${RESULT_ROOT}/PROFILES_COMPLETE"

unset CEDAR_LAYERED_ADAPTIVE_PROFILE
if [[ "${VALIDATE_ONLY:-0}" == "1" ]]; then
  printf '[%s] COMPLETE profile validation only\n' "$(date -Is)"
  exit 0
fi
printf '[%s] PHASE unified seven-optimizer matrix\n' "$(date -Is)"
PROFILE_DIR="${PROFILE_ROOT}" \
MATRIX_OUTPUT_ROOT="${MATRIX_ROOT}" \
OPTIMIZER_SET=formal_seven \
PLAN_ONLY=0 \
REPEATS=3 \
RESUME_EXISTING="${RESUME_FORMAL_RESULTS:-0}" \
TASK_TIMEOUT_SEC=3600 \
  bash evaluation/chapter6_experiments/run_formal_plan_and_matrix.sh \
    --workloads "${WORKLOADS}"

python - "${MATRIX_ROOT}" "${WORKLOADS}" \
  "${RESULT_ROOT}/execution_summary.json" <<'PY'
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
workloads = sys.argv[2].split(",")
output = Path(sys.argv[3])
optimizers = (
    "optimizer",
    "dj_optimizer",
    "pecan_optimizer",
    "dj_two_stage_optimizer",
    "pecan_two_stage_optimizer",
    "simple_dp_optimizer",
    "dp_optimizer",
)
summary = {}
for workload in workloads:
    result_root = root / workload / "results"
    row = {}
    for optimizer in optimizers:
        times = []
        for path in sorted(result_root.glob(f"round*__{optimizer}.json")):
            payload = json.loads(path.read_text())
            times.extend(float(x) for x in payload.get("epoch_run_times", []))
        row[optimizer] = {
            "successful_rounds": len(times),
            "times_sec": times,
            "mean_sec": statistics.mean(times) if times else None,
            "timeouts": len(
                list(result_root.glob(f"round*__{optimizer}.timeout.json"))
            ),
            "skipped": len(
                list(result_root.glob(f"round*__{optimizer}.skipped.json"))
            ),
        }
    summary[workload] = row
output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
PY

python evaluation/chapter6_experiments/verify_runtime_floor.py \
  --matrix-root "${MATRIX_ROOT}" \
  --workloads "${WORKLOADS}" \
  --required-workloads "${RUNTIME_FLOOR_WORKLOADS}" \
  --minimum-seconds "${MIN_FORMAL_RUNTIME_SEC}" \
  --required-rounds 3 \
  --output "${RESULT_ROOT}/runtime_floor_validation.json"

printf 'complete\n' > "${RESULT_ROOT}/COMPLETE"
printf '[%s] COMPLETE unified seven-optimizer matrix\n' "$(date -Is)"
