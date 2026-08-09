#!/usr/bin/env bash
# End-to-end formal run for the Data-Juicer general video refinement workload.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_DIR="${REPO_ROOT}/evaluation/chapter6_experiments"
FORMAL_ROOT="${GENERAL_VIDEO_FORMAL_ROOT:-${REPO_ROOT}/outputs/chapter6_experiments/general_video_refine_formal}"
PROFILE_DIR="${FORMAL_ROOT}/profiles"
MATRIX_ROOT="${FORMAL_ROOT}/matrix"
RESUME="${GENERAL_VIDEO_RESUME:-0}"

if [[ "${RESUME}" != "0" && "${RESUME}" != "1" ]]; then
  echo "GENERAL_VIDEO_RESUME must be 0 or 1" >&2
  exit 2
fi
if [[ -e "${FORMAL_ROOT}/COMPLETE" ]]; then
  echo "Formal run is already complete: ${FORMAL_ROOT}" >&2
  exit 2
fi
if [[ -e "${FORMAL_ROOT}" && "${RESUME}" != "1" ]]; then
  echo "Formal run directory exists: ${FORMAL_ROOT}" >&2
  echo "Use GENERAL_VIDEO_RESUME=1 only after auditing existing state." >&2
  exit 2
fi

cd "${REPO_ROOT}"
source env/bin/activate
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export CEDAR_PROFILE_RAY_ACTORS=1
export CEDAR_PROFILE_SMP_PROCS=1
export GENERAL_VIDEO_REFINE_MP_START_METHOD=spawn
ulimit -n 65536

mkdir -p "${FORMAL_ROOT}" "${PROFILE_DIR}" "${MATRIX_ROOT}"
cp -p "${BASE_DIR}/GENERAL_VIDEO_REFINE_PROTOCOL.md" \
  "${FORMAL_ROOT}/PROTOCOL.md"
cp -p datasets/general_video_refine/dataset_metadata.json \
  "${FORMAL_ROOT}/dataset_metadata.json"
cp -p outputs/chapter6_experiments/general_video_refine_setup/operator_equivalence.json \
  "${FORMAL_ROOT}/operator_equivalence.json"

{
  printf 'started_at=%s\n' "$(date -Is)"
  printf 'git_commit=%s\n' "$(git rev-parse HEAD)"
  printf 'data_juicer_commit=%s\n' "$(git -C data-juicer rev-parse HEAD)"
  printf 'cpu_budget=64\nlocal_workers=8\nrepeats=3\n'
  printf 'profile_stage_seconds=10\nprofile_ray_actors=1\nprofile_smp_procs=1\n'
  printf 'cuda_execution_context=torch.inference_mode\n'
  printf 'optimizer_plan_timeout_sec=3600\nexecution_timeout_sec=10800\n'
  printf 'optimizers=optimizer dj_optimizer dp_cedar_optimizer dp_optimizer dp_two_stage_optimizer pecan_optimizer\n'
  nvidia-smi --query-gpu=name,uuid,memory.total,driver_version \
    --format=csv,noheader
} > "${FORMAL_ROOT}/metadata.txt"

if [[ ! -s "${PROFILE_DIR}/general_video_refine.yaml" ]]; then
  echo "[$(date -Is)] PROFILE general_video_refine"
  CH6_RESULT_ROOT="${FORMAL_ROOT}" \
  CH6_PROFILE_ROOT="${PROFILE_DIR}" \
  CH6_PROFILE_RUN_ID="general_video_refine" \
  CH6_PROFILE_TIMEOUT_SEC=10800 \
  bash "${BASE_DIR}/run_formal_profiles.sh" \
    --workloads general_video_refine
else
  echo "[$(date -Is)] REUSE audited profile ${PROFILE_DIR}/general_video_refine.yaml"
fi

echo "[$(date -Is)] PLAN AND EXECUTION MATRIX general_video_refine"
PROFILE_DIR="${PROFILE_DIR}" \
MATRIX_OUTPUT_ROOT="${MATRIX_ROOT}" \
CPU_BUDGET=64 \
LOCAL_WORKERS=8 \
REPEATS=3 \
OPTIMIZER_PLAN_TIMEOUT_SEC=3600 \
EXECUTION_TIMEOUT_SEC=10800 \
RESUME_EXISTING="${RESUME}" \
OPTIMIZER_SET=complete \
bash "${BASE_DIR}/run_formal_plan_and_matrix.sh" \
  --workloads general_video_refine

python - "${MATRIX_ROOT}/general_video_refine/results" <<'PY'
import json
import pathlib
import re
import statistics
import sys

root = pathlib.Path(sys.argv[1])
optimizers = (
    "optimizer",
    "dj_optimizer",
    "dp_cedar_optimizer",
    "dp_optimizer",
    "dp_two_stage_optimizer",
    "pecan_optimizer",
)
summary = {}
errors = []
fatal_runtime_pattern = re.compile(
    r"CUDA out of memory|OutOfMemoryError|RayTaskError|Traceback"
)
for optimizer in optimizers:
    values = []
    samples = []
    for path in sorted(root.glob(f"round*__{optimizer}.json")):
        payload = json.loads(path.read_text())
        values.extend(float(value) for value in payload["epoch_run_times"])
        samples.extend(int(value) for value in payload["epoch_num_samples"])
    timeout_files = [
        path.name
        for path in sorted(root.glob(f"round*__{optimizer}.timeout.json"))
    ]
    skipped_files = [
        path.name
        for path in sorted(root.glob(f"round*__{optimizer}.skipped.json"))
    ]
    plan = root.parent / "plans" / f"{optimizer}.yaml"
    runtime_failures = {}
    for log in sorted((root.parent / "logs").glob(f"round*__{optimizer}.log")):
        matches = sorted(set(fatal_runtime_pattern.findall(log.read_text(errors="replace"))))
        if matches:
            runtime_failures[log.name] = matches
    summary[optimizer] = {
        "completed_repeats": len(values),
        "mean_seconds": statistics.mean(values) if values else None,
        "sample_stdev_seconds": statistics.stdev(values) if len(values) > 1 else None,
        "samples": samples,
        "timeout_files": timeout_files,
        "skipped_files": skipped_files,
        "plan": str(plan),
        "plan_exists": plan.is_file(),
        "runtime_failures": runtime_failures,
    }
    if not plan.is_file():
        errors.append(f"{optimizer}: missing materialized plan")
    if len(values) != 3:
        errors.append(f"{optimizer}: expected 3 repeats, found {len(values)}")
    if samples != [10000, 10000, 10000]:
        errors.append(f"{optimizer}: invalid output counts {samples}")
    if timeout_files or skipped_files:
        errors.append(
            f"{optimizer}: timeout={timeout_files}, skipped={skipped_files}"
        )
    if runtime_failures:
        errors.append(f"{optimizer}: fatal runtime log entries {runtime_failures}")
output = root.parent / "summary.json"
output.write_text(json.dumps(summary, indent=2) + "\n")
print(output)
if errors:
    failure = root.parent.parent.parent / "FAILED_VALIDATION"
    failure.write_text("\n".join(errors) + "\n")
    raise SystemExit("Formal result validation failed:\n" + "\n".join(errors))
PY

date -Is > "${FORMAL_ROOT}/COMPLETE"
echo "[$(date -Is)] COMPLETE ${FORMAL_ROOT}"
