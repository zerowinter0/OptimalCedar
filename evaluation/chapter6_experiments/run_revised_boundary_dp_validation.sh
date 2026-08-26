#!/usr/bin/env bash
# Regenerate the ten current DP plans, compare them with the preceding plan
# archive, and execute three formal rounds only for workloads whose physical
# plan changed.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNNER="${REPO_ROOT}/evaluation/chapter6_experiments/run_formal_plan_and_matrix.sh"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/chapter6_experiments/revised_boundary_dp_validation}"
PLAN_ROOT="${OUTPUT_ROOT}/plan_gate"
FORMAL_ROOT="${OUTPUT_ROOT}/formal_runs"
BASELINE_ROOT="${BASELINE_ROOT:-${REPO_ROOT}/outputs/chapter6_experiments/dp_batch_latency_plan_comparison}"
CUSTOM_PROFILES="${REPO_ROOT}/outputs/chapter6_experiments/operator_scaling_two_stage_study/profiles"
PAPER_PROFILES="${REPO_ROOT}/evaluation/chapter6_experiments/formal_results/paper_artifacts/optimizer/profiles"
TASK_TIMEOUT_SEC="${TASK_TIMEOUT_SEC:-3600}"

WORKLOADS=(
  general_video_refine
  pile_hackernews
  pile_europarl
  stackexchange
  pile_pubmed_abstracts
  pile_uspto_backgrounds
  alpaca_cot
  simclrv2_cache
  commonvoice
  redpajama_code
)

profile_root_for() {
  local workload="$1"
  if [[ -f "${CUSTOM_PROFILES}/${workload}.yaml" ]]; then
    printf '%s\n' "${CUSTOM_PROFILES}"
  else
    printf '%s\n' "${PAPER_PROFILES}"
  fi
}

run_one() {
  local workload="$1"
  local output_root="$2"
  local plan_only="$3"
  local profile_root
  profile_root="$(profile_root_for "${workload}")"
  PROFILE_DIR="${profile_root}" \
  MATRIX_OUTPUT_ROOT="${output_root}" \
  OPTIMIZER_SET=dp_only \
  PLAN_ONLY="${plan_only}" \
  REPEATS=3 \
  RESUME_EXISTING=1 \
  TASK_TIMEOUT_SEC="${TASK_TIMEOUT_SEC}" \
    bash "${RUNNER}" --workloads "${workload}"
}

mkdir -p "${PLAN_ROOT}" "${FORMAL_ROOT}"
printf '[%s] PHASE plan regeneration\n' "$(date -Is)"
for workload in "${WORKLOADS[@]}"; do
  if [[ -f "${PLAN_ROOT}/${workload}/plans/dp_optimizer.yaml" || \
        -f "${PLAN_ROOT}/${workload}/plans/dp_optimizer.unavailable.json" ]]; then
    printf '[%s] REUSE plan gate %s\n' "$(date -Is)" "${workload}"
    continue
  fi
  run_one "${workload}" "${PLAN_ROOT}" 1
done

python - "${BASELINE_ROOT}" "${PLAN_ROOT}" "${OUTPUT_ROOT}" \
  "${WORKLOADS[@]}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import yaml

baseline_root = Path(sys.argv[1])
plan_root = Path(sys.argv[2])
output_root = Path(sys.argv[3])
workloads = sys.argv[4:]


def canonical(path: Path):
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    physical = payload["physical_plan"]
    encoded = json.dumps(physical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


records = []
changed = []
for workload in workloads:
    old_plan = baseline_root / workload / "plans/dp_optimizer.yaml"
    new_plan = plan_root / workload / "plans/dp_optimizer.yaml"
    record = {
        "workload": workload,
        "baseline_plan": str(old_plan),
        "new_plan": str(new_plan),
    }
    if not old_plan.exists():
        record["status"] = "missing_baseline"
    elif not new_plan.exists():
        record["status"] = "new_plan_unavailable"
    else:
        record["baseline_sha256"] = canonical(old_plan)
        record["new_sha256"] = canonical(new_plan)
        record["status"] = (
            "changed"
            if record["baseline_sha256"] != record["new_sha256"]
            else "unchanged"
        )
        if record["status"] == "changed":
            changed.append(workload)
    records.append(record)

report = {
    "comparison": "canonical physical_plan SHA-256",
    "baseline_root": str(baseline_root),
    "plan_root": str(plan_root),
    "changed_workloads": changed,
    "records": records,
}
(output_root / "plan_changes.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
(output_root / "changed_workloads.txt").write_text(
    "\n".join(changed) + ("\n" if changed else ""), encoding="utf-8"
)
print("Changed workloads:", ", ".join(changed) if changed else "none")
PY

printf '[%s] PHASE formal DP execution\n' "$(date -Is)"
while IFS= read -r workload; do
  [[ -n "${workload}" ]] || continue
  run_one "${workload}" "${FORMAL_ROOT}" 0
done < "${OUTPUT_ROOT}/changed_workloads.txt"

python - "${OUTPUT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
report = json.loads((root / "plan_changes.json").read_text())
statuses = {}
for workload in report["changed_workloads"]:
    result_root = root / "formal_runs" / workload / "results"
    statuses[workload] = {
        "successful_rounds": len(list(result_root.glob("round*__dp_optimizer.json"))),
        "timeouts": len(list(result_root.glob("round*__dp_optimizer.timeout.json"))),
        "skipped": len(list(result_root.glob("round*__dp_optimizer.skipped.json"))),
    }
(root / "execution_summary.json").write_text(
    json.dumps(statuses, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
(root / "COMPLETE").write_text("complete\n", encoding="utf-8")
PY
printf '[%s] COMPLETE revised-boundary DP validation\n' "$(date -Is)"
