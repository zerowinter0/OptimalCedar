#!/usr/bin/env bash
# Profile and evaluate a registered Data-Juicer candidate batch.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_DIR="${REPO_ROOT}/evaluation/chapter6_experiments"
FORMAL_ROOT="${DJ_CANDIDATE_OUTPUT_ROOT:-${BASE_DIR}/formal_results}"
PROFILE_DIR="${DJ_CANDIDATE_PROFILE_DIR:-${FORMAL_ROOT}/profiles}"
RUN_ID="${DJ_CANDIDATE_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${FORMAL_ROOT}/datajuicer_candidate_runs/${RUN_ID}"
LOG_ROOT="${RUN_ROOT}/logs"
RAY_ADDRESS="127.0.0.1:6379"
CPU_BUDGET=64
LOCAL_WORKERS=8
REPEATS="${DJ_CANDIDATE_REPEATS:-3}"
OUTPUTS="${DJ_CANDIDATE_OUTPUTS:-20000}"
OPTIMIZER_TIMEOUT_SEC="${DJ_OPTIMIZER_TIMEOUT_SEC:-3600}"
EXECUTION_TIMEOUT_SEC="${DJ_EXECUTION_TIMEOUT_SEC:-3600}"
PROFILE_TIMEOUT_SEC="${DJ_PROFILE_TIMEOUT_SEC:-3600}"
RESUME_RUN="${DJ_CANDIDATE_RESUME:-0}"
REUSE_PROFILE_RUN_ID="${DJ_REUSE_PROFILE_RUN_ID:-}"
REUSE_EXISTING_PROFILES="${DJ_REUSE_EXISTING_PROFILES:-0}"
OPTIMIZERS_CSV="${DJ_CANDIDATE_OPTIMIZERS:-optimizer,dj_optimizer,dp_cedar_optimizer,dp_optimizer,pecan_optimizer}"
IFS=',' read -r -a OPTIMIZERS <<< "${OPTIMIZERS_CSV}"
WORKLOADS_CSV="${DJ_CANDIDATE_WORKLOADS:-pile_europarl,redpajama_code}"
IFS=',' read -r -a WORKLOADS <<< "${WORKLOADS_CSV}"

case "${RUN_ID}" in
  *[!A-Za-z0-9_.-]*|"")
    echo "DJ_CANDIDATE_RUN_ID contains unsafe characters: ${RUN_ID}" >&2
    exit 2
    ;;
esac
if [[ -e "${RUN_ROOT}" && "${RESUME_RUN}" != "1" ]]; then
  echo "Candidate run already exists: ${RUN_ROOT}" >&2
  exit 2
fi
if [[ "${REUSE_EXISTING_PROFILES}" != "0" && \
      "${REUSE_EXISTING_PROFILES}" != "1" ]]; then
  echo "DJ_REUSE_EXISTING_PROFILES must be 0 or 1." >&2
  exit 2
fi
if (( ${#OPTIMIZERS[@]} == 0 )); then
  echo "DJ_CANDIDATE_OPTIMIZERS must select at least one optimizer." >&2
  exit 2
fi
DP_SELECTED=0
for optimizer in "${OPTIMIZERS[@]}"; do
  case "${optimizer}" in
    optimizer|dj_optimizer|dp_cedar_optimizer|dp_optimizer|pecan_optimizer) ;;
    *) echo "Unknown optimizer in DJ_CANDIDATE_OPTIMIZERS: ${optimizer}" >&2; exit 2 ;;
  esac
  [[ "${optimizer}" == "dp_optimizer" ]] && DP_SELECTED=1
done
for numeric_setting in REPEATS OUTPUTS OPTIMIZER_TIMEOUT_SEC \
  EXECUTION_TIMEOUT_SEC PROFILE_TIMEOUT_SEC; do
  numeric_value="${!numeric_setting}"
  if [[ ! "${numeric_value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${numeric_setting} must be a positive integer: ${numeric_value}" >&2
    exit 2
  fi
done

cd "${REPO_ROOT}"
# shellcheck source=/dev/null
source env/bin/activate
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TF_CPP_MIN_LOG_LEVEL=2
export CEDAR_PROFILE_RAY_ACTORS=1
export CEDAR_PROFILE_SMP_PROCS=1
unset CEDAR_LOCAL_WORKERS
ulimit -n 65536

mkdir -p "${LOG_ROOT}"
exec > >(tee -a "${RUN_ROOT}/candidate_matrix.log") 2>&1

dataset_file() {
  case "$1" in
    pile_europarl)
      printf '%s\n' "evaluation/pipelines/pile_europarl/cedar_dataset.py"
      ;;
    redpajama_code)
      printf '%s\n' "evaluation/pipelines/redpajama_code/cedar_dataset.py"
      ;;
    pile_hackernews)
      printf '%s\n' "evaluation/pipelines/pile_hackernews/cedar_dataset.py"
      ;;
    pile_pubmed_abstracts)
      printf '%s\n' \
        "evaluation/pipelines/pile_pubmed_abstracts/cedar_dataset.py"
      ;;
    pile_freelaw)
      printf '%s\n' "evaluation/pipelines/pile_freelaw/cedar_dataset.py"
      ;;
    pile_uspto_backgrounds)
      printf '%s\n' \
        "evaluation/pipelines/pile_uspto_backgrounds/cedar_dataset.py"
      ;;
    *) return 2 ;;
  esac
}

dataset_path() {
  case "$1" in
    pile_europarl)
      printf '%s\n' "datasets/pile_europarl/pile-europarl-raw.jsonl"
      ;;
    redpajama_code)
      printf '%s\n' \
        "datasets/redpajama_code/redpajama-github-raw-50000.jsonl"
      ;;
    pile_hackernews)
      printf '%s\n' \
        "datasets/pile_hackernews/pile-hackernews-raw-100000.jsonl"
      ;;
    pile_pubmed_abstracts)
      printf '%s\n' \
        "datasets/pile_pubmed_abstracts/pile-pubmed-abstracts-raw-100000.jsonl"
      ;;
    pile_freelaw)
      printf '%s\n' \
        "datasets/pile_freelaw/pile-freelaw-raw-100000.jsonl"
      ;;
    pile_uspto_backgrounds)
      printf '%s\n' \
        "datasets/pile_uspto_backgrounds/pile-uspto-backgrounds-raw-100000.jsonl"
      ;;
    *) return 2 ;;
  esac
}

feasibility_path() {
  case "$1" in
    pile_hackernews)
      printf '%s\n' \
        "datasets/pile_hackernews/pile-hackernews-raw-100000.feasibility.json"
      ;;
    pile_pubmed_abstracts)
      printf '%s\n' \
        "datasets/pile_pubmed_abstracts/pile-pubmed-abstracts-raw-100000.feasibility.json"
      ;;
    pile_freelaw)
      printf '%s\n' \
        "datasets/pile_freelaw/pile-freelaw-raw-100000.feasibility.json"
      ;;
    pile_uspto_backgrounds)
      printf '%s\n' \
        "datasets/pile_uspto_backgrounds/pile-uspto-backgrounds-raw-100000.feasibility.json"
      ;;
    *) return 1 ;;
  esac
}

validate_feasibility() {
  local workload="$1" path
  if ! path="$(feasibility_path "${workload}")"; then
    return 0
  fi
  python - "${path}" "${workload}" <<'PY'
import json
import sys

path, workload = sys.argv[1:]
with open(path, encoding="utf-8") as stream:
    result = json.load(stream)
if result.get("workload") != workload:
    raise RuntimeError(f"Feasibility workload mismatch: {path}")
status = result.get("status")
if status == "benchmarkable_timeout":
    # A bounded single-process precheck is not an execution feasibility
    # verdict. The formal parallel run retains its independent 3600 s bound
    # and detects source exhaustion from the measured sample count.
    raise SystemExit(0)
if status != "feasible":
    raise SystemExit(3)
if int(result.get("retained_records", 0)) < 20_000:
    raise SystemExit(3)
PY
}

feasibility_reason() {
  local workload="$1" path
  path="$(feasibility_path "${workload}")"
  python - "${path}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    result = json.load(stream)
if result.get("status") == "benchmarkable_timeout":
    print(
        "serial source-only precheck reached its operational bound; "
        "formal execution will determine runtime/source feasibility"
    )
    raise SystemExit(0)
print(
    "official recipe retained "
    f"{int(result.get('retained_records', 0))}/"
    f"{int(result.get('target_retained', 20_000))} records after scanning "
    f"{int(result.get('source_records_scanned', 0))} source records"
)
PY
}

write_status() {
  local workload="$1" tag="$2" status="$3" reason="$4"
  local status_dir="${RUN_ROOT}/${workload}/status"
  mkdir -p "${status_dir}"
  python - "${status_dir}/${tag}.json" "${status}" "${reason}" <<'PY'
import json
import pathlib
import sys

path, status, reason = sys.argv[1:]
pathlib.Path(path).write_text(
    json.dumps({"status": status, "reason": reason}, indent=2) + "\n",
    encoding="utf-8",
)
PY
}

validate_profile_file() {
  local profile="$1"
  [[ -f "${profile}" ]] || return 1
  python - "${profile}" <<'PY'
import sys, yaml
with open(sys.argv[1], encoding="utf-8") as stream:
    profile=yaml.safe_load(stream)
expected={"schema_version":1,"profile_scope":"single_local_worker",
"profile_local_workers":1,"actors_per_stage":1,
"ray_actors_per_stage":1,"smp_procs_per_stage":1}
if not isinstance(profile,dict) or profile.get("resource_config")!=expected:
    raise SystemExit(1)
baseline=profile.get("baseline"); offloads=profile.get("offloads")
if not isinstance(baseline,dict) or not isinstance(offloads,dict):
    raise SystemExit(1)
if "RAY" not in offloads or "SMP" not in offloads:
    raise SystemExit(1)
keys=("input_counts","output_counts","selectivities","selectivity_observation_sources")
if any(key not in baseline for key in keys):
    raise SystemExit(1)
ids=set(baseline["input_counts"])
if any(set(baseline[key])!=ids for key in keys[1:]):
    raise SystemExit(1)
PY
}

profile_completed_in_reuse_run() {
  local workload="$1" run_log
  [[ -n "${REUSE_PROFILE_RUN_ID}" ]] || return 1
  run_log="${FORMAL_ROOT}/profile_runs/${REUSE_PROFILE_RUN_ID}/profile_matrix.log"
  [[ -f "${run_log}" ]] || return 1
  grep -Fq "Completed ${workload} -> ${PROFILE_DIR}/${workload}.yaml" "${run_log}" || return 1
  validate_profile_file "${PROFILE_DIR}/${workload}.yaml"
}

profile_failure_already_recorded() {
  local path="${RUN_ROOT}/$1/status/profile.json"
  [[ -f "${path}" ]] || return 1
  python - "${path}" <<'PY'
import json,sys
with open(sys.argv[1],encoding="utf-8") as stream:
    status=json.load(stream).get("status")
if status not in {"profile_timeout","profile_failed"}: raise SystemExit(1)
PY
}

start_ray() {
  ray stop --force >/dev/null 2>&1 || true
  ray start --head --node-ip-address=127.0.0.1 --port=6379 \
    --num-cpus="${CPU_BUDGET}" --disable-usage-stats >/dev/null
}

stop_ray() {
  ray stop --force >/dev/null 2>&1 || true
}

validate_plan() {
  local plan="$1"
  python - "${plan}" "${LOCAL_WORKERS}" <<'PY'
import sys
import yaml

path, expected_workers = sys.argv[1], int(sys.argv[2])
with open(path, encoding="utf-8") as stream:
    doc = yaml.safe_load(stream)
physical = doc.get("physical_plan") if isinstance(doc, dict) else None
if not isinstance(physical, dict):
    raise RuntimeError(f"missing physical_plan: {path}")
if physical.get("n_local_workers") != expected_workers:
    raise RuntimeError(
        f"expected W={expected_workers}, got "
        f"{physical.get('n_local_workers')}: {path}"
    )
graph = physical.get("graph")
pipes = physical.get("pipes")
if not isinstance(graph, dict) or not isinstance(pipes, dict):
    raise RuntimeError(f"invalid physical graph: {path}")
for pipe_id in graph:
    desc = pipes.get(pipe_id, pipes.get(str(pipe_id)))
    if not isinstance(desc, dict):
        raise RuntimeError(f"missing descriptor for active pipe {pipe_id}")
    variant = desc.get("variant")
    ctx = desc.get("variant_ctx", {})
    if variant in ("RAY", "TF_RAY") and ctx.get("n_actors") != 1:
        raise RuntimeError(f"Ray width differs from profile: {desc}")
    if variant == "SMP" and ctx.get("n_procs") != 1:
        raise RuntimeError(f"SMP width differs from profile: {desc}")
PY
}

generate_plan() {
  local workload="$1" optimizer="$2"
  local workload_root="${RUN_ROOT}/${workload}"
  local dataset profile kwargs log result tmp_plan
  dataset="$(dataset_file "${workload}")"
  profile="${PROFILE_DIR}/${workload}.yaml"
  kwargs="dataset_path=$(dataset_path "${workload}")"
  log="${workload_root}/logs/plan__${optimizer}.log"
  result="${workload_root}/plan_results/${optimizer}.json"
  tmp_plan="/tmp/cedar_optimized_plan.yml"
  local -a args=(
    python evaluation/compare_optimizer_perf.py
    --dataset_file "${dataset}"
    --profiled_stats "${profile}"
    --dataset_kwargs "${kwargs}"
    --num_total_samples "${OUTPUTS}"
    --num_epochs 1
    --num_repeats 1
    --warmup_runs 0
    --full_data_run
    --use_ray
    --ray_ip "${RAY_ADDRESS}"
    --enable_local_parallelism
    --disable_caching
    --disable_cedar_runtime_timeout
    --match_profile_resources
    --cpu_budget "${CPU_BUDGET}"
    --fixed_local_workers_ablation "${LOCAL_WORKERS}"
    --optimizer_time_limit_sec "${OPTIMIZER_TIMEOUT_SEC}"
    --plan_only
    --optimizers "${optimizer}"
    --results_path "${result}"
  )

  if [[ "${RESUME_RUN}" == "1" &&
        -f "${workload_root}/plans/${optimizer}.yaml" ]] &&
     validate_plan "${workload_root}/plans/${optimizer}.yaml"; then
    echo "[$(date -Is)] REUSE PLAN ${workload}/${optimizer}"
    return 0
  fi

  echo "[$(date -Is)] PLAN ${workload}/${optimizer}"
  printf 'command:' > "${log}"
  printf ' %q' "${args[@]}" >> "${log}"
  printf '\n' >> "${log}"
  rm -f "${tmp_plan}"
  timeout --signal=TERM --kill-after=30s "${OPTIMIZER_TIMEOUT_SEC}" \
    "${args[@]}" >> "${log}" 2>&1
  local status=$?
  if [[ "${status}" -eq 124 || "${status}" -eq 137 ]]; then
    write_status "${workload}" "plan__${optimizer}" \
      "optimizer_timeout" \
      "plan optimization exceeded ${OPTIMIZER_TIMEOUT_SEC}s"
    return 0
  fi
  if [[ "${status}" -ne 0 || ! -f "${tmp_plan}" ]]; then
    write_status "${workload}" "plan__${optimizer}" \
      "optimizer_failed" "plan command exit status ${status}"
    return 0
  fi
  if ! validate_plan "${tmp_plan}" >> "${log}" 2>&1; then
    write_status "${workload}" "plan__${optimizer}" \
      "invalid_plan" "formal W=8/profile-width validation failed"
    return 0
  fi
  cp -p "${tmp_plan}" "${workload_root}/plans/${optimizer}.yaml"
}

execute_plan() {
  local workload="$1" optimizer="$2" round="$3"
  local workload_root="${RUN_ROOT}/${workload}"
  local plan="${workload_root}/plans/${optimizer}.yaml"
  local tag="round${round}__${optimizer}"
  local dataset kwargs log result
  if [[ ! -f "${plan}" ]]; then
    write_status "${workload}" "${tag}" "skipped" \
      "optimizer did not produce a valid plan"
    return 0
  fi
  dataset="$(dataset_file "${workload}")"
  kwargs="dataset_path=$(dataset_path "${workload}")"
  log="${workload_root}/logs/${tag}.log"
  result="${workload_root}/results/${tag}.json"

  if [[ "${RESUME_RUN}" == "1" && -f "${result}" ]] &&
     python - "${result}" "${OUTPUTS}" <<'PY'
import json
import math
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    payload = json.load(stream)
times = payload.get("epoch_run_times")
samples = payload.get("epoch_num_samples")
required = int(sys.argv[2])
valid = (
    isinstance(times, list)
    and len(times) == 1
    and math.isfinite(float(times[0]))
    and float(times[0]) > 0
    and isinstance(samples, list)
    and len(samples) == 1
    and int(samples[0]) >= required
)
raise SystemExit(0 if valid else 1)
PY
  then
    echo "[$(date -Is)] REUSE DONE ${workload}/${tag}"
    return 0
  fi
  local -a args=(
    python evaluation/eval_cedar.py
    --dataset_file "${dataset}"
    --master_feature_config "${plan}"
    --dataset_kwargs "${kwargs}"
    --num_total_samples "${OUTPUTS}"
    --num_epochs 1
    --use_ray
    --ray_ip "${RAY_ADDRESS}"
    --results_path "${result}"
  )

  echo "[$(date -Is)] RUN ${workload}/${tag}"
  printf 'command:' > "${log}"
  printf ' %q' "${args[@]}" >> "${log}"
  printf '\n' >> "${log}"
  timeout --signal=TERM --kill-after=30s "${EXECUTION_TIMEOUT_SEC}" \
    "${args[@]}" >> "${log}" 2>&1
  local status=$?
  if [[ "${status}" -eq 124 || "${status}" -eq 137 ]]; then
    write_status "${workload}" "${tag}" "infeasible_timeout" \
      "execution exceeded ${EXECUTION_TIMEOUT_SEC}s"
    return 0
  fi
  if [[ "${status}" -ne 0 || ! -f "${result}" ]]; then
    write_status "${workload}" "${tag}" "execution_failed" \
      "execution command exit status ${status}"
    return 0
  fi
  local processed_samples
  if ! processed_samples="$(
    python - "${result}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    payload = json.load(stream)
samples = payload.get("epoch_num_samples")
if not isinstance(samples, list) or len(samples) != 1:
    raise RuntimeError("expected exactly one measured epoch")
print(int(samples[0]))
PY
  )"; then
    write_status "${workload}" "${tag}" "execution_failed" \
      "result does not contain one valid epoch sample count"
    return 0
  fi
  if (( processed_samples < OUTPUTS )); then
    write_status "${workload}" "${tag}" "source_exhausted" \
      "source exhausted after ${processed_samples}/${OUTPUTS} retained samples"
    write_status "${workload}" "source_infeasible" "source_infeasible" \
      "deterministic recipe retains fewer than ${OUTPUTS} records"
    echo "[$(date -Is)] SOURCE_INFEASIBLE ${workload}: ${processed_samples}/${OUTPUTS}"
    return 0
  fi
  echo "[$(date -Is)] DONE ${workload}/${tag}"
}

write_metadata() {
  local workload="$1" workload_root="${RUN_ROOT}/$1"
  local local_data metadata
  local_data="$(dataset_path "${workload}")"
  metadata="${local_data%.jsonl}.metadata.json"
  {
    printf 'protocol=datajuicer_predeclared_candidate_w8\n'
    printf 'candidate_protocol=evaluation/chapter6_experiments/DATA_JUICER_CANDIDATE_PROTOCOL.md\n'
    printf 'workload=%s\n' "${workload}"
    printf 'dataset=%s\n' "${local_data}"
    printf 'dataset_sha256=%s\n' "$(sha256sum "${local_data}" | awk '{print $1}')"
    printf 'dataset_metadata=%s\n' "${metadata}"
    printf 'profile=%s\n' "${PROFILE_DIR#${REPO_ROOT}/}/${workload}.yaml"
    printf 'data_juicer_commit=%s\n' \
      "$(git -C data-juicer -c safe.directory="${REPO_ROOT}/data-juicer" rev-parse HEAD)"
    printf 'data_juicer_hub_commit=47fc345\n'
    printf 'local_workers=%s\n' "${LOCAL_WORKERS}"
    printf 'cpu_budget=%s\n' "${CPU_BUDGET}"
    printf 'profile_ray_actors=1\nprofile_smp_procs=1\nprofile_stage_seconds=10\n'
    printf 'profile_timeout_sec=%s\n' "${PROFILE_TIMEOUT_SEC}"
    printf 'profile_filter_selectivity=%s\n' \
      "${CEDAR_PROFILE_FILTER_SELECTIVITY:-1}"
    printf 'profile_selectivity_strategy=max_coverage_existing_pass\n'
    printf 'dp_objective=additive\n'
    printf 'outputs=%s\nrepeats=%s\n' "${OUTPUTS}" "${REPEATS}"
    printf 'optimizer_timeout_sec=%s\n' "${OPTIMIZER_TIMEOUT_SEC}"
    printf 'execution_timeout_sec=%s\n' "${EXECUTION_TIMEOUT_SEC}"
    printf 'cache=off\nomitted_operator=document_simhash_deduplicator\n'
    printf 'optimizers=%s\n' "${OPTIMIZERS[*]}"
  } > "${workload_root}/metadata.txt"
  cp -p "${metadata}" "${workload_root}/dataset_metadata.json"
}

echo "[$(date -Is)] Candidate run ${RUN_ID} started (resume=${RESUME_RUN})"
SOURCE_FEASIBLE_WORKLOADS=()
for workload in "${WORKLOADS[@]}"; do
  [[ -n "${workload}" ]] || { echo "empty workload" >&2; exit 2; }
  local_data="$(dataset_path "${workload}")"
  [[ -f "${local_data}" ]] || { echo "Missing candidate dataset: ${local_data}" >&2; exit 1; }
  validate_feasibility "${workload}"; feasibility_status=$?
  if [[ "${feasibility_status}" -eq 0 ]]; then
    SOURCE_FEASIBLE_WORKLOADS+=("${workload}")
  elif [[ "${feasibility_status}" -eq 3 ]]; then
    workload_root="${RUN_ROOT}/${workload}"
    mkdir -p "${workload_root}"/{plans,plan_results,results,status,logs}
    write_metadata "${workload}"
    reason="$(feasibility_reason "${workload}")"
    write_status "${workload}" source_infeasible source_infeasible "${reason}"
    cp -p "$(feasibility_path "${workload}")" "${workload_root}/source_feasibility.json"
  else
    echo "Missing or invalid feasibility evidence for ${workload}" >&2; exit 1
  fi
done
FEASIBLE_WORKLOADS=()
for workload in "${SOURCE_FEASIBLE_WORKLOADS[@]}"; do
  workload_root="${RUN_ROOT}/${workload}"
  mkdir -p "${workload_root}"/{plans,plan_results,results,status,logs}
  write_metadata "${workload}"
  if [[ "${RESUME_RUN}" == 1 ]] && profile_failure_already_recorded "${workload}"; then
    echo "[$(date -Is)] Reusing recorded profile failure ${workload}"; continue
  fi
  if [[ "${RESUME_RUN}" == 1 ]] && profile_completed_in_reuse_run "${workload}"; then
    echo "[$(date -Is)] Reusing validated profile ${workload} from ${REUSE_PROFILE_RUN_ID}"
    FEASIBLE_WORKLOADS+=("${workload}"); continue
  fi
  if [[ "${REUSE_EXISTING_PROFILES}" == 1 ]] && \
     validate_profile_file "${PROFILE_DIR}/${workload}.yaml"; then
    echo "[$(date -Is)] Reusing validated existing profile ${workload}"
    FEASIBLE_WORKLOADS+=("${workload}"); continue
  fi
  echo "[$(date -Is)] Generating isolated formal profile ${workload}"
  profile_run_id="${RUN_ID}_${workload}_profile"
  if env CH6_RESULT_ROOT="${FORMAL_ROOT}" \
     CH6_PROFILE_ROOT="${PROFILE_DIR}" \
     CH6_PROFILE_TIMEOUT_SEC="${PROFILE_TIMEOUT_SEC}" \
     CH6_PROFILE_RUN_ID="${profile_run_id}" \
     bash "${BASE_DIR}/run_formal_profiles.sh" --workloads "${workload}" &&
     validate_profile_file "${PROFILE_DIR}/${workload}.yaml"; then
    FEASIBLE_WORKLOADS+=("${workload}")
  else
    profile_status=profile_failed
    profile_reason="isolated formal profile failed or did not pass validation"
    profile_log="${FORMAL_ROOT}/profile_runs/${profile_run_id}/profile_matrix.log"
    if [[ -f "${profile_log}" ]] && grep -Fq "PROFILE_TIMEOUT ${workload}" "${profile_log}"; then
      profile_status=profile_timeout
      profile_reason="isolated formal profile exceeded ${PROFILE_TIMEOUT_SEC}s"
    fi
    write_status "${workload}" profile "${profile_status}" "${profile_reason}"
    write_status "${workload}" candidate_failure candidate_failed "no valid shared profile; ${REPEATS} successful DP repetitions are impossible"
    echo "[$(date -Is)] PROFILE_UNAVAILABLE ${workload}; retaining it in the denominator"
  fi
done
if (( ${#FEASIBLE_WORKLOADS[@]} > 0 )); then start_ray; trap stop_ray EXIT; fi

for workload in "${FEASIBLE_WORKLOADS[@]}"; do
  workload_root="${RUN_ROOT}/${workload}"
  mkdir -p "${workload_root}/plans" "${workload_root}/plan_results" \
    "${workload_root}/results" "${workload_root}/status" \
    "${workload_root}/logs"
  write_metadata "${workload}"
  for optimizer in "${OPTIMIZERS[@]}"; do
    generate_plan "${workload}" "${optimizer}"
  done

  if [[ "${DP_SELECTED}" == "1" &&
        ! -f "${workload_root}/plans/dp_optimizer.yaml" ]]; then
    write_status "${workload}" "candidate_failure" \
      "candidate_failed" \
      "DP did not produce a valid plan; ${REPEATS} successful DP repeats are impossible"
    echo "[$(date -Is)] CANDIDATE_FAILED ${workload}: DP plan unavailable"
    continue
  fi

  declare -A execution_timed_out=()
  for ((round = 1; round <= REPEATS; round++)); do
    offset=$(((round - 1) % ${#OPTIMIZERS[@]}))
    for ((i = 0; i < ${#OPTIMIZERS[@]}; i++)); do
      optimizer="${OPTIMIZERS[$(((offset + i) % ${#OPTIMIZERS[@]}))]}"
      if [[ "${execution_timed_out[${optimizer}]:-0}" == "1" ]]; then
        write_status "${workload}" "round${round}__${optimizer}" \
          "skipped_after_timeout" \
          "earlier repeat exceeded ${EXECUTION_TIMEOUT_SEC}s"
        continue
      fi
      execute_plan "${workload}" "${optimizer}" "${round}"
      status_path="${workload_root}/status/round${round}__${optimizer}.json"
      if [[ -f "${status_path}" ]] && python - "${status_path}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    status = json.load(stream)
raise SystemExit(0 if status.get("status") == "infeasible_timeout" else 1)
PY
      then
        execution_timed_out["${optimizer}"]=1
      fi
      if [[ -f "${workload_root}/status/source_infeasible.json" ]]; then
        break 2
      fi
      if [[ "${optimizer}" == "dp_optimizer" &&
            -f "${workload_root}/status/round${round}__dp_optimizer.json" ]]; then
        write_status "${workload}" "candidate_failure" \
          "candidate_failed" \
          "DP repeat ${round} was not successful; ${REPEATS} successful DP repeats are impossible"
        echo "[$(date -Is)] CANDIDATE_FAILED ${workload}: DP round ${round}"
        break 2
      fi
    done
  done
done

if [[ "${DP_SELECTED}" == "1" ]]; then
  python "${BASE_DIR}/analyze_dp_20pct_goal.py" \
    --candidate-root "${RUN_ROOT}" \
    --expected-repeats "${REPEATS}" \
    --expected-samples "${OUTPUTS}" \
    --execution-timeout-sec "${EXECUTION_TIMEOUT_SEC}" \
    --json-output "${RUN_ROOT}/dp_20pct_report.json" \
    --markdown-output "${RUN_ROOT}/dp_20pct_report.md"
  printf '%s\n' "${RUN_ROOT}" > \
    "${FORMAL_ROOT}/datajuicer_candidates_latest_run.txt"
  cp -p "${RUN_ROOT}/dp_20pct_report.json" \
    "${FORMAL_ROOT}/datajuicer_candidates_latest.json"
  cp -p "${RUN_ROOT}/dp_20pct_report.md" \
    "${FORMAL_ROOT}/datajuicer_candidates_latest.md"
fi
echo "[$(date -Is)] Candidate run complete: ${RUN_ROOT}"
