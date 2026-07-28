#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULT_ROOT="${REPO_ROOT}/evaluation/chapter6_experiments/formal_results"
PROFILE_ROOT="${RESULT_ROOT}/profiles"
PLAN_ROOT="${RESULT_ROOT}/plans"
RUN_ID="${CH6_PLAN_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RESULT_ROOT}/plan_runs/${RUN_ID}"
LOG_ROOT="${RUN_ROOT}/logs"
JSON_ROOT="${RUN_ROOT}/results"
TMP_BACKUP_ROOT="${RUN_ROOT}/tmp_plan_backups"
SUMMARY="${RUN_ROOT}/plan_resources.tsv"
RAY_ADDRESS="127.0.0.1:6379"
CPU_BUDGET=64
OPTIMIZERS=(optimizer dj_optimizer dp_cedar_optimizer dp_optimizer dp_two_stage_optimizer)

cd "${REPO_ROOT}"
# shellcheck source=/dev/null
source env/bin/activate
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TF_CPP_MIN_LOG_LEVEL=2
export CEDAR_PROFILE_RAY_ACTORS=1
export CEDAR_PROFILE_SMP_PROCS=1
unset CEDAR_LOCAL_WORKERS
ulimit -n 65536

mkdir -p "${PLAN_ROOT}" "${LOG_ROOT}" "${JSON_ROOT}" "${TMP_BACKUP_ROOT}"
exec > >(tee -a "${RUN_ROOT}/plan_matrix.log") 2>&1
printf 'workload\toptimizer\tlocal_workers\tray_stages\tsmp_stages\tper_worker_cpus\ttotal_accounted_cpus\tplan\n' > "${SUMMARY}"

validate_profile() {
  local profile="$1"
  python - "${profile}" <<'PY'
import sys
import yaml

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as stream:
    profile = yaml.safe_load(stream)
expected = {
    "schema_version": 1,
    "profile_scope": "single_local_worker",
    "profile_local_workers": 1,
    "actors_per_stage": 1,
    "ray_actors_per_stage": 1,
    "smp_procs_per_stage": 1,
}
if not isinstance(profile, dict) or profile.get("resource_config") != expected:
    raise RuntimeError(f"Invalid formal profile resource signature: {path}")
for key in ("baseline", "disk_info", "offloads"):
    if key not in profile:
        raise RuntimeError(f"Profile {path} is missing {key}")
for variant in ("RAY", "SMP"):
    if variant not in profile["offloads"]:
        raise RuntimeError(f"Profile {path} is missing offloads.{variant}")
PY
}

validate_plan() {
  local workload="$1"
  local optimizer="$2"
  local plan="$3"
  python - "${workload}" "${optimizer}" "${plan}" "${CPU_BUDGET}" <<'PY'
import sys
import yaml

workload, optimizer, path, budget_raw = sys.argv[1:]
budget = int(budget_raw)
with open(path, "r", encoding="utf-8") as stream:
    document = yaml.safe_load(stream)
physical = document.get("physical_plan") if isinstance(document, dict) else None
if not isinstance(physical, dict):
    raise RuntimeError(f"Missing physical_plan in {path}")
graph = physical.get("graph")
pipes = physical.get("pipes")
if not isinstance(graph, dict) or not isinstance(pipes, dict):
    raise RuntimeError(f"Invalid graph/pipes in {path}")

def desc(pipe_id):
    if pipe_id in pipes:
        return pipes[pipe_id]
    if str(pipe_id) in pipes:
        return pipes[str(pipe_id)]
    raise RuntimeError(f"Active pipe {pipe_id} has no descriptor in {path}")

active = [desc(pipe_id) for pipe_id in graph]
ray = [item for item in active if item.get("variant") in ("RAY", "TF_RAY")]
smp = [item for item in active if item.get("variant") == "SMP"]
ray_ds = [item for item in active if item.get("variant") == "RAY_DS"]
if ray_ds:
    raise RuntimeError(f"Active RAY_DS stage is forbidden in strict formal plan: {path}")
for item in ray:
    if item.get("variant_ctx", {}).get("n_actors") != 1:
        raise RuntimeError(f"Ray stage width is not 1 in {path}: {item}")
for item in smp:
    if item.get("variant_ctx", {}).get("n_procs") != 1:
        raise RuntimeError(f"SMP stage width is not 1 in {path}: {item}")
per_worker = 1 + len(ray) + len(smp)
expected_workers = budget // per_worker
workers = physical.get("n_local_workers")
if workers != expected_workers:
    raise RuntimeError(
        f"Wrong local worker count in {path}: expected={expected_workers}, actual={workers}"
    )
total = workers * per_worker
if total > budget:
    raise RuntimeError(f"CPU budget exceeded in {path}: total={total}, budget={budget}")
print(
    f"{workload}\t{optimizer}\t{workers}\t{len(ray)}\t{len(smp)}\t"
    f"{per_worker}\t{total}\t{path}"
)
PY
}

generate_plan() {
  local workload="$1"
  local dataset="$2"
  local samples="$3"
  local cache_mode="$4"
  local optimizer="$5"
  local kwargs="${6:-}"
  local profile="${PROFILE_ROOT}/${workload}.yaml"
  local plan_dir="${PLAN_ROOT}/${workload}"
  local plan="${plan_dir}/${optimizer}.yaml"
  local log="${LOG_ROOT}/${workload}__${optimizer}.log"
  local result_json="${JSON_ROOT}/${workload}__${optimizer}.json"
  local tmp_plan="/tmp/cedar_optimized_plan.yml"
  local backup="${TMP_BACKUP_ROOT}/${workload}__${optimizer}__previous.yaml"
  local -a args=(
    python evaluation/compare_optimizer_perf.py
    --dataset_file "${dataset}"
    --profiled_stats "${profile}"
    --num_total_samples "${samples}"
    --num_epochs 1
    --num_repeats 1
    --warmup_runs 0
    --use_ray
    --ray_ip "${RAY_ADDRESS}"
    --enable_local_parallelism
    --disable_cedar_runtime_timeout
    --match_profile_resources
    --cpu_budget "${CPU_BUDGET}"
    --plan_only
    --optimizers "${optimizer}"
    --results_path "${result_json}"
  )
  if [[ "${cache_mode}" == "off" ]]; then
    args+=(--disable_caching)
  fi
  if [[ -n "${kwargs}" ]]; then
    args+=(--dataset_kwargs "${kwargs}")
  fi

  mkdir -p "${plan_dir}"
  if [[ -f "${tmp_plan}" ]]; then
    mv -f "${tmp_plan}" "${backup}"
  fi
  echo "[$(date -Is)] Generating ${workload}/${optimizer}"
  printf 'command:' > "${log}"
  printf ' %q' "${args[@]}" >> "${log}"
  printf '\n' >> "${log}"
  if ! "${args[@]}" >> "${log}" 2>&1; then
    echo "[$(date -Is)] FAILED ${workload}/${optimizer}: generator failed; see ${log}"
    return 1
  fi
  if [[ ! -f "${tmp_plan}" ]]; then
    echo "[$(date -Is)] FAILED ${workload}/${optimizer}: no plan was produced"
    return 1
  fi
  cp -p "${tmp_plan}" "${plan}"
  local row
  if ! row="$(validate_plan "${workload}" "${optimizer}" "${plan}")"; then
    echo "[$(date -Is)] FAILED ${workload}/${optimizer}: plan validation failed"
    return 1
  fi
  printf '%s\n' "${row}" | tee -a "${SUMMARY}"
  echo "[$(date -Is)] Completed ${workload}/${optimizer}"
}

for workload in coco commonvoice commonvoice_cache llava_pretrain redpajama_c4 simclrv2 simclrv2_cache wikitext103 wikitext103_cache; do
  validate_profile "${PROFILE_ROOT}/${workload}.yaml" || exit 1
done

echo "[$(date -Is)] Starting 45-plan formal matrix run ${RUN_ID}"
ray stop --force || true
if ! ray start --head --node-ip-address=127.0.0.1 --port=6379 \
  --num-cpus="${CPU_BUDGET}" --disable-usage-stats; then
  echo "[$(date -Is)] FAILED to start Ray"
  exit 1
fi

failures=()
run_workload() {
  local name="$1" dataset="$2" samples="$3" cache_mode="$4" kwargs="${5:-}"
  local optimizer
  for optimizer in "${OPTIMIZERS[@]}"; do
    generate_plan "${name}" "${dataset}" "${samples}" "${cache_mode}" "${optimizer}" "${kwargs}" || \
      failures+=("${name}/${optimizer}")
  done
}

run_workload coco evaluation/pipelines/coco/cedar_dataset.py 118287 off
run_workload commonvoice evaluation/pipelines/commonvoice/cedar_dataset.py 40571 off
run_workload commonvoice_cache evaluation/pipelines/commonvoice/cedar_cache_dataset.py 40571 on
run_workload llava_pretrain evaluation/pipelines/llava_pretrain/cedar_dataset.py 558128 off \
  "dataset_path=evaluation/datasets/llava_pretrain/blip_laion_cc_sbu_558k.jsonl,image_root=evaluation/datasets/llava_pretrain"
run_workload redpajama_c4 evaluation/pipelines/redpajama_c4/cedar_dataset.py 7745209 off \
  "dataset_path=datasets/redpajama_c4/redpajama-c4-refined.jsonl"
run_workload simclrv2 evaluation/pipelines/simclrv2/cedar_dataset.py 9472 off
run_workload simclrv2_cache evaluation/pipelines/simclrv2/cedar_cache_dataset.py 9472 on
run_workload wikitext103 evaluation/pipelines/wikitext103/cedar_dataset.py 1801408 off
run_workload wikitext103_cache evaluation/pipelines/wikitext103/cedar_cache_dataset.py 1801408 on

ray stop --force || true
if (( ${#failures[@]} )); then
  printf '[%s] Plan generation completed with failures:' "$(date -Is)"
  printf ' %s' "${failures[@]}"
  printf '\n'
  exit 1
fi
echo "[$(date -Is)] All 45 formal plans completed and validated"
