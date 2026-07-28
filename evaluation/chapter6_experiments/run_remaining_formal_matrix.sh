#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULT_ROOT="${REPO_ROOT}/evaluation/chapter6_experiments/formal_results"
RUN_ID="${CH6_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RESULT_ROOT}/runs/${RUN_ID}"
PROFILE_ROOT="${RUN_ROOT}/profiles"
RAW_ROOT="${RUN_ROOT}/raw"
LOG_ROOT="${RUN_ROOT}/logs"
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
mkdir -p "${PROFILE_ROOT}" "${RAW_ROOT}" "${LOG_ROOT}"

exec > >(tee -a "${RUN_ROOT}/matrix.log") 2>&1

echo "[$(date -Is)] Starting remaining formal matrix run_id=${RUN_ID}"
echo "[$(date -Is)] CPU budget=${CPU_BUDGET}; profile Ray/SMP width=1"

ray stop --force || true
ray start --head --node-ip-address=127.0.0.1 --port=6379 \
  --num-cpus="${CPU_BUDGET}" --disable-usage-stats

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
actual = profile.get("resource_config") if isinstance(profile, dict) else None
if actual != expected:
    raise RuntimeError(
        f"Invalid resource_config in {path}: expected={expected}, actual={actual}"
    )
print(f"Validated profile resource signature: {path}: {actual}")
PY
}

run_case() {
  local name="$1"
  local dataset="$2"
  local samples="$3"
  local cache_mode="$4"
  local kwargs="${5:-}"
  local profile="${PROFILE_ROOT}/${name}_actors1_10s.yaml"
  local profile_log="${LOG_ROOT}/${name}_profile.log"
  local result_json="${RAW_ROOT}/${name}.json"
  local result_log="${LOG_ROOT}/${name}_formal.log"

  echo "[$(date -Is)] Profiling ${name} -> ${profile}"
  profile_args=(taskset -c 0 python evaluation/eval_cedar.py
    --dataset_file "${dataset}"
    --profiled_stats "${profile}"
    --run_profiling
    --disable_optimizer
    --disable_controller
    --disable_prefetch
    --use_ray
    --ray_ip "${RAY_ADDRESS}")
  if [[ -n "${kwargs}" ]]; then
    profile_args+=(--dataset_kwargs "${kwargs}")
  fi
  printf 'command:' > "${profile_log}"
  printf ' %q' "${profile_args[@]}" >> "${profile_log}"
  printf '\n' >> "${profile_log}"
  "${profile_args[@]}" >> "${profile_log}" 2>&1
  validate_profile "${profile}" | tee -a "${profile_log}"

  echo "[$(date -Is)] Measuring ${name} cache=${cache_mode}"
  compare_args=(python evaluation/compare_optimizer_perf.py
    --dataset_file "${dataset}"
    --profiled_stats "${profile}"
    --num_total_samples "${samples}"
    --num_epochs 1
    --num_repeats 3
    --warmup_runs 0
    --full_data_run
    --use_ray
    --ray_ip "${RAY_ADDRESS}"
    --enable_local_parallelism
    --disable_cedar_runtime_timeout
    --match_profile_resources
    --cpu_budget "${CPU_BUDGET}"
    --optimizers "${OPTIMIZERS[@]}"
    --results_path "${result_json}")
  if [[ -n "${kwargs}" ]]; then
    compare_args+=(--dataset_kwargs "${kwargs}")
  fi
  if [[ "${cache_mode}" == "off" ]]; then
    compare_args+=(--disable_caching)
  fi
  printf 'command:' > "${result_log}"
  printf ' %q' "${compare_args[@]}" >> "${result_log}"
  printf '\n' >> "${result_log}"
  "${compare_args[@]}" >> "${result_log}" 2>&1
  echo "[$(date -Is)] Completed ${name} -> ${result_json}"
}

run_case commonvoice \
  evaluation/pipelines/commonvoice/cedar_dataset.py \
  40571 off
run_case commonvoice_cache \
  evaluation/pipelines/commonvoice/cedar_cache_dataset.py \
  40571 on
run_case llava_pretrain \
  evaluation/pipelines/llava_pretrain/cedar_dataset.py \
  558128 off \
  "dataset_path=evaluation/datasets/llava_pretrain/blip_laion_cc_sbu_558k.jsonl,image_root=evaluation/datasets/llava_pretrain"
run_case redpajama_c4 \
  evaluation/pipelines/redpajama_c4/cedar_dataset.py \
  7745209 off \
  "dataset_path=datasets/redpajama_c4/redpajama-c4-refined.jsonl"
run_case simclrv2 \
  evaluation/pipelines/simclrv2/cedar_dataset.py \
  9472 off
run_case simclrv2_cache \
  evaluation/pipelines/simclrv2/cedar_cache_dataset.py \
  9472 on
run_case wikitext103 \
  evaluation/pipelines/wikitext103/cedar_dataset.py \
  1801408 off
run_case wikitext103_cache \
  evaluation/pipelines/wikitext103/cedar_cache_dataset.py \
  1801408 on

echo "[$(date -Is)] Remaining formal matrix completed run_id=${RUN_ID}"
