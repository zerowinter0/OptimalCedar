#!/usr/bin/env bash
# Build isolated direct-compute profiles/plans and execute DpOptimizer on every
# workload represented by the latest paper figure. Nothing outside RUN_ROOT is
# replaced.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_DIR="${REPO_ROOT}/evaluation/chapter6_experiments"
FORMAL_ROOT="${BASE_DIR}/formal_results"
RUN_ID="${ALL_DP_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_PARENT="${ALL_DP_RUN_PARENT:-${FORMAL_ROOT}/dp_direct_compute_runs}"
RUN_ROOT="${RUN_PARENT}/${RUN_ID}"
PROTOCOL_NAME="${ALL_DP_PROTOCOL_NAME:-all_workloads_dp_direct_compute_w8}"
PROFILE_ROOT="${RUN_ROOT}/profiles"
SOURCE_PROFILE_ROOT="${ALL_DP_SOURCE_PROFILE_ROOT:-${FORMAL_ROOT}/profiles}"
SKIP_PROFILE_REFRESH="${ALL_DP_SKIP_PROFILE_REFRESH:-0}"
RAY_ADDRESS="127.0.0.1:6379"
CPU_BUDGET=64
LOCAL_WORKERS=8
REPEATS=3
PLAN_TIMEOUT_SEC=300
EXECUTION_TIMEOUT_SEC=3600
WORKLOADS=(
  coco commonvoice commonvoice_cache llava_pretrain redpajama_c4
  stackexchange simclrv2 simclrv2_cache wikitext103 wikitext103_cache
  redpajama_code pile_europarl pile_hackernews pile_pubmed_abstracts
  pile_freelaw pile_uspto_backgrounds
)
if [[ -n "${ALL_DP_WORKLOADS:-}" ]]; then
  IFS=',' read -r -a WORKLOADS <<< "${ALL_DP_WORKLOADS}"
fi
EXTRA_INCREMENTAL="stackexchange,redpajama_code,pile_europarl,pile_hackernews,pile_pubmed_abstracts,pile_uspto_backgrounds"

[[ ! -e "${RUN_ROOT}" ]] || {
  echo "Run already exists: ${RUN_ROOT}" >&2
  exit 2
}
cd "${REPO_ROOT}"
source env/bin/activate
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TF_CPP_MIN_LOG_LEVEL=2
export CEDAR_PROFILE_RAY_ACTORS=1
export CEDAR_PROFILE_SMP_PROCS=1
export CEDAR_PROFILE_FILTER_SELECTIVITY=1
unset CEDAR_LOCAL_WORKERS
ulimit -n 65536
mkdir -p "${PROFILE_ROOT}"
exec > >(tee -a "${RUN_ROOT}/matrix.log") 2>&1

dataset_file() {
  case "$1" in
    coco) echo evaluation/pipelines/coco/cedar_dataset.py ;;
    commonvoice) echo evaluation/pipelines/commonvoice/cedar_dataset.py ;;
    commonvoice_cache) echo evaluation/pipelines/commonvoice/cedar_cache_dataset.py ;;
    llava_pretrain) echo evaluation/pipelines/llava_pretrain/cedar_dataset.py ;;
    redpajama_c4) echo evaluation/pipelines/redpajama_c4/cedar_dataset.py ;;
    stackexchange) echo evaluation/pipelines/stackexchange/cedar_dataset.py ;;
    simclrv2) echo evaluation/pipelines/simclrv2/cedar_dataset.py ;;
    simclrv2_cache) echo evaluation/pipelines/simclrv2/cedar_cache_dataset.py ;;
    wikitext103) echo evaluation/pipelines/wikitext103/cedar_dataset.py ;;
    wikitext103_cache) echo evaluation/pipelines/wikitext103/cedar_cache_dataset.py ;;
    redpajama_code) echo evaluation/pipelines/redpajama_code/cedar_dataset.py ;;
    pile_europarl) echo evaluation/pipelines/pile_europarl/cedar_dataset.py ;;
    pile_hackernews) echo evaluation/pipelines/pile_hackernews/cedar_dataset.py ;;
    pile_pubmed_abstracts) echo evaluation/pipelines/pile_pubmed_abstracts/cedar_dataset.py ;;
    pile_freelaw) echo evaluation/pipelines/pile_freelaw/cedar_dataset.py ;;
    pile_uspto_backgrounds) echo evaluation/pipelines/pile_uspto_backgrounds/cedar_dataset.py ;;
    *) return 2 ;;
  esac
}

dataset_kwargs() {
  case "$1" in
    coco) echo split=train2017 ;;
    commonvoice|commonvoice_cache) echo dataset_path=datasets/commonvoice/cv15_en_train_5shards,max_samples=160000 ;;
    llava_pretrain) echo dataset_path=evaluation/datasets/llava_pretrain/blip_laion_cc_sbu_20000_dj_fmt_only_caption.jsonl,image_root=evaluation/datasets/llava_pretrain ;;
    redpajama_c4) echo dataset_path=datasets/redpajama_c4/redpajama-c4-raw-829916.jsonl ;;
    stackexchange) echo dataset_path=datasets/stackexchange/redpajama-stackexchange-35000.jsonl ;;
    wikitext103|wikitext103_cache) echo max_samples=100000 ;;
    redpajama_code) echo dataset_path=datasets/redpajama_code/redpajama-github-raw-50000.jsonl ;;
    pile_europarl) echo dataset_path=datasets/pile_europarl/pile-europarl-raw.jsonl ;;
    pile_hackernews) echo dataset_path=datasets/pile_hackernews/pile-hackernews-raw-100000.jsonl ;;
    pile_pubmed_abstracts) echo dataset_path=datasets/pile_pubmed_abstracts/pile-pubmed-abstracts-raw-100000.jsonl ;;
    pile_freelaw) echo dataset_path=datasets/pile_freelaw/pile-freelaw-raw-100000.jsonl ;;
    pile_uspto_backgrounds) echo dataset_path=datasets/pile_uspto_backgrounds/pile-uspto-backgrounds-raw-100000.jsonl ;;
    simclrv2|simclrv2_cache) echo "" ;;
    *) return 2 ;;
  esac
}

sample_count() {
  case "$1" in
    coco) echo 50000 ;;
    commonvoice|commonvoice_cache) echo 160000 ;;
    llava_pretrain) echo 20000 ;;
    redpajama_c4) echo 20000 ;;
    stackexchange) echo 10000 ;;
    simclrv2|simclrv2_cache) echo 9469 ;;
    wikitext103|wikitext103_cache) echo 100000 ;;
    redpajama_code|pile_europarl|pile_hackernews|pile_pubmed_abstracts|pile_freelaw|pile_uspto_backgrounds) echo 20000 ;;
    *) return 2 ;;
  esac
}

cache_enabled() {
  [[ "$1" == "commonvoice_cache" || "$1" == "simclrv2_cache" || "$1" == "wikitext103_cache" ]]
}

write_status() {
  local workload="$1" name="$2" status="$3" reason="$4"
  mkdir -p "${RUN_ROOT}/${workload}/status"
  python - "${RUN_ROOT}/${workload}/status/${name}.json" "${status}" "${reason}" <<'PY'
import json, pathlib, sys
pathlib.Path(sys.argv[1]).write_text(
    json.dumps({"status": sys.argv[2], "reason": sys.argv[3]}, indent=2) + "\n"
)
PY
}

profile_valid() {
  python - "$1" <<'PY'
import sys, yaml
p=yaml.safe_load(open(sys.argv[1]))
expected={"schema_version":1,"profile_scope":"single_local_worker","profile_local_workers":1,"actors_per_stage":1,"ray_actors_per_stage":1,"smp_procs_per_stage":1}
assert p.get("resource_config")==expected
count=0
for variant in ("RAY","SMP"):
    for entry in p["offloads"][variant].values():
        assert int(entry["backend_compute"]["count"]) > 0
        count += 1
assert count > 0
PY
}

plan_valid() {
  python - "$1" <<'PY'
import sys,yaml
p=yaml.safe_load(open(sys.argv[1]))["physical_plan"]
assert p["n_local_workers"] == 8
for pipe_id in p["graph"]:
    d=p["pipes"][pipe_id]
    if d.get("variant") in ("RAY","TF_RAY"):
        assert d["variant_ctx"]["n_actors"] == 1
    if d.get("variant") == "SMP":
        assert d["variant_ctx"]["n_procs"] == 1
PY
}

echo "[$(date -Is)] Isolated all-workload DP run ${RUN_ID}"
for workload in "${WORKLOADS[@]}"; do
  source_profile="${SOURCE_PROFILE_ROOT}/${workload}.yaml"
  [[ -f "${source_profile}" ]] && cp -p "${source_profile}" "${PROFILE_ROOT}/${workload}.yaml"
done

if [[ "${SKIP_PROFILE_REFRESH}" == "0" ]]; then
  echo "[$(date -Is)] Add direct backend timings to the six legacy profiles"
  INCREMENTAL_BACKEND_COMPUTE=1 \
  CH6_PROFILE_ROOT="${PROFILE_ROOT}" \
  CH6_PROFILE_RUN_ID="${RUN_ID}_incremental" \
  bash "${BASE_DIR}/run_formal_profiles.sh" --workloads "${EXTRA_INCREMENTAL}"

  echo "[$(date -Is)] Build the previously missing FreeLaw profile in isolation"
  set +e
  CH6_PROFILE_ROOT="${PROFILE_ROOT}" \
  CH6_PROFILE_RUN_ID="${RUN_ID}_freelaw" \
  bash "${BASE_DIR}/run_formal_profiles.sh" --workloads pile_freelaw
  freelaw_profile_status=$?
  set -e
  if [[ "${freelaw_profile_status}" -ne 0 ]] || ! profile_valid "${PROFILE_ROOT}/pile_freelaw.yaml"; then
    write_status pile_freelaw profile profile_failed "isolated formal profile failed validation"
  fi
else
  echo "[$(date -Is)] Reusing frozen profiles from ${SOURCE_PROFILE_ROOT}"
fi

ray status --address="${RAY_ADDRESS}" >/dev/null 2>&1 || \
  ray start --head --node-ip-address=127.0.0.1 --port=6379 \
    --num-cpus="${CPU_BUDGET}" --disable-usage-stats >/dev/null

generate_plan() {
  local workload="$1" root="${RUN_ROOT}/$1"
  local dataset kwargs samples profile log result status
  dataset="$(dataset_file "${workload}")"
  kwargs="$(dataset_kwargs "${workload}")"
  samples="$(sample_count "${workload}")"
  profile="${PROFILE_ROOT}/${workload}.yaml"
  mkdir -p "${root}"/{plans,plan_results,results,status,logs,cache}
  {
    echo protocol="${PROTOCOL_NAME}"
    echo workload="${workload}"
    echo profile="${profile}"
    echo samples="${samples}"
    echo repeats="${REPEATS}"
    echo cpu_budget="${CPU_BUDGET}"
    echo local_workers="${LOCAL_WORKERS}"
    echo optimizer=dp_optimizer
    echo cache_enabled="$(cache_enabled "${workload}" && echo true || echo false)"
    echo git_commit="$(git rev-parse HEAD)"
    echo data_juicer_commit="$(git -C data-juicer -c safe.directory="${REPO_ROOT}/data-juicer" rev-parse HEAD)"
  } > "${root}/metadata.txt"
  if ! profile_valid "${profile}"; then
    write_status "${workload}" plan skipped "no valid direct-compute profile"
    return 0
  fi
  log="${root}/logs/plan__dp_optimizer.log"
  result="${root}/plan_results/dp_optimizer.json"
  args=(python evaluation/compare_optimizer_perf.py --dataset_file "${dataset}" --profiled_stats "${profile}" --num_total_samples "${samples}" --num_epochs 1 --num_repeats 1 --warmup_runs 0 --full_data_run --use_ray --ray_ip "${RAY_ADDRESS}" --enable_local_parallelism --match_profile_resources --cpu_budget "${CPU_BUDGET}" --fixed_local_workers_ablation "${LOCAL_WORKERS}" --disable_cedar_runtime_timeout --plan_only --optimizers dp_optimizer --results_path "${result}")
  [[ -n "${kwargs}" ]] && args+=(--dataset_kwargs "${kwargs}")
  cache_enabled "${workload}" || args+=(--disable_caching)
  rm -f /tmp/cedar_optimized_plan.yml
  echo "[$(date -Is)] PLAN ${workload}/dp_optimizer"
  set +e
  timeout --signal=TERM --kill-after=30s "${PLAN_TIMEOUT_SEC}" "${args[@]}" >"${log}" 2>&1
  status=$?
  set -e
  if [[ "${status}" -ne 0 || ! -f /tmp/cedar_optimized_plan.yml ]] || \
     ! plan_valid /tmp/cedar_optimized_plan.yml >>"${log}" 2>&1; then
    write_status "${workload}" plan optimizer_failed "plan command exit status ${status}"
    return 0
  fi
  cp -p /tmp/cedar_optimized_plan.yml "${root}/plans/dp_optimizer.yaml"
}

plan_has_cache() {
  python - "$1" <<'PY'
import sys,yaml
p=yaml.safe_load(open(sys.argv[1]))["physical_plan"]["pipes"]
raise SystemExit(0 if any("Cache" in str(x.get("name","")) for x in p.values()) else 1)
PY
}

run_guarded_result() {
  local timeout_sec="$1" result="$2" log="$3"
  shift 3
  local pid status=0 start_time="${SECONDS}"
  rm -f "${result}"
  setsid "$@" >"${log}" 2>&1 &
  pid=$!
  while kill -0 "${pid}" 2>/dev/null; do
    if [[ -s "${result}" ]]; then
      # eval_cedar writes the result only after the measured epoch. Give
      # ordinary teardown a short grace period, then prevent orphaned SMP
      # workers from blocking the remainder of the matrix.
      sleep 10
      kill -TERM -- "-${pid}" 2>/dev/null || true
      sleep 2
      kill -KILL -- "-${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
      return 0
    fi
    if (( SECONDS - start_time >= timeout_sec )); then
      kill -TERM -- "-${pid}" 2>/dev/null || true
      sleep 2
      kill -KILL -- "-${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
      return 124
    fi
    sleep 5
  done
  wait "${pid}" || status=$?
  [[ "${status}" -eq 0 && -s "${result}" ]]
}

warm_cache() {
  local workload="$1" root="${RUN_ROOT}/$1" dataset kwargs samples
  dataset="$(dataset_file "${workload}")"; kwargs="$(dataset_kwargs "${workload}")"; samples="$(sample_count "${workload}")"
  export CEDAR_CACHE_ROOT="${root}/cache" CEDAR_CACHE_NAMESPACE="${workload}__dp_optimizer"
  args=(python evaluation/eval_cedar.py --dataset_file "${dataset}" --master_feature_config "${root}/plans/dp_optimizer.yaml" --num_total_samples "$((samples+1))" --num_epochs 1 --use_ray --ray_ip "${RAY_ADDRESS}" --results_path "${root}/results/cache_warmup.json")
  [[ -n "${kwargs}" ]] && args+=(--dataset_kwargs "${kwargs}")
  echo "[$(date -Is)] WARM ${workload}/dp_optimizer"
  run_guarded_result "${EXECUTION_TIMEOUT_SEC}" \
    "${root}/results/cache_warmup.json" \
    "${root}/logs/cache_warmup.log" "${args[@]}"
  python - "${root}/cache/${workload}__dp_optimizer" "${samples}" <<'PY'
import json,pathlib,sys
paths=list(pathlib.Path(sys.argv[1]).glob("**/.manifest.json"))
assert paths, "cache warmup produced no manifest"
count=0
for path in paths:
    m=json.load(open(path)); assert m.get("complete") is True; count += int(m["num_items"])
assert count == int(sys.argv[2]), (count,sys.argv[2])
PY
}

execute_round() {
  local workload="$1" round="$2" root="${RUN_ROOT}/$1" dataset kwargs samples status result
  dataset="$(dataset_file "${workload}")"; kwargs="$(dataset_kwargs "${workload}")"; samples="$(sample_count "${workload}")"
  result="${root}/results/round${round}__dp_optimizer.json"
  args=(python evaluation/eval_cedar.py --dataset_file "${dataset}" --master_feature_config "${root}/plans/dp_optimizer.yaml" --num_total_samples "${samples}" --num_epochs 1 --use_ray --ray_ip "${RAY_ADDRESS}" --results_path "${result}")
  [[ -n "${kwargs}" ]] && args+=(--dataset_kwargs "${kwargs}")
  echo "[$(date -Is)] RUN ${workload}/round${round}__dp_optimizer"
  set +e
  run_guarded_result "${EXECUTION_TIMEOUT_SEC}" "${result}" \
    "${root}/logs/round${round}__dp_optimizer.log" "${args[@]}"
  status=$?
  set -e
  if [[ "${status}" -ne 0 || ! -f "${result}" ]]; then
    write_status "${workload}" "round${round}__dp_optimizer" execution_failed "execution command exit status ${status}"
    return 1
  fi
  if ! python - "${result}" "${samples}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1])); values=p.get("epoch_num_samples")
assert isinstance(values,list) and len(values)==1
assert int(values[0]) >= int(sys.argv[2]), (values,sys.argv[2])
PY
  then
    write_status "${workload}" "round${round}__dp_optimizer" source_exhausted "result contains fewer than ${samples} samples"
    return 1
  fi
  echo "[$(date -Is)] DONE ${workload}/round${round}__dp_optimizer"
}

for workload in "${WORKLOADS[@]}"; do generate_plan "${workload}"; done

for workload in "${WORKLOADS[@]}"; do
  root="${RUN_ROOT}/${workload}"
  plan="${root}/plans/dp_optimizer.yaml"
  [[ -f "${plan}" ]] || continue
  unset CEDAR_CACHE_ROOT CEDAR_CACHE_NAMESPACE CEDAR_CACHE_SHARD
  if cache_enabled "${workload}" && plan_has_cache "${plan}"; then warm_cache "${workload}"; fi
  for round in $(seq 1 "${REPEATS}"); do
    execute_round "${workload}" "${round}" || break
  done
done

unset CEDAR_CACHE_ROOT CEDAR_CACHE_NAMESPACE CEDAR_CACHE_SHARD
echo "[$(date -Is)] All-workload isolated DP matrix complete: ${RUN_ROOT}"
