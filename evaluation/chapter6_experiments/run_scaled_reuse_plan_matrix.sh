#!/usr/bin/env bash
# Enlarged W=8 matrix. Reuse every persisted profile/plan, materialize only
# plans missing from the old artifact set, and isolate every cell with a
# one-hour wall-clock limit.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST_REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONTAINER_REPO="/workspace/OptimalCedar"
FORMAL_DIR="${HOST_REPO}/evaluation/chapter6_experiments/formal_results"
CEDAR_CONTAINER="${CEDAR_CONTAINER:-optimalcedar-torch201-dev}"
PLUMBER_CONTAINER="${PLUMBER_CONTAINER:-optimalcedar-plumber}"
FASTFLOW_CONTAINER="${FASTFLOW_CONTAINER:-optimalcedar-fastflow}"
RAY_ADDRESS="${RAY_ADDRESS:-127.0.0.1:6379}"
CPU_BUDGET=64
LOCAL_WORKERS=8
OPTIMIZER_TIMEOUT_SEC=3600
CELL_TIMEOUT_SEC=3600
REPEATS=3
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RESUME="${RESUME:-0}"
FORCE_REGENERATE_PLANS="${FORCE_REGENERATE_PLANS:-0}"
SELECTED_WORKLOADS="all"

WORKLOADS=(
  coco commonvoice commonvoice_cache llava_pretrain redpajama_c4
  stackexchange simclrv2 simclrv2_cache wikitext103 wikitext103_cache
  pile_europarl redpajama_code pile_hackernews pile_pubmed_abstracts
  pile_freelaw pile_uspto_backgrounds
)
OPTIMIZERS=(optimizer dj_optimizer dp_cedar_optimizer dp_optimizer pecan_optimizer)
SYSTEMS=(pytorch tensorflow ray plumber fastflow)

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --workloads) SELECTED_WORKLOADS="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    -h|--help)
      echo "Usage: $0 [--workloads a,b] [--run-id ID] [--resume]"
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]] || exit 2
RUN_REL="evaluation/chapter6_experiments/formal_results/scaled_reuse_plan_runs/${RUN_ID}"
RUN_ROOT="${HOST_REPO}/${RUN_REL}"
CONTAINER_RUN_ROOT="${CONTAINER_REPO}/${RUN_REL}"
if [[ -e "${RUN_ROOT}" && "${RESUME}" != 1 ]]; then
  echo "Run exists; use --resume: ${RUN_ROOT}" >&2
  exit 2
fi
mkdir -p "${RUN_ROOT}"/{plans,cedar,systems,status,logs,cache}
printf '%s\n' "${RUN_ROOT}" > "${FORMAL_DIR}/scaled_reuse_plan_latest_run.txt"

exec > >(tee -a "${RUN_ROOT}/matrix.log") 2>&1

selected() {
  [[ "${SELECTED_WORKLOADS}" == all ]] && return 0
  [[ ",${SELECTED_WORKLOADS}," == *",$1,"* ]]
}

status_file() { printf '%s/status/%s/%s/%s.tsv' "${RUN_ROOT}" "$1" "$2" "$3"; }
write_status() {
  local path
  path="$(status_file "$1" "$2" "$3")"
  mkdir -p "$(dirname "${path}")"
  printf '%s\t%s\n' "$4" "${5//$'\n'/ }" > "${path}"
}
recorded() { [[ "${RESUME}" == 1 && -s "$(status_file "$1" "$2" "$3")" ]]; }
previous_timeout() {
  local workload="$1" entity="$2" attempt="$3" round="$4" prior state
  for ((prior=1; prior<round; prior++)); do
    state="$(cut -f1 "$(status_file "${workload}" "${entity}" "attempt${attempt}__round${prior}")" 2>/dev/null || true)"
    [[ "${state}" == execution_timeout ]] && return 0
  done
  return 1
}

cedar_exec() {
  docker exec -i -e CPU_BUDGET=64 -e TF_CPP_MIN_LOG_LEVEL=2 \
    -e PYTHONWARNINGS=ignore "${CEDAR_CONTAINER}" bash -lc '
      cd /workspace/OptimalCedar
      source env/bin/activate
      export PYTHONPATH=/workspace/OptimalCedar:${PYTHONPATH:-}
      ulimit -n 65536
      exec "$@"
    ' bash "$@"
}

cedar_timed() {
  docker exec -i -e CPU_BUDGET=64 -e TF_CPP_MIN_LOG_LEVEL=2 \
    -e PYTHONWARNINGS=ignore -e CELL_TIMEOUT_SEC=3600 \
    "${CEDAR_CONTAINER}" bash -lc '
      cd /workspace/OptimalCedar
      source env/bin/activate
      export PYTHONPATH=/workspace/OptimalCedar:${PYTHONPATH:-}
      ulimit -n 65536
      exec timeout --signal=TERM --kill-after=30s "${CELL_TIMEOUT_SEC}" "$@"
    ' bash "$@"
}

restart_ray() {
  cedar_exec ray stop --force >/dev/null 2>&1 || true
  cedar_exec ray start --head --node-ip-address=127.0.0.1 --port=6379 \
    --num-cpus=64 --disable-usage-stats >/dev/null
}

set_config() {
  local workload="$1"
  DATASET="evaluation/pipelines/${workload%_cache}/cedar_dataset.py"
  PROFILE="evaluation/chapter6_experiments/formal_results/profiles/${workload}.yaml"
  KWARGS=""
  NATIVE_KWARGS='{}'
  NATIVE_PATH=""
  NATIVE_IMAGE_ROOT=""
  CEDAR_DATASET_PATH=""
  CACHE_MODE=off
  TARGET=20000
  FLOOR=1000
  PRIOR_SAMPLES=20000
  PLAN_SOURCE="evaluation/chapter6_experiments/formal_results/plans/${workload}"
  case "${workload}" in
    coco)
      TARGET=50000; FLOOR=12500; PRIOR_SAMPLES=5000
      DATASET=evaluation/pipelines/coco/cedar_dataset.py
      KWARGS=split=train2017
      NATIVE_PATH=datasets/coco
      NATIVE_KWARGS='{"split":"train2017"}' ;;
    commonvoice|commonvoice_cache)
      TARGET=160000; FLOOR=40000; PRIOR_SAMPLES=10000
      CEDAR_DATASET_PATH=datasets/commonvoice/cv15_en_train_5shards
      NATIVE_PATH="${CEDAR_DATASET_PATH}"
      KWARGS="dataset_path=${CEDAR_DATASET_PATH},max_samples=${TARGET}"
      [[ "${workload}" == *_cache ]] && { CACHE_MODE=on; DATASET=evaluation/pipelines/commonvoice/cedar_cache_dataset.py; }
      ;;
    llava_pretrain)
      TARGET=20000; FLOOR=5000; PRIOR_SAMPLES=5000
      NATIVE_PATH=evaluation/datasets/llava_pretrain/blip_laion_cc_sbu_20000_dj_fmt_only_caption.jsonl
      NATIVE_IMAGE_ROOT=evaluation/datasets/llava_pretrain
      KWARGS="dataset_path=${NATIVE_PATH},image_root=${NATIVE_IMAGE_ROOT}" ;;
    redpajama_c4)
      TARGET=100000; FLOOR=20000; PRIOR_SAMPLES=20000
      NATIVE_PATH=datasets/redpajama_c4/redpajama-c4-raw-829916.jsonl
      KWARGS="dataset_path=${NATIVE_PATH}" ;;
    stackexchange)
      TARGET=20000; FLOOR=10000; PRIOR_SAMPLES=10000
      NATIVE_PATH=datasets/stackexchange/redpajama-stackexchange-35000.jsonl
      KWARGS="dataset_path=${NATIVE_PATH}"
      PLAN_SOURCE=evaluation/chapter6_experiments/stackexchange/plans ;;
    simclrv2|simclrv2_cache)
      TARGET=9469; FLOOR=9469; PRIOR_SAMPLES=9469
      [[ "${workload}" == *_cache ]] && { CACHE_MODE=on; DATASET=evaluation/pipelines/simclrv2/cedar_cache_dataset.py; }
      ;;
    wikitext103|wikitext103_cache)
      TARGET=500000; FLOOR=100000; PRIOR_SAMPLES=100000
      [[ "${workload}" == *_cache ]] && { CACHE_MODE=on; DATASET=evaluation/pipelines/wikitext103/cedar_cache_dataset.py; }
      ;;
    pile_europarl)
      NATIVE_PATH=datasets/pile_europarl/pile-europarl-raw.jsonl
      KWARGS="dataset_path=${NATIVE_PATH}"
      PLAN_SOURCE=evaluation/chapter6_experiments/formal_results/datajuicer_candidate_runs/20260725T051352Z/pile_europarl/plans ;;
    redpajama_code)
      NATIVE_PATH=datasets/redpajama_code/redpajama-github-raw-50000.jsonl
      KWARGS="dataset_path=${NATIVE_PATH}"
      PLAN_SOURCE=evaluation/chapter6_experiments/formal_results/datajuicer_candidate_runs/20260725T072333Z_code_selectivity/redpajama_code/plans ;;
    pile_hackernews)
      NATIVE_PATH=datasets/pile_hackernews/pile-hackernews-raw-100000.jsonl
      KWARGS="dataset_path=${NATIVE_PATH}"
      PLAN_SOURCE=evaluation/chapter6_experiments/formal_results/datajuicer_candidate_runs/20260725T072333Z_selectivity/pile_hackernews/plans ;;
    pile_pubmed_abstracts)
      NATIVE_PATH=datasets/pile_pubmed_abstracts/pile-pubmed-abstracts-raw-100000.jsonl
      KWARGS="dataset_path=${NATIVE_PATH}"
      PLAN_SOURCE=evaluation/chapter6_experiments/formal_results/datajuicer_candidate_runs/20260725T072333Z_selectivity/pile_pubmed_abstracts/plans ;;
    pile_freelaw)
      NATIVE_PATH=datasets/pile_freelaw/pile-freelaw-raw-100000.jsonl
      KWARGS="dataset_path=${NATIVE_PATH}" ;;
    pile_uspto_backgrounds)
      NATIVE_PATH=datasets/pile_uspto_backgrounds/pile-uspto-backgrounds-raw-100000.jsonl
      KWARGS="dataset_path=${NATIVE_PATH}"
      PLAN_SOURCE=evaluation/chapter6_experiments/formal_results/datajuicer_candidate_runs/20260725T072333Z_selectivity/pile_uspto_backgrounds/plans ;;
  esac
}

effective_kwargs() {
  local samples="$1"
  case "${CURRENT_WORKLOAD}" in
    commonvoice|commonvoice_cache)
      printf 'dataset_path=%s,max_samples=%s' "${CEDAR_DATASET_PATH}" "${samples}" ;;
    wikitext103|wikitext103_cache)
      printf 'max_samples=%s' "${samples}" ;;
    *) printf '%s' "${KWARGS}" ;;
  esac
}

plan_is_strict_w8() {
  awk '
    $1 == "n_local_workers:" { seen=1; if ($2 != 8) bad=1 }
    END { exit !(seen && !bad) }
  ' "$1"
}

ensure_plan() {
  local workload="$1" optimizer="$2"
  local destination="${RUN_ROOT}/plans/${workload}/${optimizer}.yaml"
  local source="${HOST_REPO}/${PLAN_SOURCE}/${optimizer}.yaml"
  mkdir -p "$(dirname "${destination}")"
  if [[ -f "${destination}" ]] && plan_is_strict_w8 "${destination}"; then
    return 0
  fi
  if [[ -f "${destination%.yaml}.source.tsv" ]] && \
     grep -q '^unavailable' "${destination%.yaml}.source.tsv"; then
    return 1
  fi
  if [[ "${FORCE_REGENERATE_PLANS}" != 1 && -f "${source}" ]] && \
     plan_is_strict_w8 "${source}"; then
    cp -p "${source}" "${destination}"
    printf 'reused\t%s\n' "${source#${HOST_REPO}/}" > "${destination%.yaml}.source.tsv"
    return 0
  fi
  if [[ ! -f "${HOST_REPO}/${PROFILE}" ]]; then
    printf 'unavailable\tmissing persisted profile; profile regeneration prohibited\n' \
      > "${destination%.yaml}.source.tsv"
    return 1
  fi

  local log="${RUN_ROOT}/logs/${workload}__plan__${optimizer}.log"
  local result="${CONTAINER_RUN_ROOT}/plans/${workload}/${optimizer}.json"
  local tmp=/tmp/cedar_optimized_plan.yml
  local -a args=(python evaluation/compare_optimizer_perf.py
    --dataset_file "${DATASET}" --profiled_stats "${PROFILE}"
    --num_total_samples "${PRIOR_SAMPLES}" --num_epochs 1 --num_repeats 1
    --warmup_runs 0 --full_data_run --use_ray --ray_ip "${RAY_ADDRESS}"
    --enable_local_parallelism --match_profile_resources --cpu_budget 64
    --fixed_local_workers_ablation 8 --disable_cedar_runtime_timeout
    --plan_only --optimizers "${optimizer}" --results_path "${result}")
  [[ "${CACHE_MODE}" == off ]] && args+=(--disable_caching)
  [[ -n "${KWARGS}" ]] && args+=(--dataset_kwargs "${KWARGS}")
  [[ "${optimizer}" == optimizer ]] && args+=(--cedar_reorder_timeout_sec 3600)
  echo "[$(date -Is)] MATERIALIZE-MISSING-PLAN ${workload}/${optimizer}"
  restart_ray
  cedar_exec rm -f "${tmp}"
  local start end status
  start="$(date +%s%N)"
  cedar_timed "${args[@]}" > "${log}" 2>&1
  status=$?
  end="$(date +%s%N)"
  awk -v a="${start}" -v b="${end}" 'BEGIN {print (b-a)/1e9}' \
    > "${destination%.yaml}.wall_sec"
  if [[ "${status}" -eq 0 ]] && docker exec "${CEDAR_CONTAINER}" test -f "${tmp}"; then
    docker cp "${CEDAR_CONTAINER}:${tmp}" "${destination}"
    if ! plan_is_strict_w8 "${destination}"; then
      printf 'unavailable\tgenerated plan violated local_workers=8\n' > "${destination%.yaml}.source.tsv"
      return 1
    fi
    printf 'generated_strict_w8\toptimizer_timeout_sec=3600\n' > "${destination%.yaml}.source.tsv"
    return 0
  fi
  printf 'unavailable\tplan generation status=%s\n' "${status}" > "${destination%.yaml}.source.tsv"
  return 1
}

prepare_cache() {
  local workload="$1" optimizer="$2" attempt="$3" samples="$4"
  [[ "${CACHE_MODE}" == on ]] || return 0
  local marker="${RUN_ROOT}/status/cache_ready/${workload}/${optimizer}/attempt${attempt}.ready"
  [[ -f "${marker}" ]] && return 0
  local namespace="${workload}__${optimizer}__attempt${attempt}"
  local cache_root="${CONTAINER_RUN_ROOT}/cache/${workload}/${optimizer}/attempt${attempt}"
  local kwargs; kwargs="$(effective_kwargs "${samples}")"
  local log="${RUN_ROOT}/logs/${workload}__${optimizer}__attempt${attempt}__cache_warmup.log"
  local -a args=(python evaluation/eval_cedar.py --dataset_file "${DATASET}"
    --master_feature_config "${CONTAINER_RUN_ROOT}/plans/${workload}/${optimizer}.yaml"
    --num_total_samples "$((samples + 1))" --num_epochs 1 --use_ray
    --ray_ip "${RAY_ADDRESS}" --results_path "${cache_root}/warmup.json")
  [[ -n "${kwargs}" ]] && args+=(--dataset_kwargs "${kwargs}")
  restart_ray
  docker exec -i -e CEDAR_CACHE_ROOT="${cache_root}" \
    -e CEDAR_CACHE_NAMESPACE="${namespace}" "${CEDAR_CONTAINER}" bash -lc '
      cd /workspace/OptimalCedar && source env/bin/activate
      export PYTHONPATH=/workspace/OptimalCedar:${PYTHONPATH:-}
      exec timeout --signal=TERM --kill-after=30s 3600 "$@"
    ' bash "${args[@]}" > "${log}" 2>&1
  local status=$?
  if [[ "${status}" -eq 0 ]]; then
    mkdir -p "$(dirname "${marker}")"; touch "${marker}"; return 0
  fi
  return "${status}"
}

run_cedar_cell() {
  local workload="$1" optimizer="$2" attempt="$3" round="$4" samples="$5"
  local tag="attempt${attempt}__round${round}"
  recorded "${workload}" "${optimizer}" "${tag}" && return 0
  if previous_timeout "${workload}" "${optimizer}" "${attempt}" "${round}"; then
    write_status "${workload}" "${optimizer}" "${tag}" skipped_after_timeout "earlier round exceeded 3600s"
    return 0
  fi
  if ! ensure_plan "${workload}" "${optimizer}"; then
    write_status "${workload}" "${optimizer}" "${tag}" unavailable "no reusable plan"
    return 0
  fi
  if ! prepare_cache "${workload}" "${optimizer}" "${attempt}" "${samples}"; then
    write_status "${workload}" "${optimizer}" "${tag}" execution_timeout "cache warmup exceeded 3600s or failed"
    return 0
  fi
  local out="${CONTAINER_RUN_ROOT}/cedar/${workload}/${optimizer}/${tag}.json"
  local host_out="${RUN_ROOT}/cedar/${workload}/${optimizer}/${tag}.json"
  local log="${RUN_ROOT}/logs/${workload}__${optimizer}__${tag}.log"
  local kwargs; kwargs="$(effective_kwargs "${samples}")"
  mkdir -p "$(dirname "${host_out}")"
  local -a args=(python evaluation/eval_cedar.py --dataset_file "${DATASET}"
    --master_feature_config "${CONTAINER_RUN_ROOT}/plans/${workload}/${optimizer}.yaml"
    --num_total_samples "${samples}" --num_epochs 1 --use_ray
    --ray_ip "${RAY_ADDRESS}" --results_path "${out}")
  [[ -n "${kwargs}" ]] && args+=(--dataset_kwargs "${kwargs}")
  echo "[$(date -Is)] RUN ${workload}/${optimizer}/${tag} samples=${samples}"
  restart_ray
  if [[ "${CACHE_MODE}" == on ]]; then
    local cache_root="${CONTAINER_RUN_ROOT}/cache/${workload}/${optimizer}/attempt${attempt}"
    local namespace="${workload}__${optimizer}__attempt${attempt}"
    docker exec -i -e CEDAR_CACHE_ROOT="${cache_root}" \
      -e CEDAR_CACHE_NAMESPACE="${namespace}" "${CEDAR_CONTAINER}" bash -lc '
        cd /workspace/OptimalCedar && source env/bin/activate
        export PYTHONPATH=/workspace/OptimalCedar:${PYTHONPATH:-}
        exec timeout --signal=TERM --kill-after=30s 3600 "$@"
      ' bash "${args[@]}" > "${log}" 2>&1
  else
    cedar_timed "${args[@]}" > "${log}" 2>&1
  fi
  local status=$?
  if [[ "${status}" -eq 124 || "${status}" -eq 137 ]]; then
    write_status "${workload}" "${optimizer}" "${tag}" execution_timeout "execution exceeded 3600s"
  elif [[ "${status}" -eq 0 && -s "${host_out}" ]]; then
    write_status "${workload}" "${optimizer}" "${tag}" success ""
  else
    write_status "${workload}" "${optimizer}" "${tag}" failed "exit_status=${status}"
  fi
}

timeout_count() {
  local workload="$1" attempt="$2" round_limit="$3" count=0 optimizer round state
  for optimizer in "${OPTIMIZERS[@]}"; do
    for ((round=1; round<=round_limit; round++)); do
      state="$(cut -f1 "$(status_file "${workload}" "${optimizer}" "attempt${attempt}__round${round}")" 2>/dev/null || true)"
      if [[ "${state}" == execution_timeout ]]; then count=$((count + 1)); break; fi
    done
  done
  printf '%s' "${count}"
}

run_no_optimizer() {
  local workload="$1" attempt="$2" samples="$3" round tag out host_out log kwargs status
  kwargs="$(effective_kwargs "${samples}")"
  for ((round=1; round<=REPEATS; round++)); do
    tag="attempt${attempt}__round${round}"
    recorded "${workload}" no_optimizer "${tag}" && continue
    if previous_timeout "${workload}" no_optimizer "${attempt}" "${round}"; then
      write_status "${workload}" no_optimizer "${tag}" skipped_after_timeout "earlier round exceeded 3600s"
      continue
    fi
    out="${CONTAINER_RUN_ROOT}/cedar/${workload}/no_optimizer/${tag}.json"
    host_out="${RUN_ROOT}/cedar/${workload}/no_optimizer/${tag}.json"
    log="${RUN_ROOT}/logs/${workload}__no_optimizer__${tag}.log"
    mkdir -p "$(dirname "${host_out}")"
    args=(python evaluation/eval_cedar.py --dataset_file "${DATASET}"
      --num_total_samples "${samples}" --num_epochs 1 --disable_optimizer
      --disable_controller --results_path "${out}")
    [[ -n "${kwargs}" ]] && args+=(--dataset_kwargs "${kwargs}")
    cedar_timed "${args[@]}" > "${log}" 2>&1; status=$?
    if [[ "${status}" -eq 124 || "${status}" -eq 137 ]]; then
      write_status "${workload}" no_optimizer "${tag}" execution_timeout "execution exceeded 3600s"
    elif [[ "${status}" -eq 0 && -s "${host_out}" ]]; then
      write_status "${workload}" no_optimizer "${tag}" success ""
    else
      write_status "${workload}" no_optimizer "${tag}" failed "exit_status=${status}"
    fi
  done
}

run_native_cell() {
  local workload="$1" system="$2" attempt="$3" round="$4" samples="$5"
  local tag="attempt${attempt}__round${round}"
  recorded "${workload}" "${system}" "${tag}" && return
  if previous_timeout "${workload}" "${system}" "${attempt}" "${round}"; then
    write_status "${workload}" "${system}" "${tag}" skipped_after_timeout "earlier round exceeded 3600s"
    return
  fi
  local entry
  entry="$(cedar_exec python -c 'from evaluation.baselines.registry import get_entry; import sys; e=get_entry(sys.argv[1],sys.argv[2]); print(e.status, e.reason or "", sep="\t")' "${system}" "${workload}")"
  if [[ "${entry%%$'\t'*}" != supported ]]; then
    write_status "${workload}" "${system}" "${tag}" unsupported "${entry#*$'\t'}"
    return
  fi
  local out="${CONTAINER_RUN_ROOT}/systems/${workload}/${system}/${tag}.json"
  local host_out="${RUN_ROOT}/systems/${workload}/${system}/${tag}.json"
  local log="${RUN_ROOT}/logs/${workload}__${system}__${tag}.log"
  mkdir -p "$(dirname "${host_out}")"
  local -a args=(python -m evaluation.baselines.run --system "${system}"
    --workload "${workload}" --batch-size 1 --workers 8 --epochs 1
    --num-samples "${samples}" --cache-dir "${CONTAINER_RUN_ROOT}/cache/native"
    --ray-address "${RAY_ADDRESS}" --results-path "${out}")
  args+=(--dataset-kwargs "${NATIVE_KWARGS}")
  [[ -n "${NATIVE_PATH}" ]] && args+=(--dataset-path "${NATIVE_PATH}")
  [[ -n "${NATIVE_IMAGE_ROOT}" ]] && args+=(--image-root "${NATIVE_IMAGE_ROOT}")
  [[ "${system}" == ray ]] && restart_ray
  echo "[$(date -Is)] RUN ${workload}/${system}/${tag} samples=${samples}"
  cedar_timed "${args[@]}" > "${log}" 2>&1
  local status=$?
  if [[ "${status}" -eq 124 || "${status}" -eq 137 ]]; then
    write_status "${workload}" "${system}" "${tag}" execution_timeout "execution exceeded 3600s"
  elif [[ "${status}" -eq 0 && -s "${host_out}" ]]; then
    write_status "${workload}" "${system}" "${tag}" success ""
  else
    write_status "${workload}" "${system}" "${tag}" failed "exit_status=${status}"
  fi
}

container_running() {
  docker inspect "$1" >/dev/null 2>&1 && \
    [[ "$(docker inspect -f '{{.State.Running}}' "$1")" == true ]]
}

run_plumber_cell() {
  local workload="$1" attempt="$2" round="$3" samples="$4"
  local tag="attempt${attempt}__round${round}"
  recorded "${workload}" plumber "${tag}" && return
  if previous_timeout "${workload}" plumber "${attempt}" "${round}"; then
    write_status "${workload}" plumber "${tag}" skipped_after_timeout "earlier round exceeded 3600s"; return
  fi
  local entry
  entry="$(cedar_exec python -c 'from evaluation.baselines.registry import get_entry; import sys; e=get_entry("plumber",sys.argv[1]); print(e.status, e.reason or "", sep="\t")' "${workload}")"
  if [[ "${entry%%$'\t'*}" != supported ]]; then
    write_status "${workload}" plumber "${tag}" unsupported "${entry#*$'\t'}"; return
  fi
  if ! container_running "${PLUMBER_CONTAINER}"; then
    write_status "${workload}" plumber "${tag}" environment_unavailable "Plumber container is not running"; return
  fi
  local base="${workload%_cache}" kwargs='{}' dataset=''
  case "${base}" in
    coco) dataset=/workspace/OptimalCedar/datasets/coco; kwargs='{"dataset_path":"/workspace/OptimalCedar/datasets/coco","split":"train2017"}' ;;
    commonvoice) dataset=/workspace/OptimalCedar/datasets/commonvoice/cv15_en_train_5shards; kwargs="{\"dataset_path\":\"${dataset}\"}" ;;
    simclrv2) dataset=/workspace/OptimalCedar/datasets/imagenette2/imagenette2/train; kwargs="{\"dataset_path\":\"${dataset}\"}" ;;
  esac
  local out="${CONTAINER_RUN_ROOT}/systems/${workload}/plumber/${tag}.json"
  local host_out="${RUN_ROOT}/systems/${workload}/plumber/${tag}.json"
  local stats="${CONTAINER_RUN_ROOT}/systems/${workload}/plumber/${tag}.pb"
  local log="${RUN_ROOT}/logs/${workload}__plumber__${tag}.log"
  mkdir -p "$(dirname "${host_out}")"
  local -a cache_arg=()
  [[ "${workload}" == *_cache ]] && cache_arg+=(--cache)
  docker exec -e PYTHONPATH="${CONTAINER_REPO}" -e CPU_BUDGET=64 \
    -w "${CONTAINER_REPO}" "${PLUMBER_CONTAINER}" \
    timeout --signal=TERM --kill-after=30s 3600 \
    python evaluation/plumber/run_formal_cell.py \
      --dataset-file "evaluation/pipelines/${base}/tf_dataset.py" \
      --dataset-kwargs "${kwargs}" --stats-file "${stats}" \
      --results-path "${out}" --num-samples "${samples}" \
      --profile-samples 1000 --profile-seconds 10 --benchmark-seconds 42 \
      "${cache_arg[@]}" \
      > "${log}" 2>&1
  local status=$?
  if [[ "${status}" -eq 124 || "${status}" -eq 137 ]]; then
    write_status "${workload}" plumber "${tag}" execution_timeout "execution exceeded 3600s"
  elif [[ "${status}" -eq 0 && -s "${host_out}" ]]; then
    write_status "${workload}" plumber "${tag}" success ""
  else
    write_status "${workload}" plumber "${tag}" failed "exit_status=${status}"
  fi
}

run_fastflow_cell() {
  local workload="$1" attempt="$2" round="$3" samples="$4"
  local tag="attempt${attempt}__round${round}"
  recorded "${workload}" fastflow "${tag}" && return
  if previous_timeout "${workload}" fastflow "${attempt}" "${round}"; then
    write_status "${workload}" fastflow "${tag}" skipped_after_timeout "earlier round exceeded 3600s"; return
  fi
  local entry
  entry="$(cedar_exec python -c 'from evaluation.baselines.registry import get_entry; import sys; e=get_entry("fastflow",sys.argv[1]); print(e.status, e.reason or "", sep="\t")' "${workload}")"
  if [[ "${entry%%$'\t'*}" != supported ]]; then
    write_status "${workload}" fastflow "${tag}" unsupported "${entry#*$'\t'}"; return
  fi
  if ! container_running "${FASTFLOW_CONTAINER}"; then
    write_status "${workload}" fastflow "${tag}" environment_unavailable "FastFlow container is not running"; return
  fi
  local base="${workload%_cache}" dataset=''
  case "${base}" in
    coco) dataset=/workspace/OptimalCedar/datasets/coco ;;
    commonvoice) dataset=/workspace/OptimalCedar/datasets/commonvoice/cv15_en_train_5shards ;;
    simclrv2) dataset=/workspace/OptimalCedar/datasets/imagenette2/imagenette2/train ;;
    wikitext103) dataset=/workspace/OptimalCedar/datasets/wikitext103/wikitext-103/wiki.train.tokens ;;
  esac
  local out="${CONTAINER_RUN_ROOT}/systems/${workload}/fastflow/${tag}.json"
  local host_out="${RUN_ROOT}/systems/${workload}/fastflow/${tag}.json"
  local log="${RUN_ROOT}/logs/${workload}__fastflow__${tag}.log"
  mkdir -p "$(dirname "${host_out}")"
  docker exec -e PYTHONPATH="${CONTAINER_REPO}" -e COCO_SPLIT=train2017 -w "${CONTAINER_REPO}" \
    "${FASTFLOW_CONTAINER}" timeout --signal=TERM --kill-after=30s 3600 \
    python evaluation/fastflow/examples/eval_app_runner.py \
      "evaluation/fastflow/workloads/${base}_app.py" "${dataset}" ff \
      evaluation/fastflow/examples/config.yaml --epochs 1 --batch 1 \
      --parallel 8 --num_local_workers 8 --num_samples "${samples}" \
      --results_path "${out}" > "${log}" 2>&1
  local status=$?
  if [[ "${status}" -eq 0 && -s "${host_out}" ]]; then
    cedar_exec python - "${out}" "${samples}" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1]); samples = int(sys.argv[2])
data = json.loads(path.read_text())
times = data.get("epoch_times_sec", [])
data["num_samples"] = samples
data["measured_time_sec"] = times[0] if len(times) == 1 else None
path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
PY
    write_status "${workload}" fastflow "${tag}" success ""
  elif [[ "${status}" -eq 124 || "${status}" -eq 137 ]]; then
    write_status "${workload}" fastflow "${tag}" execution_timeout "execution exceeded 3600s"
  else
    write_status "${workload}" fastflow "${tag}" failed "exit_status=${status}"
  fi
}

run_external_systems() {
  local workload="$1" attempt="$2" samples="$3" round offset i system
  for ((round=1; round<=REPEATS; round++)); do
    offset=$(((round - 1) % 3))
    for ((i=0; i<3; i++)); do
      system="${SYSTEMS[$(((offset + i) % 3))]}"
      run_native_cell "${workload}" "${system}" "${attempt}" "${round}" "${samples}"
    done
    run_plumber_cell "${workload}" "${attempt}" "${round}" "${samples}"
    run_fastflow_cell "${workload}" "${attempt}" "${round}" "${samples}"
  done
}

cat > "${RUN_ROOT}/metadata.txt" <<EOF
protocol=enlarged_reused_plan_w8
selected_workloads=${SELECTED_WORKLOADS}
local_workers=8
cpu_budget=64
repeats=3
optimizer_timeout_sec=3600
execution_timeout_sec=3600
profile_policy=reuse_only
plan_policy=strict_w8_reuse_when_valid_otherwise_regenerate_from_profile
forced_plan_regeneration=${FORCE_REGENERATE_PLANS}
shrink_rule=halve_output_target_if_more_than_two_optimizers_time_out
datajuicer_system=excluded_by_request
no_optimizer=excluded_by_request
EOF

: > "${RUN_ROOT}/final_samples.tsv"
echo "[$(date -Is)] START run_id=${RUN_ID}"
for CURRENT_WORKLOAD in "${WORKLOADS[@]}"; do
  selected "${CURRENT_WORKLOAD}" || continue
  set_config "${CURRENT_WORKLOAD}"
  samples="${TARGET}"
  attempt=1
  while true; do
    echo "[$(date -Is)] WORKLOAD ${CURRENT_WORKLOAD} attempt=${attempt} samples=${samples}"
    shrink=0
    for ((round=1; round<=REPEATS; round++)); do
      offset=$(((round - 1) % ${#OPTIMIZERS[@]}))
      for ((i=0; i<${#OPTIMIZERS[@]}; i++)); do
        optimizer="${OPTIMIZERS[$(((offset + i) % ${#OPTIMIZERS[@]}))]}"
        run_cedar_cell "${CURRENT_WORKLOAD}" "${optimizer}" "${attempt}" "${round}" "${samples}"
      done
      timed_out="$(timeout_count "${CURRENT_WORKLOAD}" "${attempt}" "${round}")"
      if (( timed_out > 2 )); then shrink=1; break; fi
    done
    if (( shrink == 0 )); then break; fi
    if (( samples <= FLOOR )); then
      echo "[$(date -Is)] CANNOT-SHRINK ${CURRENT_WORKLOAD}: floor=${FLOOR}, timeouts=${timed_out}"
      break
    fi
    next=$(((samples + 1) / 2)); (( next < FLOOR )) && next="${FLOOR}"
    echo "[$(date -Is)] SHRINK ${CURRENT_WORKLOAD}: ${samples} -> ${next}; optimizer_timeouts=${timed_out}"
    samples="${next}"; attempt=$((attempt + 1))
  done
  printf '%s\t%s\t%s\n' "${CURRENT_WORKLOAD}" "${samples}" "${attempt}" >> "${RUN_ROOT}/final_samples.tsv"
  run_external_systems "${CURRENT_WORKLOAD}" "${attempt}" "${samples}"
done

cedar_exec python evaluation/chapter6_experiments/analyze_scaled_reuse_plan_matrix.py \
  --run-root "${CONTAINER_RUN_ROOT}" --repo-root "${CONTAINER_REPO}" \
  --json-output "${CONTAINER_RUN_ROOT}/scaled_reuse_plan_results.json" \
  --markdown-output "${CONTAINER_RUN_ROOT}/scaled_reuse_plan_results.md"
cp -p "${RUN_ROOT}/scaled_reuse_plan_results.json" "${FORMAL_DIR}/scaled_reuse_plan_latest.json"
cp -p "${RUN_ROOT}/scaled_reuse_plan_results.md" "${FORMAL_DIR}/scaled_reuse_plan_latest.md"
echo "[$(date -Is)] COMPLETE run_id=${RUN_ID}"
