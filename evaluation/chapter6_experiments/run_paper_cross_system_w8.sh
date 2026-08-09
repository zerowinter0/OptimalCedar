#!/usr/bin/env bash
# Formal three-repeat cross-system matrix for the workloads in the optimizer figure.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONTAINER_REPO=/workspace/OptimalCedar
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/outputs/chapter6_experiments/cross_system_w8}"
RUN_REL="${RUN_ROOT#${REPO_ROOT}/}"
CONTAINER_RUN_ROOT="${CONTAINER_REPO}/${RUN_REL}"
CEDAR_CONTAINER="${CEDAR_CONTAINER:-optimalcedar-torch201-dev}"
PLUMBER_CONTAINER="${PLUMBER_CONTAINER:-optimalcedar-plumber}"
FASTFLOW_CONTAINER="${FASTFLOW_CONTAINER:-optimalcedar-fastflow}"
CELL_TIMEOUT_SEC=3600
WORKERS=8
CPU_BUDGET=64
REPEATS=3
RESUME="${RESUME:-0}"

WORKLOADS=(
  coco commonvoice commonvoice_cache llava_pretrain redpajama_c4
  stackexchange simclrv2 simclrv2_cache wikitext103 wikitext103_cache
  redpajama_code pile_hackernews pile_pubmed_abstracts
  pile_uspto_backgrounds pile_europarl
)
SYSTEMS=(pytorch tensorflow ray plumber fastflow)

if [[ "${RESUME}" != 0 && "${RESUME}" != 1 ]]; then
  echo "RESUME must be 0 or 1" >&2
  exit 2
fi
if [[ -e "${RUN_ROOT}" && "${RESUME}" != 1 ]]; then
  echo "Formal cross-system directory already exists: ${RUN_ROOT}" >&2
  echo "Use RESUME=1 to continue it." >&2
  exit 2
fi

mkdir -p "${RUN_ROOT}"/{results,status,logs,cache,artifacts,figures}

cedar_exec() {
  docker exec -i \
    -e CPU_BUDGET="${CPU_BUDGET}" \
    -e TF_CPP_MIN_LOG_LEVEL=2 \
    -e PYTHONWARNINGS=ignore \
    "${CEDAR_CONTAINER}" bash -lc '
      cd /workspace/OptimalCedar
      source env/bin/activate
      export PYTHONPATH=/workspace/OptimalCedar:${PYTHONPATH:-}
      ulimit -n 65536
      exec "$@"
    ' bash "$@"
}

container_running() {
  docker inspect "$1" >/dev/null 2>&1 &&
    [[ "$(docker inspect -f '{{.State.Running}}' "$1")" == true ]]
}

status_path() {
  printf '%s/status/%s/round%s__%s.tsv\n' "${RUN_ROOT}" "$1" "$2" "$3"
}

write_status() {
  local workload="$1" round="$2" system="$3" state="$4" reason="${5:-}"
  local path
  path="$(status_path "${workload}" "${round}" "${system}")"
  mkdir -p "$(dirname "${path}")"
  reason="${reason//$'\n'/ }"
  reason="${reason//$'\t'/ }"
  printf '%s\t%s\n' "${state}" "${reason}" > "${path}"
}

cell_recorded() {
  [[ -s "$(status_path "$1" "$2" "$3")" ]]
}

samples_for() {
  case "$1" in
    coco) echo 50000 ;;
    commonvoice|commonvoice_cache) echo 160000 ;;
    llava_pretrain) echo 5000 ;;
    redpajama_c4) echo 20000 ;;
    stackexchange) echo 10000 ;;
    simclrv2|simclrv2_cache) echo 9469 ;;
    wikitext103|wikitext103_cache) echo 100000 ;;
    redpajama_code|pile_hackernews|pile_pubmed_abstracts|pile_uspto_backgrounds) echo 20000 ;;
    pile_europarl) echo 2500 ;;
    *) return 1 ;;
  esac
}

set_workload_config() {
  local workload="$1"
  DATASET_PATH=""
  IMAGE_ROOT=""
  DATASET_KWARGS='{}'
  PLUMBER_KWARGS='{}'
  FASTFLOW_PREFIX=""
  case "${workload}" in
    coco)
      DATASET_KWARGS='{"split":"train2017"}'
      PLUMBER_KWARGS='{"dataset_path":"/workspace/OptimalCedar/datasets/coco","split":"train2017"}'
      FASTFLOW_PREFIX=/workspace/OptimalCedar/datasets/coco
      ;;
    commonvoice|commonvoice_cache)
      PLUMBER_KWARGS='{"dataset_path":"/workspace/OptimalCedar/datasets/commonvoice/cv-corpus-15.0-delta-2023-09-08/en/clips"}'
      FASTFLOW_PREFIX=/workspace/OptimalCedar/datasets/commonvoice/cv-corpus-15.0-delta-2023-09-08/en/clips
      ;;
    llava_pretrain)
      DATASET_PATH=evaluation/datasets/llava_pretrain/blip_laion_cc_sbu_20000_dj_fmt_only_caption.jsonl
      IMAGE_ROOT=evaluation/datasets/llava_pretrain
      ;;
    redpajama_c4)
      DATASET_PATH=datasets/redpajama_c4/redpajama-c4-raw-829916.jsonl ;;
    stackexchange)
      DATASET_PATH=datasets/stackexchange/redpajama-stackexchange-35000.jsonl ;;
    simclrv2|simclrv2_cache)
      PLUMBER_KWARGS='{"dataset_path":"/workspace/OptimalCedar/datasets/imagenette2/imagenette2/train"}'
      FASTFLOW_PREFIX=/workspace/OptimalCedar/datasets/imagenette2/imagenette2/train
      ;;
    wikitext103|wikitext103_cache)
      PLUMBER_KWARGS='{"dataset_path":"/workspace/OptimalCedar/datasets/wikitext103/wikitext-103/wiki.train.tokens"}'
      FASTFLOW_PREFIX=/workspace/OptimalCedar/datasets/wikitext103/wikitext-103/wiki.train.tokens
      ;;
    redpajama_code)
      DATASET_PATH=datasets/redpajama_code/redpajama-github-raw-50000.jsonl ;;
    pile_hackernews)
      DATASET_PATH=datasets/pile_hackernews/pile-hackernews-raw-100000.jsonl ;;
    pile_pubmed_abstracts)
      DATASET_PATH=datasets/pile_pubmed_abstracts/pile-pubmed-abstracts-raw-100000.jsonl ;;
    pile_uspto_backgrounds)
      DATASET_PATH=datasets/pile_uspto_backgrounds/pile-uspto-backgrounds-raw-100000.jsonl ;;
    pile_europarl)
      DATASET_PATH=datasets/pile_europarl/pile-europarl-raw.jsonl ;;
  esac
}

unsupported_reason() {
  local workload="$1" system="$2" base="${workload%_cache}"
  case "${system}" in
    tensorflow)
      [[ "${workload}" == redpajama_c4 ]] && {
        echo "tf.py_function is GIL-bound for this opaque callback pipeline at W=8"
        return 0
      }
      ;;
    plumber)
      case "${base}" in
        coco|commonvoice|simclrv2) ;;
        wikitext103)
          echo "TextLineDataset FlatMap source lacks Plumber byte-ratio metadata"
          return 0
          ;;
        *) echo "opaque Python callbacks cannot be modeled and safely reconstructed by Plumber"; return 0 ;;
      esac
      ;;
    fastflow)
      case "${base}" in
        coco|commonvoice|simclrv2|wikitext103) ;;
        *) echo "opaque callbacks cannot be serialized by tf.data service/FastFlow"; return 0 ;;
      esac
      ;;
  esac
  return 1
}

record_exit() {
  local workload="$1" round="$2" system="$3" code="$4" result="$5"
  if [[ "${code}" -eq 0 && -s "${result}" ]]; then
    write_status "${workload}" "${round}" "${system}" success ""
  elif [[ "${code}" -eq 124 || "${code}" -eq 137 ]]; then
    write_status "${workload}" "${round}" "${system}" infeasible_timeout \
      "execution exceeded ${CELL_TIMEOUT_SEC}s"
  elif [[ "${code}" -eq 0 ]]; then
    write_status "${workload}" "${round}" "${system}" failed "no result file"
  else
    write_status "${workload}" "${round}" "${system}" failed "exit_status=${code}"
  fi
}

run_native() {
  local workload="$1" round="$2" system="$3" samples="$4"
  local result="${RUN_ROOT}/results/${workload}/round${round}__${system}.json"
  local container_result="${CONTAINER_RUN_ROOT}/results/${workload}/round${round}__${system}.json"
  local log="${RUN_ROOT}/logs/${workload}__round${round}__${system}.log"
  mkdir -p "$(dirname "${result}")"
  local -a args=(python -m evaluation.baselines.run
    --system "${system}" --workload "${workload}" --batch-size 1
    --workers "${WORKERS}" --epochs 1 --num-samples "${samples}"
    --cache-dir "${CONTAINER_RUN_ROOT}/cache/round${round}/${system}"
    --ray-address 127.0.0.1:6379 --dataset-kwargs "${DATASET_KWARGS}"
    --results-path "${container_result}")
  [[ -n "${DATASET_PATH}" ]] && args+=(--dataset-path "${DATASET_PATH}")
  [[ -n "${IMAGE_ROOT}" ]] && args+=(--image-root "${IMAGE_ROOT}")
  echo "[$(date -Is)] RUN ${workload}/round${round}/${system} outputs=${samples}"
  docker exec -i -e TF_CPP_MIN_LOG_LEVEL=2 -e PYTHONWARNINGS=ignore \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    "${CEDAR_CONTAINER}" bash -lc '
      cd /workspace/OptimalCedar
      source env/bin/activate
      export PYTHONPATH=/workspace/OptimalCedar:${PYTHONPATH:-}
      ulimit -n 65536
      exec timeout --signal=TERM --kill-after=30s 3600 "$@"
    ' bash "${args[@]}" > "${log}" 2>&1
  local code=$?
  if [[ "${code}" -eq 0 && -s "${result}" ]]; then
    cedar_exec python - "${container_result}" "${samples}" <<'PY' >> "${log}" 2>&1 || code=$?
import json, sys
p=json.load(open(sys.argv[1]))
if p.get("num_samples") != int(sys.argv[2]) or not p.get("measured_time_sec", 0) > 0:
    raise SystemExit("invalid cardinality or timing")
PY
  fi
  record_exit "${workload}" "${round}" "${system}" "${code}" "${result}"
}

run_plumber() {
  local workload="$1" round="$2" samples="$3"
  local result="${RUN_ROOT}/results/${workload}/round${round}__plumber.json"
  local container_result="${CONTAINER_RUN_ROOT}/results/${workload}/round${round}__plumber.json"
  local stats="${CONTAINER_RUN_ROOT}/artifacts/${workload}/round${round}__plumber.pb"
  local log="${RUN_ROOT}/logs/${workload}__round${round}__plumber.log"
  mkdir -p "$(dirname "${result}")"
  local -a cache_arg=()
  [[ "${workload}" == *_cache ]] && cache_arg=(--cache)
  echo "[$(date -Is)] RUN ${workload}/round${round}/plumber outputs=${samples}"
  docker exec -i -e PYTHONPATH="${CONTAINER_REPO}" -e CPU_BUDGET="${CPU_BUDGET}" \
    -w "${CONTAINER_REPO}" "${PLUMBER_CONTAINER}" \
    timeout --signal=TERM --kill-after=30s "${CELL_TIMEOUT_SEC}" \
    python evaluation/plumber/run_formal_cell.py \
      --dataset-file "evaluation/pipelines/${workload%_cache}/tf_dataset.py" \
      --dataset-kwargs "${PLUMBER_KWARGS}" --stats-file "${stats}" \
      --results-path "${container_result}" --num-samples "${samples}" \
      --profile-samples 1000 --profile-seconds 10 --benchmark-seconds 42 \
      "${cache_arg[@]}" > "${log}" 2>&1
  record_exit "${workload}" "${round}" plumber "$?" "${result}"
}

run_fastflow() {
  local workload="$1" round="$2" samples="$3" base="${workload%_cache}"
  local result="${RUN_ROOT}/results/${workload}/round${round}__fastflow.json"
  local container_result="${CONTAINER_RUN_ROOT}/results/${workload}/round${round}__fastflow.json"
  local log="${RUN_ROOT}/logs/${workload}__round${round}__fastflow.log"
  mkdir -p "$(dirname "${result}")"
  echo "[$(date -Is)] RUN ${workload}/round${round}/fastflow outputs=${samples}"
  docker exec -i -e PYTHONPATH="${CONTAINER_REPO}" -e COCO_SPLIT=train2017 \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -w "${CONTAINER_REPO}" "${FASTFLOW_CONTAINER}" \
    timeout --signal=TERM --kill-after=30s "${CELL_TIMEOUT_SEC}" \
    python evaluation/fastflow/examples/eval_app_runner.py \
      "evaluation/fastflow/workloads/${base}_app.py" "${FASTFLOW_PREFIX}" ff \
      evaluation/fastflow/examples/config.yaml --epochs 1 --batch 1 \
      --parallel "${WORKERS}" --num_local_workers "${WORKERS}" \
      --num_samples "${samples}" --results_path "${container_result}" \
      > "${log}" 2>&1
  local code=$?
  if [[ "${code}" -eq 0 && -s "${result}" ]]; then
    cedar_exec python - "${container_result}" "${workload}" "${samples}" <<'PY' >> "${log}" 2>&1 || code=$?
import json, sys
path, workload, samples=sys.argv[1],sys.argv[2],int(sys.argv[3])
p=json.load(open(path))
times=p.get("epoch_times_sec")
if not isinstance(times,list) or len(times)!=1 or not times[0]>0:
    raise SystemExit("invalid FastFlow timing")
p.update(workload=workload,num_samples=samples,measured_time_sec=float(times[0]))
open(path,"w").write(json.dumps(p,indent=2,sort_keys=True)+"\n")
PY
  fi
  record_exit "${workload}" "${round}" fastflow "${code}" "${result}"
}

cat > "${RUN_ROOT}/README.md" <<EOF
# Formal cross-system results (W=8)

This directory combines all five Cedar optimizer results with native PyTorch
DataLoader, tf.data, Ray Data, Plumber, and FastFlow on the 15
workloads in the current optimizer figure. FreeLaw is excluded. Every supported
cell has three round-robin repeats, an 8-worker setting, a 64-CPU budget, and
the exact output count in
\`../paper_artifacts/optimizer/figures/latest_optimizer_data.tsv\`. The primary figure,
\`figures/optimizer_and_system_execution_time.pdf\`, reports absolute execution
time in seconds. Every workload has an independent linear y-axis; the figure
uses neither a speedup axis nor a logarithmic axis.

Unsupported cells and one-hour feasibility timeouts remain explicit in status/.
Superseded attempts are preserved by failure class under invalidated_attempts/;
the runner ignores that archive when resuming or plotting.
EOF

cat > "${RUN_ROOT}/metadata.txt" <<EOF
protocol=formal_cross_system_w8_three_repeat
workloads=${WORKLOADS[*]}
systems=${SYSTEMS[*]}
excluded_workload=pile_freelaw
workers=${WORKERS}
cpu_budget=${CPU_BUDGET}
repeats=${REPEATS}
cell_timeout_sec=${CELL_TIMEOUT_SEC}
optimizer_source=evaluation/chapter6_experiments/formal_results/paper_artifacts/optimizer/figures/latest_optimizer_data.tsv
EOF

if ! container_running "${CEDAR_CONTAINER}"; then
  echo "Cedar container is unavailable: ${CEDAR_CONTAINER}" >&2
  exit 1
fi

cedar_exec python -m evaluation.baselines.run --validate > "${RUN_ROOT}/baseline_validation.json" || exit 1
docker exec "${CEDAR_CONTAINER}" bash -lc "
  cd ${CONTAINER_REPO}; source env/bin/activate
  ray status --address=127.0.0.1:6379 >/dev/null 2>&1 ||
    ray start --head --node-ip-address=127.0.0.1 --port=6379 \
      --num-cpus=${CPU_BUDGET} --disable-usage-stats >/dev/null
" || exit 1

echo "[$(date -Is)] Starting formal cross-system W=8 matrix"
for ((round=1; round<=REPEATS; round++)); do
  offset=$(((round-1)%${#SYSTEMS[@]}))
  for workload in "${WORKLOADS[@]}"; do
    samples="$(samples_for "${workload}")" || exit 1
    set_workload_config "${workload}"
    for ((i=0; i<${#SYSTEMS[@]}; i++)); do
      system="${SYSTEMS[$(((offset+i)%${#SYSTEMS[@]}))]}"
      cell_recorded "${workload}" "${round}" "${system}" && continue
      if reason="$(unsupported_reason "${workload}" "${system}")"; then
        write_status "${workload}" "${round}" "${system}" unsupported "${reason}"
        continue
      fi
      case "${system}" in
        pytorch|tensorflow|ray)
          run_native "${workload}" "${round}" "${system}" "${samples}" ;;
        plumber)
          if container_running "${PLUMBER_CONTAINER}"; then
            run_plumber "${workload}" "${round}" "${samples}"
          else
            write_status "${workload}" "${round}" "${system}" environment_unavailable "pinned Plumber container is unavailable"
          fi ;;
        fastflow)
          if container_running "${FASTFLOW_CONTAINER}"; then
            run_fastflow "${workload}" "${round}" "${samples}"
          else
            write_status "${workload}" "${round}" "${system}" environment_unavailable "pinned FastFlow container is unavailable"
          fi ;;
      esac
    done
  done
done

if ! cedar_exec python evaluation/chapter6_experiments/plot_paper_cross_system_absolute.py \
    --run-root "${CONTAINER_RUN_ROOT}" \
    --optimizer-tsv "${CONTAINER_REPO}/evaluation/chapter6_experiments/formal_results/paper_artifacts/optimizer/figures/latest_optimizer_data.tsv" \
    --output-dir "${CONTAINER_RUN_ROOT}/figures"; then
  echo "plot_or_validation_failed" > "${RUN_ROOT}/STATUS"
  exit 1
fi
echo "complete" > "${RUN_ROOT}/STATUS"
echo "[$(date -Is)] Formal cross-system matrix complete: ${RUN_ROOT}"
