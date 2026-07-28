#!/usr/bin/env bash
# Run the reduced single-run W=8 protocol across Cedar plans and external
# systems. Data-Juicer is deliberately excluded.
#
# This is a host-side orchestrator: every Python workload runs in its actual
# Docker environment. It continues after unsupported cells, optimizer
# timeouts, execution timeouts, and individual failures, then renders one
# machine-readable JSON report and one paper-readable Markdown report.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST_REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONTAINER_REPO="/workspace/OptimalCedar"
FORMAL_DIR="${HOST_REPO}/evaluation/chapter6_experiments/formal_results"

CEDAR_CONTAINER="${CEDAR_CONTAINER:-optimalcedar-torch201-dev}"
PLUMBER_CONTAINER="${PLUMBER_CONTAINER:-optimalcedar-plumber}"
FASTFLOW_CONTAINER="${FASTFLOW_CONTAINER:-optimalcedar-fastflow}"
RAY_ADDRESS="${RAY_ADDRESS:-127.0.0.1:6379}"
CPU_BUDGET="${CPU_BUDGET:-64}"
LOCAL_WORKERS="${LOCAL_WORKERS:-8}"
OPTIMIZER_TIMEOUT_SEC="${OPTIMIZER_TIMEOUT_SEC:-300}"
CELL_TIMEOUT_SEC="${CELL_TIMEOUT_SEC:-3600}"
PLUMBER_BENCHMARK_SECONDS="${PLUMBER_BENCHMARK_SECONDS:-42}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RESUME="${RESUME:-0}"
SELECTED_WORKLOADS="all"

WORKLOADS=(
  coco
  commonvoice
  commonvoice_cache
  llava_pretrain
  redpajama_c4
  stackexchange
  simclrv2
  simclrv2_cache
  wikitext103
  wikitext103_cache
)
CEDAR_OPTIMIZERS=(
  optimizer
  dp_optimizer
  dp_cedar_optimizer
  dj_optimizer
  pecan_optimizer
)
NATIVE_SYSTEMS=(pytorch tensorflow ray)

usage() {
  cat <<'EOF'
Usage: run_cross_system_w8_matrix.sh [options]

Options:
  --workloads NAME[,NAME...]  Run a subset; default is all ten workloads.
  --run-id ID                 Stable output directory name for resume.
  --resume                    Keep completed cell statuses in the run directory.
  -h, --help                  Show this help.

Environment:
  CEDAR_CONTAINER             Default: optimalcedar-torch201-dev
  PLUMBER_CONTAINER           Pinned Plumber container (default: optimalcedar-plumber)
  FASTFLOW_CONTAINER          Pinned FastFlow container (default: optimalcedar-fastflow)
  CELL_TIMEOUT_SEC            Per execution cell; fixed at 3600 (one hour)
  OPTIMIZER_TIMEOUT_SEC       Cedar optimization/setup limit; fixed default: 300

Protocol:
  W=8, CPU_BUDGET=64, one measured run, batch size 1, the exact output counts
  and bounded inputs used by formal_results/w8_acceptance_latest.md.
EOF
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --workloads)
      [[ "$#" -ge 2 ]] || { echo "--workloads requires a value" >&2; exit 2; }
      SELECTED_WORKLOADS="$2"
      shift 2
      ;;
    --run-id)
      [[ "$#" -ge 2 ]] || { echo "--run-id requires a value" >&2; exit 2; }
      RUN_ID="$2"
      shift 2
      ;;
    --resume)
      RESUME=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "${LOCAL_WORKERS}" != "8" || "${CPU_BUDGET}" != "64" ]]; then
  echo "This protocol requires LOCAL_WORKERS=8 and CPU_BUDGET=64." >&2
  exit 2
fi
if [[ "${OPTIMIZER_TIMEOUT_SEC}" != "300" ]]; then
  echo "The acceptance protocol requires OPTIMIZER_TIMEOUT_SEC=300." >&2
  exit 2
fi
if [[ "${CELL_TIMEOUT_SEC}" != "3600" ]]; then
  echo "The formal protocol requires CELL_TIMEOUT_SEC=3600." >&2
  exit 2
fi
if [[ "${RESUME}" != "0" && "${RESUME}" != "1" ]]; then
  echo "RESUME must be 0 or 1." >&2
  exit 2
fi
if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "RUN_ID may contain only letters, digits, dot, underscore, or dash." >&2
  exit 2
fi
command -v docker >/dev/null 2>&1 || {
  echo "docker is required on the host." >&2
  exit 1
}
docker inspect "${CEDAR_CONTAINER}" >/dev/null 2>&1 || {
  echo "Cedar container not found: ${CEDAR_CONTAINER}" >&2
  exit 1
}
if [[ "$(docker inspect -f '{{.State.Running}}' "${CEDAR_CONTAINER}")" != "true" ]]; then
  echo "Cedar container is not running: ${CEDAR_CONTAINER}" >&2
  exit 1
fi
docker exec "${CEDAR_CONTAINER}" test -f \
  "${CONTAINER_REPO}/evaluation/compare_optimizer_perf.py" || {
  echo "${CONTAINER_REPO} is not mounted in ${CEDAR_CONTAINER}." >&2
  exit 1
}

RUN_REL="evaluation/chapter6_experiments/formal_results/cross_system_w8_runs/${RUN_ID}"
RUN_ROOT="${HOST_REPO}/${RUN_REL}"
CONTAINER_RUN_ROOT="${CONTAINER_REPO}/${RUN_REL}"
if [[ -e "${RUN_ROOT}" && "${RESUME}" == "0" ]]; then
  echo "Run directory already exists: ${RUN_ROOT}" >&2
  echo "Use --resume with the same --run-id or choose a new run id." >&2
  exit 2
fi
mkdir -p \
  "${RUN_ROOT}/cedar" \
  "${RUN_ROOT}/systems" \
  "${RUN_ROOT}/status" \
  "${RUN_ROOT}/logs" \
  "${RUN_ROOT}/cache"
printf '%s\n' "${RUN_ROOT}" > "${FORMAL_DIR}/cross_system_w8_latest_run.txt"

workload_selected() {
  local workload="$1"
  [[ "${SELECTED_WORKLOADS}" == "all" ]] && return 0
  case ",${SELECTED_WORKLOADS}," in
    *,"${workload}",*) return 0 ;;
    *) return 1 ;;
  esac
}

container_running() {
  local name="$1"
  [[ -n "${name}" ]] || return 1
  docker inspect "${name}" >/dev/null 2>&1 || return 1
  [[ "$(docker inspect -f '{{.State.Running}}' "${name}")" == "true" ]]
}

cedar_exec() {
  docker exec -i \
    -e CPU_BUDGET="${CPU_BUDGET}" \
    -e TF_CPP_MIN_LOG_LEVEL=2 \
    -e PYTHONWARNINGS=ignore \
    "${CEDAR_CONTAINER}" \
    bash -lc '
      cd /workspace/OptimalCedar
      source env/bin/activate
      export PYTHONPATH=/workspace/OptimalCedar:${PYTHONPATH:-}
      ulimit -n 65536
      exec "$@"
    ' bash "$@"
}

cedar_exec_timed() {
  docker exec -i \
    -e CPU_BUDGET="${CPU_BUDGET}" \
    -e TF_CPP_MIN_LOG_LEVEL=2 \
    -e PYTHONWARNINGS=ignore \
    -e CELL_TIMEOUT_SEC="${CELL_TIMEOUT_SEC}" \
    "${CEDAR_CONTAINER}" \
    bash -lc '
      cd /workspace/OptimalCedar
      source env/bin/activate
      export PYTHONPATH=/workspace/OptimalCedar:${PYTHONPATH:-}
      ulimit -n 65536
      exec timeout --signal=TERM --kill-after=30s \
        "${CELL_TIMEOUT_SEC}" "$@"
    ' bash "$@"
}

status_path() {
  printf '%s/status/%s/%s.tsv' "${RUN_ROOT}" "$1" "$2"
}

write_status() {
  local workload="$1" entity="$2" state="$3" reason="${4:-}"
  local path
  path="$(status_path "${workload}" "${entity}")"
  mkdir -p "$(dirname "${path}")"
  printf '%s\t%s\n' "${state}" "${reason//$'\n'/ }" > "${path}"
}

cell_is_recorded() {
  local path
  path="$(status_path "$1" "$2")"
  [[ "${RESUME}" == "1" && -s "${path}" ]]
}

elapsed_seconds() {
  local start_ns="$1" end_ns="$2"
  awk -v start="${start_ns}" -v end="${end_ns}" \
    'BEGIN { printf "%.9f\n", (end - start) / 1000000000 }'
}

record_command_status() {
  local workload="$1" entity="$2" status="$3" result="$4"
  if [[ "${status}" -eq 0 && -s "${result}" ]]; then
    write_status "${workload}" "${entity}" success ""
  elif [[ "${status}" -eq 124 || "${status}" -eq 137 ]]; then
    write_status "${workload}" "${entity}" infeasible_timeout \
      "execution exceeded the one-hour (${CELL_TIMEOUT_SEC}s) feasibility limit"
  elif [[ "${status}" -eq 0 ]]; then
    write_status "${workload}" "${entity}" failed \
      "command succeeded but produced no result"
  else
    write_status "${workload}" "${entity}" failed "exit_status=${status}"
  fi
}

# Sets the workload-specific globals used by every system.
set_workload_config() {
  local workload="$1"
  PROFILE="${CONTAINER_REPO}/evaluation/chapter6_experiments/formal_results/profiles/${workload}.yaml"
  CACHE_MODE=off
  CEDAR_KWARGS=""
  NATIVE_DATASET=""
  NATIVE_IMAGE_ROOT=""
  NATIVE_KWARGS_JSON="{}"
  PLUMBER_KWARGS_JSON="{}"
  FASTFLOW_DATASET=""

  case "${workload}" in
    coco)
      SAMPLES=5000
      CEDAR_DATASET="evaluation/pipelines/coco/cedar_dataset.py"
      CEDAR_KWARGS="split=val2017"
      NATIVE_KWARGS_JSON='{"split":"val2017"}'
      PLUMBER_KWARGS_JSON='{"dataset_path":"/workspace/OptimalCedar/datasets/coco"}'
      FASTFLOW_DATASET="/workspace/OptimalCedar/datasets/coco"
      ;;
    commonvoice|commonvoice_cache)
      SAMPLES=10000
      if [[ "${workload}" == *_cache ]]; then
        CACHE_MODE=on
        CEDAR_DATASET="evaluation/pipelines/commonvoice/cedar_cache_dataset.py"
      else
        CEDAR_DATASET="evaluation/pipelines/commonvoice/cedar_dataset.py"
      fi
      CEDAR_KWARGS="max_samples=10000"
      PLUMBER_KWARGS_JSON='{"dataset_path":"/workspace/OptimalCedar/datasets/commonvoice/cv-corpus-15.0-delta-2023-09-08/en/clips"}'
      FASTFLOW_DATASET="/workspace/OptimalCedar/datasets/commonvoice/cv-corpus-15.0-delta-2023-09-08/en/clips"
      ;;
    llava_pretrain)
      SAMPLES=5000
      CEDAR_DATASET="evaluation/pipelines/llava_pretrain/cedar_dataset.py"
      NATIVE_DATASET="evaluation/datasets/llava_pretrain/blip_laion_cc_sbu_20000_dj_fmt_only_caption.jsonl"
      NATIVE_IMAGE_ROOT="evaluation/datasets/llava_pretrain"
      CEDAR_KWARGS="dataset_path=${NATIVE_DATASET},image_root=${NATIVE_IMAGE_ROOT}"
      ;;
    redpajama_c4)
      SAMPLES=20000
      CEDAR_DATASET="evaluation/pipelines/redpajama_c4/cedar_dataset.py"
      NATIVE_DATASET="datasets/redpajama_c4/redpajama-c4-raw-829916.jsonl"
      CEDAR_KWARGS="dataset_path=${NATIVE_DATASET}"
      ;;
    stackexchange)
      SAMPLES=10000
      CEDAR_DATASET="evaluation/pipelines/stackexchange/cedar_dataset.py"
      NATIVE_DATASET="datasets/stackexchange/redpajama-stackexchange-35000.jsonl"
      CEDAR_KWARGS="dataset_path=${NATIVE_DATASET}"
      ;;
    simclrv2|simclrv2_cache)
      SAMPLES=9469
      if [[ "${workload}" == *_cache ]]; then
        CACHE_MODE=on
        CEDAR_DATASET="evaluation/pipelines/simclrv2/cedar_cache_dataset.py"
      else
        CEDAR_DATASET="evaluation/pipelines/simclrv2/cedar_dataset.py"
      fi
      PLUMBER_KWARGS_JSON='{"dataset_path":"/workspace/OptimalCedar/datasets/imagenette2/imagenette2/train"}'
      FASTFLOW_DATASET="/workspace/OptimalCedar/datasets/imagenette2/imagenette2/train"
      ;;
    wikitext103|wikitext103_cache)
      SAMPLES=100000
      if [[ "${workload}" == *_cache ]]; then
        CACHE_MODE=on
        CEDAR_DATASET="evaluation/pipelines/wikitext103/cedar_cache_dataset.py"
      else
        CEDAR_DATASET="evaluation/pipelines/wikitext103/cedar_dataset.py"
      fi
      CEDAR_KWARGS="max_samples=100000"
      PLUMBER_KWARGS_JSON='{"dataset_path":"/workspace/OptimalCedar/datasets/wikitext103/wikitext-103/wiki.train.tokens"}'
      FASTFLOW_DATASET="/workspace/OptimalCedar/datasets/wikitext103/wikitext-103/wiki.train.tokens"
      ;;
    *)
      echo "Unknown workload: ${workload}" >&2
      return 1
      ;;
  esac
}

run_cedar_optimizer() {
  local workload="$1" optimizer="$2"
  local entity="cedar_${optimizer}"
  cell_is_recorded "${workload}" "${entity}" && return

  local out="${CONTAINER_RUN_ROOT}/cedar/${workload}/${optimizer}.json"
  local host_out="${RUN_ROOT}/cedar/${workload}/${optimizer}.json"
  local log="${RUN_ROOT}/logs/${workload}__${entity}.log"
  mkdir -p "$(dirname "${host_out}")"
  local -a args=(
    python evaluation/compare_optimizer_perf.py
    --dataset_file "${CEDAR_DATASET}"
    --profiled_stats "${PROFILE}"
    --num_total_samples "${SAMPLES}"
    --num_epochs 1
    --num_repeats 1
    --warmup_runs 0
    --full_data_run
    --use_ray
    --ray_ip "${RAY_ADDRESS}"
    --enable_local_parallelism
    --match_profile_resources
    --cpu_budget "${CPU_BUDGET}"
    --fixed_local_workers_ablation "${LOCAL_WORKERS}"
    --disable_cedar_runtime_timeout
    --optimizers "${optimizer}"
    --results_path "${out}"
  )
  [[ "${optimizer}" == "optimizer" ]] && \
    args+=(--cedar_reorder_timeout_sec "${OPTIMIZER_TIMEOUT_SEC}")
  [[ "${CACHE_MODE}" == "off" ]] && args+=(--disable_caching)
  [[ -n "${CEDAR_KWARGS}" ]] && args+=(--dataset_kwargs "${CEDAR_KWARGS}")

  echo "[$(date -Is)] RUN ${workload}/${entity}"
  cedar_exec_timed "${args[@]}" > "${log}" 2>&1
  local status=$?
  record_command_status \
    "${workload}" "${entity}" "${status}" "${host_out}"
}

run_cedar_no_optimizer() {
  local workload="$1" entity="cedar_no_optimizer"
  cell_is_recorded "${workload}" "${entity}" && return

  local out="${CONTAINER_RUN_ROOT}/cedar/${workload}/no_optimizer.json"
  local host_out="${RUN_ROOT}/cedar/${workload}/no_optimizer.json"
  local wall="${RUN_ROOT}/cedar/${workload}/no_optimizer.wall_sec"
  local log="${RUN_ROOT}/logs/${workload}__${entity}.log"
  mkdir -p "$(dirname "${host_out}")"
  local -a args=(
    python evaluation/eval_cedar.py
    --dataset_file "${CEDAR_DATASET}"
    --num_total_samples "${SAMPLES}"
    --num_epochs 1
    --disable_optimizer
    --disable_controller
    --results_path "${out}"
  )
  [[ -n "${CEDAR_KWARGS}" ]] && args+=(--dataset_kwargs "${CEDAR_KWARGS}")

  echo "[$(date -Is)] RUN ${workload}/${entity}"
  local start_ns end_ns
  start_ns="$(date +%s%N)"
  cedar_exec_timed "${args[@]}" > "${log}" 2>&1
  local status=$?
  end_ns="$(date +%s%N)"
  elapsed_seconds "${start_ns}" "${end_ns}" > "${wall}"
  record_command_status \
    "${workload}" "${entity}" "${status}" "${host_out}"
}

run_native_system() {
  local workload="$1" system="$2"
  cell_is_recorded "${workload}" "${system}" && return
  if [[ "${workload}" == "redpajama_c4" && "${system}" == "tensorflow" ]]; then
    write_status "${workload}" "${system}" infeasible \
      "tf.py_function is GIL-bound at approximately one effective CPU core, so the required W=8 comparison is infeasible"
    return
  fi

  local out="${CONTAINER_RUN_ROOT}/systems/${workload}/${system}.json"
  local host_out="${RUN_ROOT}/systems/${workload}/${system}.json"
  local log="${RUN_ROOT}/logs/${workload}__${system}.log"
  mkdir -p "$(dirname "${host_out}")"
  local -a args=(python -m evaluation.baselines.run)
  args+=(
    --system "${system}"
    --workload "${workload}"
    --batch-size 1
    --workers "${LOCAL_WORKERS}"
    --epochs 1
    --num-samples "${SAMPLES}"
    --cache-dir "${CONTAINER_RUN_ROOT}/cache/${system}"
    --ray-address "${RAY_ADDRESS}"
    --dataset-kwargs "${NATIVE_KWARGS_JSON}"
    --results-path "${out}"
  )
  [[ -n "${NATIVE_DATASET}" ]] && args+=(--dataset-path "${NATIVE_DATASET}")
  [[ -n "${NATIVE_IMAGE_ROOT}" ]] && args+=(--image-root "${NATIVE_IMAGE_ROOT}")

  echo "[$(date -Is)] RUN ${workload}/${system}"
  cedar_exec_timed "${args[@]}" > "${log}" 2>&1
  local status=$?
  record_command_status \
    "${workload}" "${system}" "${status}" "${host_out}"
}

plumber_supported() {
  case "$1" in
    coco|commonvoice|commonvoice_cache|simclrv2|simclrv2_cache)
      return 0 ;;
    *) return 1 ;;
  esac
}

run_plumber() {
  local workload="$1" entity="plumber"
  cell_is_recorded "${workload}" "${entity}" && return
  if ! plumber_supported "${workload}"; then
    write_status "${workload}" "${entity}" unsupported \
      "Plumber cannot model/reconstruct this workload's opaque callback or FlatMap source"
    return
  fi
  if ! container_running "${PLUMBER_CONTAINER}"; then
    write_status "${workload}" "${entity}" environment_unavailable \
      "set PLUMBER_CONTAINER to a running pinned Plumber environment"
    return
  fi

  local base="${workload%_cache}"
  local out="${RUN_ROOT}/systems/${workload}/plumber.json"
  local container_out="${CONTAINER_RUN_ROOT}/systems/${workload}/plumber.json"
  local stats="${CONTAINER_RUN_ROOT}/systems/${workload}/plumber_stats.pb"
  local profile_wall="${RUN_ROOT}/systems/${workload}/plumber_profile.wall_sec"
  local profile_log="${RUN_ROOT}/logs/${workload}__plumber_profile.log"
  local optimize_log="${RUN_ROOT}/logs/${workload}__plumber_optimize.log"
  mkdir -p "$(dirname "${out}")"

  echo "[$(date -Is)] PROFILE ${workload}/plumber"
  local start_ns end_ns
  start_ns="$(date +%s%N)"
  docker exec \
      -e PYTHONPATH="${CONTAINER_REPO}" \
      -w "${CONTAINER_REPO}" \
      "${PLUMBER_CONTAINER}" \
      timeout --signal=TERM --kill-after=30s "${CELL_TIMEOUT_SEC}" \
      python evaluation/plumber/profile_pipeline.py \
        --dataset-file "evaluation/pipelines/${base}/tf_dataset.py" \
        --stats-file "${stats}" \
        --dataset-kwargs "${PLUMBER_KWARGS_JSON}" \
        --profile-samples 1000 \
        --profile-seconds 10 \
        --parallelism 1 \
        --threadpool-size "${CPU_BUDGET}" \
      > "${profile_log}" 2>&1
  local status=$?
  end_ns="$(date +%s%N)"
  elapsed_seconds "${start_ns}" "${end_ns}" > "${profile_wall}"
  if [[ "${status}" -ne 0 ]]; then
    if [[ "${status}" -eq 124 || "${status}" -eq 137 ]]; then
      write_status "${workload}" "${entity}" infeasible_timeout \
        "Plumber profiling exceeded the one-hour (${CELL_TIMEOUT_SEC}s) feasibility limit"
    else
      write_status "${workload}" "${entity}" failed \
        "Plumber profiling exit_status=${status}"
    fi
    return
  fi

  echo "[$(date -Is)] RUN ${workload}/plumber"
  local profile_seconds
  profile_seconds="$(cat "${profile_wall}")"
  docker exec -i \
      -e PYTHONPATH="${CONTAINER_REPO}" \
      -w "${CONTAINER_REPO}" \
      "${PLUMBER_CONTAINER}" \
      timeout --signal=TERM --kill-after=30s "${CELL_TIMEOUT_SEC}" \
      python - "${stats}" "${container_out}" "${SAMPLES}" \
        "${PLUMBER_BENCHMARK_SECONDS}" "${profile_seconds}" "${CACHE_MODE}" \
        "evaluation/pipelines/${base}/tf_dataset.py" \
        "${PLUMBER_KWARGS_JSON}" \
      > "${optimize_log}" 2>&1 <<'PY'
import json
import sys
import time
from pathlib import Path

import tensorflow as tf
from plumber_analysis import gen_util, pipeline_optimizer
from evaluation.plumber.profile_pipeline import _import_dataset
from evaluation.tf_utils import TFEvalSpec

stats_path = Path(sys.argv[1])
result_path = Path(sys.argv[2])
num_samples = int(sys.argv[3])
benchmark_seconds = int(sys.argv[4])
profile_time = float(sys.argv[5])
cache_mode = sys.argv[6]
dataset_file = Path(sys.argv[7])
dataset_kwargs = json.loads(sys.argv[8])

# Plumber serializes tf.py_function/from_generator callback tokens but not the
# Python callback registry. Rebuilding the source graph in this fresh process
# registers the identical callbacks before the optimized graph is restored.
source_module = _import_dataset(dataset_file)
callback_registry_keepalive = source_module.get_dataset(
    TFEvalSpec(
        batch_size=1,
        num_parallel_calls=1,
        num_total_samples=1000,
        kwargs=dataset_kwargs,
    )
)

optimization_start = time.perf_counter()
plumber = tf.data.experimental.analysis.PlumberPerformanceModel(
    str(stats_path.resolve())
)
optimizer = pipeline_optimizer.DataPipelineOptimizer(
    plumber,
    calibrate_system=False,
    step_size=None,
)
optimizer.apply_parallelism()
if cache_mode == "on":
    optimizer.apply_cache(add_take_repeat=False)
dataset = optimizer.instantiate_pipeline()
# Plumber's benchmark helper assumes the first tensor dimension is a batch
# dimension.  Most of these source pipelines yield one unbatched sample, so
# make that assumption explicit; otherwise a [256, T] spectrogram is reported
# as a batch of 256 and global_minibatch_rate becomes 256x too small.
if dataset_file.parent.name != "simclrv2":
    dataset = dataset.batch(1)
options = tf.data.Options()
gen_util.add_analysis_to_dataset_options(options, hard_fail=True)
dataset = dataset.with_options(options)
optimization_time = time.perf_counter() - optimization_start

measurement_start = time.perf_counter()
summary = gen_util.benchmark_dataset(
    dataset,
    time_limit_s=benchmark_seconds,
)
measurement_time = time.perf_counter() - measurement_start
throughput = float(summary["global_minibatch_rate"])

payload = {
    "schema_version": 1,
    "system": "plumber",
    "num_samples": num_samples,
    "profile_time_sec": profile_time,
    "optimization_time_sec": optimization_time,
    "measured_benchmark_time_sec": measurement_time,
    "throughput_samples_per_sec": throughput,
    "normalized_execution_time_sec": (
        num_samples / throughput if throughput > 0 else None
    ),
    "benchmark_summary": {
        key: (
            value.item()
            if hasattr(value, "item")
            else value
        )
        for key, value in summary.items()
        if isinstance(value, (str, int, float, bool)) or hasattr(value, "item")
    },
}
result_path.parent.mkdir(parents=True, exist_ok=True)
result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
  status=$?
  record_command_status "${workload}" "${entity}" "${status}" "${out}"
}

fastflow_supported() {
  case "$1" in
    coco|simclrv2|simclrv2_cache)
      return 0 ;;
    *) return 1 ;;
  esac
}

run_fastflow() {
  local workload="$1" entity="fastflow"
  cell_is_recorded "${workload}" "${entity}" && return
  if ! fastflow_supported "${workload}"; then
    write_status "${workload}" "${entity}" unsupported \
      "opaque Python/Hugging Face callbacks cannot be serialized by tf.data service"
    return
  fi
  if ! container_running "${FASTFLOW_CONTAINER}"; then
    write_status "${workload}" "${entity}" environment_unavailable \
      "set FASTFLOW_CONTAINER to a running pinned FastFlow environment"
    return
  fi

  local base="${workload%_cache}"
  local out="${RUN_ROOT}/systems/${workload}/fastflow.json"
  local container_out="${CONTAINER_RUN_ROOT}/systems/${workload}/fastflow.json"
  local wall="${RUN_ROOT}/systems/${workload}/fastflow.wall_sec"
  local log="${RUN_ROOT}/logs/${workload}__fastflow.log"
  mkdir -p "$(dirname "${out}")"

  echo "[$(date -Is)] RUN ${workload}/fastflow"
  local start_ns end_ns
  start_ns="$(date +%s%N)"
  docker exec \
      -e PYTHONPATH="${CONTAINER_REPO}" \
      -w "${CONTAINER_REPO}" \
      "${FASTFLOW_CONTAINER}" \
      timeout --signal=TERM --kill-after=30s "${CELL_TIMEOUT_SEC}" \
      python evaluation/fastflow/examples/eval_app_runner.py \
        "evaluation/fastflow/workloads/${base}_app.py" \
        "${FASTFLOW_DATASET}" \
        ff \
        evaluation/fastflow/examples/config.yaml \
        --epochs 1 \
        --batch 1 \
        --parallel "${LOCAL_WORKERS}" \
        --num_local_workers "${LOCAL_WORKERS}" \
        --num_samples "${SAMPLES}" \
        --results_path "${container_out}" \
      > "${log}" 2>&1
  local status=$?
  end_ns="$(date +%s%N)"
  elapsed_seconds "${start_ns}" "${end_ns}" > "${wall}"
  record_command_status "${workload}" "${entity}" "${status}" "${out}"
}

cat > "${RUN_ROOT}/metadata.txt" <<EOF
protocol=reduced_single_run_w8_cross_system
reference=evaluation/chapter6_experiments/formal_results/w8_acceptance_latest.md
run_id=${RUN_ID}
selected_workloads=${SELECTED_WORKLOADS}
datajuicer=excluded_by_request
local_workers=${LOCAL_WORKERS}
cpu_budget=${CPU_BUDGET}
repeats=1
batch_size=1
optimizer_timeout_sec=${OPTIMIZER_TIMEOUT_SEC}
cell_timeout_sec=${CELL_TIMEOUT_SEC}
cedar_container=${CEDAR_CONTAINER}
plumber_container=${PLUMBER_CONTAINER:-unavailable}
fastflow_container=${FASTFLOW_CONTAINER:-unavailable}
cedar_optimizers=${CEDAR_OPTIMIZERS[*]}
additional_baseline=cedar_no_optimizer
native_systems=${NATIVE_SYSTEMS[*]}
EOF

echo "[$(date -Is)] Starting cross-system W=8 run ${RUN_ID}"
docker exec "${CEDAR_CONTAINER}" bash -lc "
  cd ${CONTAINER_REPO}
  source env/bin/activate
  if ! ray status --address='${RAY_ADDRESS}' >/dev/null 2>&1; then
    ray start --head --node-ip-address=127.0.0.1 --port=6379 \
      --num-cpus='${CPU_BUDGET}' --disable-usage-stats >/dev/null
  fi
"

for workload in "${WORKLOADS[@]}"; do
  if ! workload_selected "${workload}"; then
    continue
  fi
  set_workload_config "${workload}" || exit 1
  echo "[$(date -Is)] WORKLOAD ${workload} outputs=${SAMPLES}"

  run_cedar_no_optimizer "${workload}"
  for optimizer in "${CEDAR_OPTIMIZERS[@]}"; do
    run_cedar_optimizer "${workload}" "${optimizer}"
  done
  for system in "${NATIVE_SYSTEMS[@]}"; do
    run_native_system "${workload}" "${system}"
  done
  run_plumber "${workload}"
  run_fastflow "${workload}"
done

# Render the complete report inside Cedar's pinned Python environment. Keeping
# this analyzer embedded makes this shell script the only new experiment entry.
cedar_exec python - "${CONTAINER_RUN_ROOT}" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
workloads = {
    "coco": 5000,
    "commonvoice": 10000,
    "commonvoice_cache": 10000,
    "llava_pretrain": 5000,
    "redpajama_c4": 20000,
    "stackexchange": 10000,
    "simclrv2": 9469,
    "simclrv2_cache": 9469,
    "wikitext103": 100000,
    "wikitext103_cache": 100000,
}
optimizers = (
    "optimizer",
    "dp_optimizer",
    "dp_cedar_optimizer",
    "dj_optimizer",
    "pecan_optimizer",
)
entities = (
    "cedar_no_optimizer",
    *(f"cedar_{name}" for name in optimizers),
    "pytorch",
    "tensorflow",
    "ray",
    "plumber",
    "fastflow",
)


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError, TypeError):
        return None


def read_float(path):
    try:
        return float(path.read_text().strip())
    except (OSError, ValueError):
        return None


def status(workload, entity):
    path = root / "status" / workload / f"{entity}.tsv"
    try:
        state, _, reason = path.read_text().rstrip("\n").partition("\t")
        return state, reason
    except OSError:
        return "not_run", "no status was recorded"


def base_record(workload, entity):
    state, reason = status(workload, entity)
    return {
        "entity": entity,
        "status": state,
        "reason": reason,
        "num_samples": None,
        "execution_time_sec": None,
        "throughput_samples_per_sec": None,
        "optimization_or_setup_time_sec": None,
        "profile_or_cache_warmup_time_sec": None,
        "speedup_vs_cedar_no_optimizer": None,
        "speedup_vs_cedar_optimizer": None,
    }


def valid_number(value):
    return (
        isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) >= 0
    )


def load_record(workload, entity, expected):
    record = base_record(workload, entity)
    if record["status"] != "success":
        return record

    if entity == "cedar_no_optimizer":
        payload = read_json(root / "cedar" / workload / "no_optimizer.json")
        wall = read_float(root / "cedar" / workload / "no_optimizer.wall_sec")
        if payload:
            times = payload.get("epoch_run_times", [])
            counts = payload.get("epoch_num_samples", [])
            if len(times) == 1 and len(counts) == 1:
                execution = float(times[0])
                count = int(counts[0])
                record.update(
                    num_samples=count,
                    execution_time_sec=execution,
                    throughput_samples_per_sec=(
                        count / execution if execution > 0 else None
                    ),
                    optimization_or_setup_time_sec=(
                        max(0.0, wall - execution)
                        if wall is not None
                        else None
                    ),
                )
    elif entity.startswith("cedar_"):
        name = entity.removeprefix("cedar_")
        payload = read_json(root / "cedar" / workload / f"{name}.json")
        item = payload["runs"][0] if payload and payload.get("runs") else None
        if item:
            record["optimization_or_setup_time_sec"] = item.get(
                "setup_time_sec"
            )
            record["profile_or_cache_warmup_time_sec"] = item.get(
                "cache_warmup_wall_time_sec", 0.0
            )
            if item.get("optimizer_overhead_too_high") or item.get("timed_out"):
                record["status"] = "optimizer_timeout"
                record["reason"] = (
                    item.get("skip_reason")
                    or item.get("error")
                    or "optimizer overhead exceeded limit"
                )
                return record
            record.update(
                num_samples=item.get("num_samples"),
                execution_time_sec=item.get("perf_time_sec"),
                throughput_samples_per_sec=item.get(
                    "throughput_samples_per_sec"
                ),
            )
    elif entity in {"pytorch", "tensorflow", "ray"}:
        payload = read_json(root / "systems" / workload / f"{entity}.json")
        if payload:
            record.update(
                num_samples=payload.get("num_samples"),
                execution_time_sec=payload.get("measured_time_sec"),
                throughput_samples_per_sec=payload.get(
                    "throughput_samples_per_sec"
                ),
                optimization_or_setup_time_sec=payload.get("setup_time_sec"),
                profile_or_cache_warmup_time_sec=payload.get(
                    "cache_warmup_time_sec"
                ),
            )
            cache_policy = payload.get("entry", {}).get("cache_policy")
            if cache_policy and cache_policy != "not_requested":
                record["reason"] = f"cache_policy={cache_policy}"
    elif entity == "plumber":
        payload = read_json(root / "systems" / workload / "plumber.json")
        if payload:
            record.update(
                num_samples=payload.get("num_samples"),
                execution_time_sec=payload.get(
                    "normalized_execution_time_sec"
                ),
                throughput_samples_per_sec=payload.get(
                    "throughput_samples_per_sec"
                ),
                optimization_or_setup_time_sec=payload.get(
                    "optimization_time_sec"
                ),
                profile_or_cache_warmup_time_sec=payload.get(
                    "profile_time_sec"
                ),
            )
            record["reason"] = (
                "execution time normalized from Plumber's measured "
                "batch-1 steady-state throughput"
            )
    elif entity == "fastflow":
        payload = read_json(root / "systems" / workload / "fastflow.json")
        wall = read_float(root / "systems" / workload / "fastflow.wall_sec")
        if payload and payload.get("epoch_times_sec"):
            execution = float(sum(payload["epoch_times_sec"]))
            record.update(
                num_samples=expected,
                execution_time_sec=execution,
                throughput_samples_per_sec=(
                    expected / execution if execution > 0 else None
                ),
                optimization_or_setup_time_sec=(
                    max(0.0, wall - execution)
                    if wall is not None
                    else None
                ),
            )
            record["reason"] = "setup/auto-offload is total wall time minus epoch time"
            if workload.endswith("_cache"):
                record["reason"] += "; FastFlow has no native cache policy"

    count = record["num_samples"]
    runtime = record["execution_time_sec"]
    throughput = record["throughput_samples_per_sec"]
    if count != expected:
        record["status"] = "invalid"
        record["reason"] = (
            f"processed {count!r} outputs; expected exactly {expected}"
        )
    elif not valid_number(runtime) or float(runtime) <= 0:
        record["status"] = "invalid"
        record["reason"] = "missing or invalid execution time"
    elif not valid_number(throughput) or float(throughput) <= 0:
        record["status"] = "invalid"
        record["reason"] = "missing or invalid throughput"
    return record


report = {
    "schema_version": 1,
    "protocol": {
        "name": "reduced_single_run_w8_cross_system",
        "reference": (
            "evaluation/chapter6_experiments/formal_results/"
            "w8_acceptance_latest.md"
        ),
        "local_workers": 8,
        "cpu_budget": 64,
        "repeats": 1,
        "batch_size": 1,
        "optimizer_timeout_sec": 300,
        "execution_feasibility_timeout_sec": 3600,
        "datajuicer": "excluded_by_request",
        "speedup_definition": "baseline_execution_time / entity_execution_time",
    },
    "workloads": {},
}

for workload, expected in workloads.items():
    records = {
        entity: load_record(workload, entity, expected)
        for entity in entities
    }
    raw_time = records["cedar_no_optimizer"]["execution_time_sec"]
    cedar_time = records["cedar_optimizer"]["execution_time_sec"]
    for record in records.values():
        runtime = record["execution_time_sec"]
        if record["status"] != "success" or not valid_number(runtime) or runtime <= 0:
            continue
        if (
            records["cedar_no_optimizer"]["status"] == "success"
            and valid_number(raw_time)
            and raw_time > 0
        ):
            record["speedup_vs_cedar_no_optimizer"] = raw_time / runtime
        if (
            records["cedar_optimizer"]["status"] == "success"
            and valid_number(cedar_time)
            and cedar_time > 0
        ):
            record["speedup_vs_cedar_optimizer"] = cedar_time / runtime
    report["workloads"][workload] = {
        "expected_samples": expected,
        "entities": records,
    }

optimizer_totals = {}
for optimizer in optimizers:
    entity = f"cedar_{optimizer}"
    records = [
        report["workloads"][workload]["entities"][entity]
        for workload in workloads
    ]
    observed = [
        float(item["optimization_or_setup_time_sec"])
        for item in records
        if valid_number(item["optimization_or_setup_time_sec"])
    ]
    optimizer_totals[optimizer] = {
        "total_optimization_or_setup_time_sec": sum(observed),
        "workloads_with_recorded_time": len(observed),
        "successful_workloads": sum(
            item["status"] == "success" for item in records
        ),
        "optimizer_timeouts": sum(
            item["status"] == "optimizer_timeout" for item in records
        ),
        "other_unavailable_or_failed": sum(
            item["status"] not in {"success", "optimizer_timeout"}
            for item in records
        ),
    }
report["optimizer_totals"] = optimizer_totals


def fmt(value, digits=3):
    return "N/A" if not valid_number(value) else f"{float(value):.{digits}f}"


def escape(value):
    return str(value or "—").replace("|", "\\|").replace("\n", " ")


labels = {
    "cedar_no_optimizer": "Cedar raw plan (no optimizer)",
    "cedar_optimizer": "Cedar optimizer",
    "cedar_dp_optimizer": "DP optimizer",
    "cedar_dp_cedar_optimizer": "DP-Cedar optimizer",
    "cedar_dj_optimizer": "DJ optimizer",
    "cedar_pecan_optimizer": "Pecan optimizer",
    "pytorch": "PyTorch",
    "tensorflow": "tf.data",
    "ray": "Ray Data",
    "plumber": "Plumber",
    "fastflow": "FastFlow",
}

lines = [
    "# Cross-system reduced single-run W=8 results",
    "",
    "Settings match `w8_acceptance_latest.md`: W=8, CPU budget 64, one "
    "measured run, batch size 1, and the same bounded inputs/output counts. "
    "Data-Juicer is excluded.",
    "",
    "Execution time and throughput exclude Cedar optimizer setup and cache "
    "warmup. Plumber reports steady-state throughput and its normalized time "
    "for the target output count. FastFlow epoch time excludes the separately "
    "reported setup/auto-offload estimate.",
    "",
    "Speedup is baseline execution time divided by entity execution time; "
    "values above 1 are faster.",
    "",
    "## Compact time and speedup matrix",
    "",
    "| workload | " + " | ".join(labels[e] for e in entities) + " |",
    "|---|" + "|".join("---:" for _ in entities) + "|",
]
for workload, item in report["workloads"].items():
    cells = []
    for entity in entities:
        record = item["entities"][entity]
        if record["status"] == "success":
            cells.append(
                f"{fmt(record['execution_time_sec'])}s "
                f"({fmt(record['speedup_vs_cedar_no_optimizer'])}×)"
            )
        else:
            cells.append(escape(record["status"]))
    lines.append(f"| {workload} | " + " | ".join(cells) + " |")

for workload, item in report["workloads"].items():
    lines.extend(
        [
            "",
            f"## {workload}",
            "",
            f"Expected outputs: {item['expected_samples']}.",
            "",
            "| system / optimizer | status | execution (s) | throughput "
            "(samples/s) | optimization/setup (s) | profile/cache warmup "
            "(s) | speedup vs raw | speedup vs Cedar | note |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for entity in entities:
        record = item["entities"][entity]
        lines.append(
            f"| {labels[entity]} | {record['status']} | "
            f"{fmt(record['execution_time_sec'])} | "
            f"{fmt(record['throughput_samples_per_sec'])} | "
            f"{fmt(record['optimization_or_setup_time_sec'], 6)} | "
            f"{fmt(record['profile_or_cache_warmup_time_sec'], 6)} | "
            f"{fmt(record['speedup_vs_cedar_no_optimizer'], 4)} | "
            f"{fmt(record['speedup_vs_cedar_optimizer'], 4)} | "
            f"{escape(record['reason'])} |"
        )

lines.extend(
    [
        "",
        "## Cedar optimizer total optimization/setup time",
        "",
        "| optimizer | recorded workloads | successful | optimizer timeout | "
        "other unavailable/failed | total optimization/setup (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
)
for optimizer, item in optimizer_totals.items():
    lines.append(
        f"| {optimizer} | {item['workloads_with_recorded_time']} | "
        f"{item['successful_workloads']} | {item['optimizer_timeouts']} | "
        f"{item['other_unavailable_or_failed']} | "
        f"{fmt(item['total_optimization_or_setup_time_sec'], 6)} |"
    )
lines.append("")

(root / "report.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n"
)
(root / "report.md").write_text("\n".join(lines))
print(root / "report.md")
PY

analysis_status=$?
if [[ "${analysis_status}" -ne 0 ]]; then
  echo "[$(date -Is)] Report generation failed with status ${analysis_status}" >&2
  exit "${analysis_status}"
fi

cp -f "${RUN_ROOT}/report.json" \
  "${FORMAL_DIR}/cross_system_w8_latest.json"
cp -f "${RUN_ROOT}/report.md" \
  "${FORMAL_DIR}/cross_system_w8_latest.md"

echo "[$(date -Is)] COMPLETE ${RUN_ID}"
echo "Run directory: ${RUN_ROOT}"
echo "Markdown report: ${RUN_ROOT}/report.md"
echo "Latest report: ${FORMAL_DIR}/cross_system_w8_latest.md"
