#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PLUMBER_CONTAINER="${PLUMBER_CONTAINER:-optimalcedar-plumber}"
FASTFLOW_CONTAINER="${FASTFLOW_CONTAINER:-optimalcedar-fastflow}"

if [[ "$#" -lt 2 ]]; then
  echo "usage: run_external.sh SYSTEM WORKLOAD [DATASET_PATH] [RESULTS_PATH]" >&2
  exit 2
fi

SYSTEM="$1"
WORKLOAD="$2"
DATASET_PATH="${3:-}"
RESULTS_PATH="${4:-${REPO_ROOT}/evaluation/baselines/results/${SYSTEM}/${WORKLOAD}.json}"
RESULTS_PATH="$(realpath -m "${RESULTS_PATH}")"
if [[ "${RESULTS_PATH}" != "${REPO_ROOT}/"* ]]; then
  echo "RESULTS_PATH must be inside ${REPO_ROOT} so all containers can write it" >&2
  exit 2
fi
mkdir -p "$(dirname "${RESULTS_PATH}")"
RESULTS_REL="${RESULTS_PATH#${REPO_ROOT}/}"
CONTAINER_RESULTS="/workspace/OptimalCedar/${RESULTS_REL}"
LOG_PATH="${RESULTS_PATH%.json}.log"
STARTED_AT="$(date +%s.%N)"
ARTIFACT_PATH=""

case "${SYSTEM}" in
  datajuicer)
    case "${WORKLOAD}" in
      llava_pretrain|redpajama_c4|stackexchange)
        ;;
      *)
        echo "Data-Juicer has no native recipe for ${WORKLOAD}" >&2
        exit 2
        ;;
    esac
    bash "${SCRIPT_DIR}/bootstrap_datajuicer.sh"
    CONFIG="/optimalcedar-configs/${WORKLOAD}.yaml"
    ARTIFACT_PATH="${RESULTS_PATH%.json}.dataset.jsonl"
    ARTIFACT_REL="${ARTIFACT_PATH#${REPO_ROOT}/}"
    CONTAINER_OUTPUT="/workspace/OptimalCedar/${ARTIFACT_REL}"
    args=(
      docker compose -f "${REPO_ROOT}/docker-compose.baselines.yml"
      run --rm datajuicer
      dj-process --config "${CONFIG}"
      --export_path "${CONTAINER_OUTPUT}"
    )
    if [[ -n "${DATASET_PATH}" ]]; then
      args+=(--dataset_path "${DATASET_PATH}")
    fi
    "${args[@]}" 2>&1 | tee "${LOG_PATH}"
    ;;
  plumber)
    OUTPUT_DIR="$(dirname "${CONTAINER_RESULTS}")/plumber_artifacts"
    ARTIFACT_PATH="$(dirname "${RESULTS_PATH}")/plumber_artifacts/result.json"
    DATASET_KWARGS="{}"
    if [[ -n "${DATASET_PATH}" ]]; then
      DATASET_KWARGS="{\"dataset_path\":\"${DATASET_PATH}\"}"
    fi
    docker exec "${PLUMBER_CONTAINER}" bash -lc "
      cd /workspace/OptimalCedar
      PYTHONPATH=/workspace/OptimalCedar \
        bash evaluation/plumber/run_workload.sh \
          '${WORKLOAD}' '${OUTPUT_DIR}' '${DATASET_KWARGS}'
    " 2>&1 | tee "${LOG_PATH}"
    ;;
  fastflow)
    BASE_WORKLOAD="${WORKLOAD%_cache}"
    APP="/workspace/OptimalCedar/evaluation/fastflow/workloads/${BASE_WORKLOAD}_app.py"
    ARTIFACT_PATH="${RESULTS_PATH%.json}.fastflow.json"
    ARTIFACT_REL="${ARTIFACT_PATH#${REPO_ROOT}/}"
    CONTAINER_ARTIFACT="/workspace/OptimalCedar/${ARTIFACT_REL}"
    docker exec "${FASTFLOW_CONTAINER}" bash -lc "
      cd /workspace/OptimalCedar
      PYTHONPATH=/workspace/OptimalCedar \
        python evaluation/fastflow/examples/eval_app_runner.py \
          '${APP}' '${DATASET_PATH}' ff \
          evaluation/fastflow/examples/config.yaml \
          --epochs 1 --batch 1 \
          --results_path '${CONTAINER_ARTIFACT}'
    " 2>&1 | tee "${LOG_PATH}"
    ;;
  *)
    echo "External runner supports datajuicer, plumber, or fastflow." >&2
    exit 2
    ;;
esac

docker exec optimalcedar-torch201-dev bash -lc '
  cd /workspace/OptimalCedar
  source env/bin/activate
  python evaluation/baselines/record_external_result.py "$@"
' bash \
  --system "${SYSTEM}" \
  --workload "${WORKLOAD}" \
  --started-at "${STARTED_AT}" \
  --artifact "/workspace/OptimalCedar/${ARTIFACT_PATH#${REPO_ROOT}/}" \
  --log "/workspace/OptimalCedar/${LOG_PATH#${REPO_ROOT}/}" \
  --results-path "${CONTAINER_RESULTS}"
