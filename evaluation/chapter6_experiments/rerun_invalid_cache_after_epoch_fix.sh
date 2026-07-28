#!/usr/bin/env bash
set -euo pipefail

REPO="/home/xieruiyang/OptimalCedar"
CONTAINER_REPO="/workspace/OptimalCedar"
CEDAR_CONTAINER="${CEDAR_CONTAINER:-optimalcedar-torch201-dev}"
RUN_ID="${RUN_ID:-cross_system_w8_formal_20260724_164724}"
RUN_ROOT="${REPO}/evaluation/chapter6_experiments/formal_results/cross_system_w8_runs/${RUN_ID}"
BACKUP="${RUN_ROOT}/pre_epoch_boundary_fix_$(date +%Y%m%d_%H%M%S)"

if [[ ! -d "${RUN_ROOT}" ]]; then
  echo "Run directory does not exist: ${RUN_ROOT}" >&2
  exit 1
fi

mkdir -p \
  "${BACKUP}/cedar/commonvoice_cache" \
  "${BACKUP}/cedar/simclrv2_cache" \
  "${BACKUP}/status/commonvoice_cache" \
  "${BACKUP}/status/simclrv2_cache" \
  "${BACKUP}/logs" \
  "${BACKUP}/cache"

cp -f "${RUN_ROOT}/report.json" "${BACKUP}/report.json"
cp -f "${RUN_ROOT}/report.md" "${BACKUP}/report.md"

move_into_backup() {
  local host_source="$1"
  local host_destination="$2"
  local container_source="${host_source/#"${REPO}"/"${CONTAINER_REPO}"}"
  local container_destination="${host_destination/#"${REPO}"/"${CONTAINER_REPO}"}"

  docker exec "${CEDAR_CONTAINER}" bash -lc '
    source_path="$1"
    destination_path="$2"
    if [[ -e "${source_path}" ]]; then
      mkdir -p "$(dirname "${destination_path}")"
      mv "${source_path}" "${destination_path}"
    fi
  ' bash "${container_source}" "${container_destination}"
}

archive_optimizer_cell() {
  local workload="$1"
  local optimizer="$2"
  local entity="cedar_${optimizer}"
  local path

  for path in \
    "${RUN_ROOT}/cedar/${workload}/${optimizer}.json" \
    "${RUN_ROOT}/status/${workload}/${entity}.tsv" \
    "${RUN_ROOT}/logs/${workload}__${entity}.log"; do
    move_into_backup "${path}" "${BACKUP}/${path#"${RUN_ROOT}/"}"
  done

  # Cache namespaces omit the workload's "_cache" suffix.
  local cache_workload="${workload%_cache}"
  path="${RUN_ROOT}/cache/${cache_workload}__${optimizer}"
  move_into_backup "${path}" "${BACKUP}/cache/$(basename "${path}")"
}

for optimizer in \
  optimizer dp_optimizer dp_cedar_optimizer dj_optimizer pecan_optimizer; do
  archive_optimizer_cell commonvoice_cache "${optimizer}"
done

for optimizer in \
  optimizer dp_cedar_optimizer dj_optimizer pecan_optimizer; do
  archive_optimizer_cell simclrv2_cache "${optimizer}"
done

echo "[$(date -Is)] Archived invalid cells under ${BACKUP}"
cd "${REPO}"
exec bash evaluation/chapter6_experiments/run_cross_system_w8_matrix.sh \
  --run-id "${RUN_ID}" \
  --workloads commonvoice_cache,simclrv2_cache \
  --resume
