#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${REPO_ROOT}/evaluation/chapter6_experiments/container_setup_logs"
STATUS_FILE="${LOG_DIR}/fastflow_setup.status"
TF_ARCHIVE="fastflow-tensorflow-8dc4caf647dc.tar.gz"
TF_ARCHIVE_PATH="${SCRIPT_DIR}/sources/${TF_ARCHIVE}"
TF_ARCHIVE_SHA256="9d0f2eb1dd1d88c1f9b1c2218eb9236dffc6f5de326a0d57cc19c07314bc0d3f"
APP_ARCHIVE_PATH="${SCRIPT_DIR}/sources/fastflow-f2e3a3363e95.tar.gz"
APP_ARCHIVE_SHA256="a2d2281377130074586384b3d4ea2e61e6f5d777e055d67c344b249d54e7cd3a"
VPN_HTTP_PROXY="${VPN_HTTP_PROXY:-http://127.0.0.1:17890}"
VPN_ALL_PROXY="${VPN_ALL_PROXY:-socks5h://127.0.0.1:17890}"
VPN_NO_PROXY="${VPN_NO_PROXY:-localhost,127.0.0.1,::1,pypi.org,files.pythonhosted.org}"

mkdir -p "${LOG_DIR}"

write_status() {
  local state="$1"
  local detail="$2"
  {
    printf 'state=%s\n' "${state}"
    printf 'updated_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'detail=%s\n' "${detail}"
  } >"${STATUS_FILE}.tmp"
  mv "${STATUS_FILE}.tmp" "${STATUS_FILE}"
}

on_error() {
  local exit_code="$?"
  local line_no="${BASH_LINENO[0]:-unknown}"
  write_status FAILED "exit=${exit_code}; line=${line_no}; inspect fastflow_setup.log"
  exit "${exit_code}"
}
trap on_error ERR

write_status RUNNING "validating pinned source archives"
echo "[$(date --iso-8601=seconds)] Validating source archives"
echo "${TF_ARCHIVE_SHA256}  ${TF_ARCHIVE_PATH}" | sha256sum -c -
echo "${APP_ARCHIVE_SHA256}  ${APP_ARCHIVE_PATH}" | sha256sum -c -
tar -tzf "${TF_ARCHIVE_PATH}" >/dev/null
tar -tzf "${APP_ARCHIVE_PATH}" >/dev/null

write_status RUNNING "building FastFlow TensorFlow 2.7"
echo "[$(date --iso-8601=seconds)] Building FastFlow TensorFlow"
DOCKER_BUILDKIT=1 docker build --progress=plain \
  --network=host \
  -f "${SCRIPT_DIR}/tensorflow-builder.Dockerfile" \
  --build-arg "HTTP_PROXY=${VPN_HTTP_PROXY}" \
  --build-arg "HTTPS_PROXY=${VPN_HTTP_PROXY}" \
  --build-arg "ALL_PROXY=${VPN_ALL_PROXY}" \
  --build-arg "NO_PROXY=${VPN_NO_PROXY}" \
  --build-arg "TF_ARCHIVE=${TF_ARCHIVE}" \
  --build-arg "TF_ARCHIVE_SHA256=${TF_ARCHIVE_SHA256}" \
  -t optimalcedar-tensorflow-builder:fastflow-8dc4caf647dc \
  "${SCRIPT_DIR}"

write_status RUNNING "building FastFlow runtime image"
echo "[$(date --iso-8601=seconds)] Building FastFlow runtime image"
DOCKER_BUILDKIT=1 docker build --progress=plain \
  --network=host \
  -f "${SCRIPT_DIR}/fastflow.Dockerfile" \
  --build-arg "HTTP_PROXY=${VPN_HTTP_PROXY}" \
  --build-arg "HTTPS_PROXY=${VPN_HTTP_PROXY}" \
  --build-arg "ALL_PROXY=${VPN_ALL_PROXY}" \
  --build-arg "NO_PROXY=${VPN_NO_PROXY}" \
  -t optimalcedar-fastflow:tf2.7-f2e3a3363e95 \
  "${SCRIPT_DIR}"

write_status RUNNING "starting and validating FastFlow container"
echo "[$(date --iso-8601=seconds)] Starting FastFlow container"
docker compose \
  -f "${REPO_ROOT}/docker-compose.external-baselines.yml" \
  up -d --force-recreate fastflow

echo "[$(date --iso-8601=seconds)] Validating FastFlow runtime"
docker exec -i optimalcedar-fastflow python - <<'PY'
import inspect
import os
from pathlib import Path

import fastflow
import tensorflow as tf

assert os.cpu_count() == 64, os.cpu_count()
assert Path("/workspace/OptimalCedar").is_dir()
gpus = tf.config.list_physical_devices("GPU")
assert len(gpus) == 1, gpus
params = inspect.signature(tf.data.experimental.service.distribute).parameters
assert "partial_offload_enabled" in params, tuple(params)
assert "ratio_local" in params, tuple(params)
print("fastflow_import=ok")
print(f"tensorflow_version={tf.__version__}")
print(f"cpu_count={os.cpu_count()}")
print(f"gpu_count={len(gpus)}")
print("custom_distribute_parameters=ok")
PY

docker exec optimalcedar-fastflow bash -lc '
  set -euo pipefail
  test "$(df --output=size -B1 /dev/shm | tail -1 | tr -d " ")" -ge 68000000000
  test "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -eq 1
  test "${HTTP_PROXY}" = "http://127.0.0.1:17890"
  curl -fsSIL --max-time 30 https://github.com/ >/dev/null
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  df -h /dev/shm
  echo "vpn_connectivity=ok"
'

trap - ERR
write_status SUCCEEDED "FastFlow image built, container running, runtime validation passed"
echo "[$(date --iso-8601=seconds)] FastFlow setup completed successfully"
