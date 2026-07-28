#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VPN_HTTP_PROXY="${VPN_HTTP_PROXY:-http://127.0.0.1:17890}"
VPN_ALL_PROXY="${VPN_ALL_PROXY:-socks5h://127.0.0.1:17890}"
VPN_NO_PROXY="${VPN_NO_PROXY:-localhost,127.0.0.1,::1,pypi.org,files.pythonhosted.org}"

bash "${SCRIPT_DIR}/prepare_sources.sh"

build_tf() {
  local system="$1"
  local archive="$2"
  local sha256="$3"
  local tag="$4"
  DOCKER_BUILDKIT=1 docker build --progress=plain \
    --network=host \
    -f "${SCRIPT_DIR}/tensorflow-builder.Dockerfile" \
    --build-arg "HTTP_PROXY=${VPN_HTTP_PROXY}" \
    --build-arg "HTTPS_PROXY=${VPN_HTTP_PROXY}" \
    --build-arg "ALL_PROXY=${VPN_ALL_PROXY}" \
    --build-arg "NO_PROXY=${VPN_NO_PROXY}" \
    --build-arg "TF_ARCHIVE=${archive}" \
    --build-arg "TF_ARCHIVE_SHA256=${sha256}" \
    -t "${tag}" \
    "${SCRIPT_DIR}"
}

build_tf \
  plumber \
  plumber-tensorflow-08bf144ec13b.tar.gz \
  93da08fdd73fff1c423c0db5c0eeefe6cc3b47d87171010b9acdd48e50856a20 \
  optimalcedar-tensorflow-builder:plumber-08bf144ec13b

DOCKER_BUILDKIT=1 docker build --progress=plain \
  --network=host \
  -f "${SCRIPT_DIR}/plumber.Dockerfile" \
  --build-arg "HTTP_PROXY=${VPN_HTTP_PROXY}" \
  --build-arg "HTTPS_PROXY=${VPN_HTTP_PROXY}" \
  --build-arg "ALL_PROXY=${VPN_ALL_PROXY}" \
  --build-arg "NO_PROXY=${VPN_NO_PROXY}" \
  -t optimalcedar-plumber:tf2.7-6123f5bce36e \
  "${SCRIPT_DIR}"

FASTFLOW_TF_ARCHIVE="fastflow-tensorflow-8dc4caf647dc.tar.gz"
FASTFLOW_TF_SHA256="9d0f2eb1dd1d88c1f9b1c2218eb9236dffc6f5de326a0d57cc19c07314bc0d3f"
build_tf \
  fastflow \
  "${FASTFLOW_TF_ARCHIVE}" \
  "${FASTFLOW_TF_SHA256}" \
  optimalcedar-tensorflow-builder:fastflow-8dc4caf647dc

DOCKER_BUILDKIT=1 docker build --progress=plain \
  --network=host \
  -f "${SCRIPT_DIR}/fastflow.Dockerfile" \
  --build-arg "HTTP_PROXY=${VPN_HTTP_PROXY}" \
  --build-arg "HTTPS_PROXY=${VPN_HTTP_PROXY}" \
  --build-arg "ALL_PROXY=${VPN_ALL_PROXY}" \
  --build-arg "NO_PROXY=${VPN_NO_PROXY}" \
  -t optimalcedar-fastflow:tf2.7-f2e3a3363e95 \
  "${SCRIPT_DIR}"

docker compose \
  -f "${REPO_ROOT}/docker-compose.external-baselines.yml" \
  up -d --force-recreate plumber fastflow
