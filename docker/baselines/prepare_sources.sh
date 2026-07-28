#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCES_DIR="${SCRIPT_DIR}/sources"
mkdir -p "${SOURCES_DIR}"

download_locked() {
  local name="$1"
  local url="$2"
  local expected_sha256="$3"
  local path="${SOURCES_DIR}/${name}"

  if [[ ! -f "${path}" ]]; then
    curl -fL --retry 5 --retry-delay 3 -o "${path}.part" "${url}"
    mv "${path}.part" "${path}"
  fi
  echo "${expected_sha256}  ${path}" | sha256sum -c -
}

download_locked \
  bazel-3.7.2-linux-x86_64 \
  https://releases.bazel.build/3.7.2/release/bazel-3.7.2-linux-x86_64 \
  70dc0bee198a4c3d332925a32d464d9036a831977501f66d4996854ad4e4fc0d
chmod 0755 "${SOURCES_DIR}/bazel-3.7.2-linux-x86_64"

download_locked \
  plumber-tensorflow-08bf144ec13b.tar.gz \
  https://github.com/mkuchnik/PlumberTensorflow/archive/08bf144ec13b0c27f2a02aaba975546506ee0f6a.tar.gz \
  93da08fdd73fff1c423c0db5c0eeefe6cc3b47d87171010b9acdd48e50856a20

download_locked \
  plumber-app-6123f5bce36e.tar.gz \
  https://github.com/mkuchnik/PlumberApp/archive/6123f5bce36eec7dc75b6b9298054b493d930bdc.tar.gz \
  e799ab4b559361e4f5dac1950a2d207b0b3afc21386d1ae04e1555f5aeed9a34

download_locked \
  fastflow-f2e3a3363e95.tar.gz \
  https://archive.softwareheritage.org/api/1/vault/flat/swh:1:dir:85cbfadf61d642f3c4fd914fd84564182a1b5c45/raw/ \
  a2d2281377130074586384b3d4ea2e61e6f5d777e055d67c344b249d54e7cd3a

FASTFLOW_TF_SWHID="swh:1:dir:8fe31ad60626b9bcd9ce2917c8458e8fdec0f1e8"
FASTFLOW_TF_PATH="${SOURCES_DIR}/fastflow-tensorflow-8dc4caf647dc.tar.gz"
FASTFLOW_TF_ENDPOINT="https://archive.softwareheritage.org/api/1/vault/flat/${FASTFLOW_TF_SWHID}/"
FASTFLOW_TF_SHA256="9d0f2eb1dd1d88c1f9b1c2218eb9236dffc6f5de326a0d57cc19c07314bc0d3f"

swh_curl() {
  env \
    -u http_proxy -u https_proxy -u all_proxy \
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    curl "$@"
}

if [[ ! -f "${FASTFLOW_TF_PATH}" ]]; then
  swh_curl -fsSL -X POST "${FASTFLOW_TF_ENDPOINT}" >/dev/null
  while true; do
    status="$(
      swh_curl -fsSL "${FASTFLOW_TF_ENDPOINT}" |
        python3 -c 'import json,sys; print(json.load(sys.stdin)["status"])'
    )"
    case "${status}" in
      done)
        swh_curl -fL --retry 5 --retry-delay 3 \
          -o "${FASTFLOW_TF_PATH}.part" \
          "${FASTFLOW_TF_ENDPOINT}raw/"
        mv "${FASTFLOW_TF_PATH}.part" "${FASTFLOW_TF_PATH}"
        break
        ;;
      failed)
        echo "Software Heritage failed to cook ${FASTFLOW_TF_SWHID}" >&2
        exit 1
        ;;
      *)
        echo "Waiting for FastFlow TensorFlow archive (${status})..."
        sleep 30
        ;;
    esac
  done
fi

echo "${FASTFLOW_TF_SHA256}  ${FASTFLOW_TF_PATH}" | sha256sum -c -
