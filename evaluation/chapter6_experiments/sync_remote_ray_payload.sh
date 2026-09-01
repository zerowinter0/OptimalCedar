#!/usr/bin/env bash
# Synchronize only the runtime assets that remote Ray actors may open.

set -euo pipefail

LOCAL_CONTAINER="${LOCAL_CONTAINER:-optimalcedar-torch201-dev}"
REMOTE_SSH="${REMOTE_SSH:-xieruiyang@172.23.166.105}"
REMOTE_CONTAINER="${REMOTE_CONTAINER:-optimalcedar-ray-remote}"
SETUP_ROOT="${SETUP_ROOT:-/home/xieruiyang/remote_ray_setup}"
COMPLETE_MARKER="${SETUP_ROOT}/PAYLOAD_COMPLETE"
TRANSFER_RETRIES="${TRANSFER_RETRIES:-3}"
SSH_OPTIONS=(
  -o BatchMode=yes
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=20
)

mkdir -p "${SETUP_ROOT}"
if [[ -e "${COMPLETE_MARKER}" ]]; then
  echo "Refusing to overwrite existing completion marker: ${COMPLETE_MARKER}" >&2
  exit 2
fi

stamp() {
  date -Is
}

stream_archive() {
  local source_root="$1"
  local destination_root="$2"
  shift 2
  local attempt
  local pipe_status

  for ((attempt = 1; attempt <= TRANSFER_RETRIES; attempt++)); do
    echo "[$(stamp)] transfer attempt ${attempt}/${TRANSFER_RETRIES}: $*"
    set +e
    docker exec "${LOCAL_CONTAINER}" tar -C "${source_root}" -cf - "$@" |
      ssh "${SSH_OPTIONS[@]}" "${REMOTE_SSH}" \
        "docker exec -i ${REMOTE_CONTAINER} tar -C ${destination_root} -xf -"
    pipe_status=("${PIPESTATUS[@]}")
    set -e
    echo "[$(stamp)] transfer exit codes: local_tar=${pipe_status[0]} remote_ssh_tar=${pipe_status[1]}"
    if [[ "${pipe_status[0]}" -eq 0 && "${pipe_status[1]}" -eq 0 ]]; then
      return 0
    fi
  done

  echo "[$(stamp)] transfer failed after ${TRANSFER_RETRIES} attempts: $*" >&2
  return 1
}

echo "[$(stamp)] START remote environment payload transfer"
echo "[$(stamp)] CACHE /root/.cache"
stream_archive /root /root .cache
echo "[$(stamp)] CACHE complete"

echo "[$(stamp)] MEDIA commonvoice, general_video_refine, imagenette2"
stream_archive /workspace/OptimalCedar /workspace/OptimalCedar \
  datasets/commonvoice \
  datasets/general_video_refine \
  evaluation/datasets/commonvoice \
  evaluation/datasets/imagenette2
echo "[$(stamp)] MEDIA complete"

ssh "${REMOTE_SSH}" "docker exec ${REMOTE_CONTAINER} du -sh \
  /root/.cache \
  /workspace/OptimalCedar/datasets/commonvoice \
  /workspace/OptimalCedar/datasets/general_video_refine \
  /workspace/OptimalCedar/evaluation/datasets/commonvoice \
  /workspace/OptimalCedar/evaluation/datasets/imagenette2"

touch "${COMPLETE_MARKER}"
echo "[$(stamp)] COMPLETE remote environment payload transfer"
