#!/usr/bin/env bash
set -euo pipefail

# This watchdog intentionally runs on the host: the Codex CLI is installed on
# the host, while AGENTS.md requires project commands to run in Docker.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXPERIMENT_LOG="${EXPERIMENT_LOG:-${REPO_ROOT}/evaluation/chapter6_experiments/formal_experiments.log}"
WATCHDOG_LOG="${WATCHDOG_LOG:-${REPO_ROOT}/evaluation/chapter6_experiments/formal_experiment_watchdog.log}"
STATE_DIR="${WATCHDOG_STATE_DIR:-${REPO_ROOT}/evaluation/chapter6_experiments/.watchdog}"
INTERVAL_SECONDS="${WATCHDOG_INTERVAL_SECONDS:-900}"
CODEX_MODEL="${CODEX_MODEL:-gpt-5.6-sol}"
CODEX_REASONING_EFFORT="${CODEX_REASONING_EFFORT:-medium}"
CONTAINER_NAME="${CONTAINER_NAME:-optimalcedar-torch201-dev}"
CODEX_TIMEOUT_SECONDS="${CODEX_TIMEOUT_SECONDS:-3600}"

SNAPSHOT_LOG="${STATE_DIR}/previous_formal_experiments.log"
SNAPSHOT_STATUS="${STATE_DIR}/previous_formal_experiments.status"
LOCK_FILE="${STATE_DIR}/watchdog.lock"

if ! [[ "${INTERVAL_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "WATCHDOG_INTERVAL_SECONDS must be a positive integer" >&2
  exit 2
fi

mkdir -p "${STATE_DIR}"
if ! [[ "${CODEX_TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "CODEX_TIMEOUT_SECONDS must be a positive integer" >&2
  exit 2
fi
touch "${WATCHDOG_LOG}"

# Let flock own the lock in a wrapper process and close its descriptor before
# launching this script. Neither sleep nor Codex can then inherit the lock.
if [[ "${WATCHDOG_LOCK_HELD:-0}" != "1" ]]; then
  export WATCHDOG_LOCK_HELD=1
  exec flock --close --nonblock --conflict-exit-code 0 \
    "${LOCK_FILE}" "${BASH_SOURCE[0]}" "$@"
fi

log_message() {
  echo "[$(date -Is)] $*" >>"${WATCHDOG_LOG}"
}

current_status() {
  if [[ -f "${EXPERIMENT_LOG}" ]]; then
    printf 'present\n'
  else
    printf 'missing\n'
  fi
}

refresh_snapshot() {
  local status tmp_log tmp_status
  status="$(current_status)"
  tmp_log="${SNAPSHOT_LOG}.tmp.$$"
  tmp_status="${SNAPSHOT_STATUS}.tmp.$$"

  if [[ "${status}" == "present" ]]; then
    cp -- "${EXPERIMENT_LOG}" "${tmp_log}"
  else
    : >"${tmp_log}"
  fi
  printf '%s\n' "${status}" >"${tmp_status}"
  mv -f -- "${tmp_log}" "${SNAPSHOT_LOG}"
  mv -f -- "${tmp_status}" "${SNAPSHOT_STATUS}"
}

log_is_unchanged() {
  local status previous_status
  status="$(current_status)"
  [[ -f "${SNAPSHOT_STATUS}" && -f "${SNAPSHOT_LOG}" ]] || return 1
  previous_status="$(<"${SNAPSHOT_STATUS}")"
  [[ "${status}" == "${previous_status}" ]] || return 1

  if [[ "${status}" == "missing" ]]; then
    return 0
  fi
  cmp -s -- "${EXPERIMENT_LOG}" "${SNAPSHOT_LOG}"
}

experiment_completed() {
  [[ -f "${EXPERIMENT_LOG}" ]] && grep -Fq "Formal matrix completed" "${EXPERIMENT_LOG}"
}

invoke_codex() {
  local codex_bin
  codex_bin="${CODEX_BIN:-$(command -v codex || true)}"
  if [[ -z "${codex_bin}" ]]; then
    log_message "ERROR: Codex CLI was not found on the host PATH."
    return 127
  fi

  local prompt
  prompt="你正在实验室服务器上维护 OptimalCedar 的 chapter 6 正式实验。固定实验日志 ${EXPERIMENT_LOG} 已连续 ${INTERVAL_SECONDS} 秒逐字节完全没有变化（或持续不存在），watchdog 因此触发本次检查。

请先完整阅读 ${REPO_ROOT}/AGENTS.md，并严格遵守其中的实验矩阵、公平性、并行度和长实验运行要求。项目宿主机目录 ${REPO_ROOT} 已整体挂载到 Docker 容器 ${CONTAINER_NAME} 的 /workspace/OptimalCedar。任何项目代码/实验命令都必须通过 docker exec 进入该容器，cd /workspace/OptimalCedar，并先 source env/bin/activate。

请检查固定日志、宿主机与容器内实验进程、Ray 状态、CPU/内存/共享内存/磁盘等，判断当前实验是否正常。日志在合理的长计算阶段不更新不等于异常，必须结合进程和资源活动判断。

如果实验确实正常：不要修改任何代码或配置，不要停止或重启任何进程，只给出简短结论后结束。

如果实验异常：独立定位根因或 bug，按正式顶会实验标准修复并做必要验证；不得通过降低样本量、并行度、优化器/负载覆盖或其他削弱实验公平性的方式绕过问题。随后停止残留的异常实验进程，按照仓库当前正式实验入口用 nohup 重启完整实验，并覆盖写入同一个固定日志 ${EXPERIMENT_LOG}。不要另建正式实验日志。无需询问用户，完成修复和重启后再结束。"

  log_message "Log is unchanged; invoking Codex model=${CODEX_MODEL}, reasoning=${CODEX_REASONING_EFFORT}, timeout=${CODEX_TIMEOUT_SECONDS}s."
  if printf '%s\n' "${prompt}" | timeout --signal=TERM --kill-after=60s "${CODEX_TIMEOUT_SECONDS}s" \
      "${codex_bin}" exec \
      --ephemeral \
      --color never \
      --dangerously-bypass-approvals-and-sandbox \
      --model "${CODEX_MODEL}" \
      --config "model_reasoning_effort=\"${CODEX_REASONING_EFFORT}\"" \
      --cd "${REPO_ROOT}" \
      - >>"${WATCHDOG_LOG}" 2>&1; then
    log_message "Codex inspection finished successfully."
  else
    local rc=$?
    if [[ "${rc}" -eq 124 ]]; then
      log_message "ERROR: Codex inspection exceeded ${CODEX_TIMEOUT_SECONDS}s and was terminated; will retry if the log remains unchanged."
    else
      log_message "ERROR: Codex inspection exited with status ${rc}; will retry if the log remains unchanged."
    fi
    return "${rc}"
  fi
}

log_message "Watchdog started: interval=${INTERVAL_SECONDS}s, experiment_log=${EXPERIMENT_LOG}."
refresh_snapshot

while sleep "${INTERVAL_SECONDS}"; do
  if experiment_completed; then
    log_message "Formal matrix completion marker found; watchdog exiting."
    exit 0
  fi

  if log_is_unchanged; then
    invoke_codex || true
  fi

  # Codex may have overwritten the experiment log while repairing/restarting it.
  refresh_snapshot
done
