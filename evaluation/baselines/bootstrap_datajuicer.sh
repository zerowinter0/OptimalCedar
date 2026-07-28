#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CHECKOUT="${REPO_ROOT}/data-juicer"
REPOSITORY="https://github.com/datajuicer/data-juicer.git"
COMMIT="bb3d88aac183cc22b6f816262a812a9e5d5abb57"

if [[ ! -e "${CHECKOUT}" ]]; then
  git clone "${REPOSITORY}" "${CHECKOUT}"
  git -C "${CHECKOUT}" checkout --detach "${COMMIT}"
fi

if [[ ! -d "${CHECKOUT}/.git" ]]; then
  echo "${CHECKOUT} exists but is not a Git checkout" >&2
  exit 1
fi

actual_commit="$(git -c safe.directory="${CHECKOUT}" -C "${CHECKOUT}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${COMMIT}" ]]; then
  echo "Data-Juicer revision mismatch." >&2
  echo "expected: ${COMMIT}" >&2
  echo "actual:   ${actual_commit}" >&2
  echo "Refusing to modify an existing checkout." >&2
  exit 1
fi

if [[ -n "$(git -c safe.directory="${CHECKOUT}" -C "${CHECKOUT}" status --short)" ]]; then
  echo "Data-Juicer checkout is dirty; refusing an irreproducible build." >&2
  exit 1
fi

echo "Data-Juicer checkout verified at ${COMMIT}"
