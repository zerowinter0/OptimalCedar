#!/usr/bin/env bash
# Re-join the local machine (172.23.166.103) to the remote Ray head with a
# raised file-descriptor limit, mirroring the previous topology (64 CPUs plus
# the local GPU node label the earlier cluster exposed).
set -euo pipefail

ulimit -n 524288
ray stop --force >/dev/null 2>&1 || true
sleep 2
ray start \
  --address=172.23.166.105:6379 \
  --num-cpus=64 \
  --resources='{"accelerator_type:RTX": 1, "node:__internal_head_local__": 1}' \
  --object-store-memory=65000000000
echo "ulimit -n = $(ulimit -n)"
