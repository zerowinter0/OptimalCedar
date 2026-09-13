#!/usr/bin/env bash
# Restart the remote Ray head on 172.23.166.105 with a raised file-descriptor
# limit.  The previous raylet died because its runtime_env_agent hit
# `OSError: [Errno 24] Too many open files` (the container's default limit is
# 1024), and Ray fate-shares the raylet with its agents.
set -euo pipefail

ulimit -n 524288
ray stop --force >/dev/null 2>&1 || true
sleep 2
ray start --head \
  --port=6379 \
  --node-ip-address=172.23.166.105 \
  --num-cpus=64 \
  --resources='{"cedar_remote": 1}' \
  --object-store-memory=65000000000 \
  --dashboard-host=0.0.0.0 \
  --include-dashboard=true
echo "ulimit -n = $(ulimit -n)"
