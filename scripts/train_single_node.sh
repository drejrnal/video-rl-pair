#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec torchrun --rdzv-backend=c10d --rdzv-endpoint=127.0.0.1:0 \
  --local-addr=127.0.0.1 --nnodes=1 --nproc-per-node="${GPUS_PER_NODE:-8}" \
  --module video_rl_pair.train "$@"
