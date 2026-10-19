#!/usr/bin/env bash
set -euo pipefail

: "${NNODES:?Set NNODES to the number of nodes}"
: "${NODE_RANK:?Set NODE_RANK uniquely from 0 to NNODES-1}"
: "${MASTER_ADDR:?Set MASTER_ADDR to the rank-zero node hostname}"
: "${MASTER_PORT:?Set a free MASTER_PORT reachable from all nodes}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec torchrun --nnodes="$NNODES" --node-rank="$NODE_RANK" \
  --nproc-per-node="${GPUS_PER_NODE:-8}" \
  --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
  --module video_rl_pair.train "$@"
