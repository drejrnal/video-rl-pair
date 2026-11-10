#!/usr/bin/env bash
set -euo pipefail

: "${NNODES:?Set NNODES to the number of nodes}"
: "${NODE_RANK:?Set NODE_RANK uniquely from 0 to NNODES-1}"
: "${MASTER_ADDR:?Set MASTER_ADDR to the rank-zero node hostname}"
: "${MASTER_PORT:?Set a free MASTER_PORT reachable from all nodes}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python_bin="${PYTHON:-.venv/bin/python}"
if [[ -z "${PYTHON:-}" && ! -x "$python_bin" ]]; then
  python_bin=python
fi
exec "$python_bin" -m torch.distributed.run \
  --nnodes="$NNODES" --node-rank="$NODE_RANK" \
  --nproc-per-node="${GPUS_PER_NODE:-1}" \
  --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
  --module video_rl_pair.train "$@"
