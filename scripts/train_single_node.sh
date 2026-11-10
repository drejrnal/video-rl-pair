#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
python_bin="${PYTHON:-.venv/bin/python}"
if [[ -z "${PYTHON:-}" && ! -x "$python_bin" ]]; then
  python_bin=python
fi
exec "$python_bin" -m torch.distributed.run \
  --rdzv-backend=c10d --rdzv-endpoint=127.0.0.1:0 \
  --local-addr=127.0.0.1 --nnodes=1 --nproc-per-node="${GPUS_PER_NODE:-1}" \
  --module video_rl_pair.train "$@"
