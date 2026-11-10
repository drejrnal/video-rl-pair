#!/usr/bin/env bash
# Download a complete, local Hugging Face snapshot; training never downloads weights.
set -euo pipefail

usage() {
  cat <<'HELP'
Usage: bash scripts/download_dinov3.sh [--dry-run]

First request access to facebook/dinov3-vitb16-pretrain-lvd1689m on Hugging Face,
then run .venv/bin/hf auth login (or supply HF_TOKEN in the environment).

Environment:
  MODEL_DIR     checkpoints/dinov3-vitb16-pretrain-lvd1689m
  HF_REVISION   main (set a commit hash for a reproducible download)
  HF            repository .venv/bin/hf, otherwise hf
HELP
}
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  usage
  exit 0
fi
dry_run=false
if [[ "${1:-}" == --dry-run ]]; then
  dry_run=true
  shift
fi
if (( $# )); then
  usage >&2
  exit 2
fi
cd "$(dirname "${BASH_SOURCE[0]}")/.."
hf_bin="${HF:-.venv/bin/hf}"
if [[ -z "${HF:-}" && ! -x "$hf_bin" ]]; then
  hf_bin=hf
fi
command=(
  "$hf_bin" download facebook/dinov3-vitb16-pretrain-lvd1689m
  --revision "${HF_REVISION:-main}"
  --local-dir "${MODEL_DIR:-checkpoints/dinov3-vitb16-pretrain-lvd1689m}"
)
printf '%q ' "${command[@]}"
printf '\n'
if "$dry_run"; then
  exit 0
fi
exec "${command[@]}"
