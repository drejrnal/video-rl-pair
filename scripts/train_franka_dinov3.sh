#!/usr/bin/env bash
# Run from any directory; relative paths resolve from the repository root.
set -euo pipefail

usage() {
  cat <<'HELP'
Usage: GPUS=0,1 bash scripts/train_franka_dinov3.sh [--dry-run] [Hydra overrides...]

One process per GPU; GPUS overrides CUDA_VISIBLE_DEVICES. Select GPUs explicitly.

Environment settings (defaults):
  DATA_ROOT        /data/mxy/collection_datasets/franka_robotic_operation_video
  MODEL_DIR        checkpoints/dinov3-vitb16-pretrain-lvd1689m
  MODE             stage1 (stage1: head only; stage2: head + final N ViT blocks)
  UNFREEZE_LAST_N   2 (stage2 only; 12 means all blocks)
  INIT_CHECKPOINT  unset (load model weights for a new stage2 run)
  RESUME           unset (restore same-stage optimizer, step, RNG)
  OUTPUT_DIR       runs/franka-dinov3-<mode>-<timestamp>-<pid>
  MAX_STEPS        6000        BATCH_SIZE       2 videos per GPU (even)
  ACCUMULATE       1           LEARNING_RATE    0.0001 (head)
  BACKBONE_LR      0.00001     PRECISION        fp32 (or bf16/fp16)
  POSITIONS        32          NUM_WORKERS      0
  LOG_INTERVAL     100         SAVE_INTERVAL    1200
  SEED             42          EVALUATE         true
  WANDB_MODE       disabled    WANDB_PROJECT    video-rl-pair
  WANDB_ENTITY     unset       WANDB_RUN_NAME   output directory basename
  PYTHON           repository .venv/bin/python, otherwise python

Extra Hydra overrides take precedence. Examples:
  GPUS=0 MAX_STEPS=100 OUTPUT_DIR=runs/check bash scripts/train_franka_dinov3.sh
  GPUS=0,1 MODE=stage2 INIT_CHECKPOINT=runs/stage1/last.pt \
    OUTPUT_DIR=runs/stage2 bash scripts/train_franka_dinov3.sh
  GPUS=0 bash scripts/train_franka_dinov3.sh --dry-run

The baseline is evaluation-only: use python -m video_rl_pair.evaluate model.mode=baseline.
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
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${GPUS:-${CUDA_VISIBLE_DEVICES:-}}"
if [[ ! "$CUDA_VISIBLE_DEVICES" =~ ^[a-zA-Z0-9_-]+(,[a-zA-Z0-9_-]+)*$ || "$CUDA_VISIBLE_DEVICES" == -1 ]]; then
  echo 'Set GPUS to a comma-separated GPU list, e.g. GPUS=0,1.' >&2
  exit 2
fi
IFS=',' read -r -a gpu_ids <<< "$CUDA_VISIBLE_DEVICES"
declare -A selected=()
for gpu in "${gpu_ids[@]}"; do
  if [[ -n "${selected[$gpu]:-}" ]]; then
    echo "Duplicate GPU selection: $gpu" >&2
    exit 2
  fi
  selected[$gpu]=1
done
python_bin="${PYTHON:-.venv/bin/python}"
if [[ -z "${PYTHON:-}" && ! -x "$python_bin" ]]; then
  python_bin=python
fi
output_dir="${OUTPUT_DIR:-/data1/mxy/runs/franka-dinov3-${MODE:-stage1}-$(date +%Y%m%d-%H%M%S)-$$}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
hydra_string() {
  local value="${1//\\/\\\\}"
  value="${value//\"/\\\"}"
  printf '"%s"' "$value"
}
overrides=(
  "data.root=$(hydra_string "${DATA_ROOT:-/data/mxy/collection_datasets/franka_robotic_operation_video}")"
  "model.source=$(hydra_string "${MODEL_DIR:-checkpoints/dinov3-vitb16-pretrain-lvd1689m}")"
  "model.mode=${MODE:-stage1}"
  "model.unfreeze_last_n=${UNFREEZE_LAST_N:-2}"
  "train.output_dir=$(hydra_string "$output_dir")"
  "train.max_steps=${MAX_STEPS:-6000}"
  "train.batch_size=${BATCH_SIZE:-2}"
  "train.accumulate=${ACCUMULATE:-1}"
  "train.learning_rate=${LEARNING_RATE:-0.0001}"
  "train.backbone_learning_rate=${BACKBONE_LR:-0.00001}"
  "train.log_every=${LOG_INTERVAL:-100}"
  "train.save_every=${SAVE_INTERVAL:-1200}"
  "train.precision=${PRECISION:-fp32}"
  "train.seed=${SEED:-42}"
  "input.positions=${POSITIONS:-32}"
  "data.num_workers=${NUM_WORKERS:-0}"
  "evaluation.enabled=${EVALUATE:-true}"
  "wandb.mode=${WANDB_MODE:-disabled}"
  "wandb.project=$(hydra_string "${WANDB_PROJECT:-video-rl-pair}")"
  "wandb.name=$(hydra_string "${WANDB_RUN_NAME:-${output_dir##*/}}")"
)
if [[ -n "${INIT_CHECKPOINT:-}" ]]; then
  overrides+=("train.init_checkpoint=$(hydra_string "$INIT_CHECKPOINT")")
fi
if [[ -n "${RESUME:-}" ]]; then
  overrides+=("train.resume=$(hydra_string "$RESUME")")
fi
if [[ -n "${WANDB_ENTITY:-}" ]]; then
  overrides+=("wandb.entity=$(hydra_string "$WANDB_ENTITY")")
fi
command=(
  "$python_bin" -m torch.distributed.run
  --rdzv-backend=c10d --rdzv-endpoint=127.0.0.1:0 --local-addr=127.0.0.1
  --nnodes=1 "--nproc-per-node=${#gpu_ids[@]}"
  --module video_rl_pair.train --config-name franka
  "${overrides[@]}" "$@"
)
printf 'CUDA_VISIBLE_DEVICES=%q ' "$CUDA_VISIBLE_DEVICES"
printf '%q ' "${command[@]}"
printf '\n'
if "$dry_run"; then
  exit 0
fi
"$python_bin" - "${#gpu_ids[@]}" "${overrides[@]}" "$@" <<'PY'
import importlib.util
import sys

import torch
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf
from video_rl_pair.config import ExperimentConfig, WandbConfig

with initialize_config_module(config_module="video_rl_pair.conf", version_base="1.3"):
    config = compose(config_name="franka", overrides=sys.argv[2:])
settings = ExperimentConfig.model_validate(
    OmegaConf.to_container(
        OmegaConf.masked_copy(config, ["model", "loss", "train", "input"]), resolve=True
    )
)
wandb = WandbConfig.model_validate(OmegaConf.to_container(config.wandb, resolve=True))
if settings.model.mode == "baseline":
    sys.exit("baseline is evaluation-only; use video_rl_pair.evaluate.")
if settings.train.batch_size % 2:
    sys.exit("BATCH_SIZE / train.batch_size must be even (two videos per pair).")
if wandb.mode != "disabled" and importlib.util.find_spec("wandb") is None:
    sys.exit("W&B is missing. Install it with: uv sync --extra wandb")
if not torch.cuda.is_available() or torch.cuda.device_count() != int(sys.argv[1]):
    sys.exit("Selected GPUs are unavailable. Check GPU IDs and the CUDA-enabled PyTorch installation.")
PY
exec "${command[@]}"
