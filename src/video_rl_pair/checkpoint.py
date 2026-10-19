# Copyright (c) 2026 video-rl-pair contributors.
"""Atomic checkpoint files, including per-rank torch RNG and BN buffers."""

from collections.abc import Mapping
from pathlib import Path
from typing import NotRequired, TypedDict

import torch
from torch import Tensor, nn
from torch import distributed as dist
from torch.optim.optimizer import StateDict

from video_rl_pair.distributed import Runtime


class RankState(TypedDict):
    """State that can differ even when DDP keeps parameters synchronized."""

    cpu_rng: Tensor
    cuda_rng: Tensor | None
    buffers: dict[str, Tensor]


class Checkpoint(TypedDict):
    """State persisted by the trainer; optimizer fields follow PyTorch's schema."""

    version: int
    step: int
    config: str
    model: Mapping[str, Tensor]
    optimizer: StateDict
    scaler: dict[str, float | int]
    ranks: list[RankState | None]
    data_provenance: NotRequired[str | None]


def gather_rank_states(model: nn.Module, runtime: Runtime) -> list[RankState | None]:
    """All ranks participate; the caller writes only on rank zero."""
    local: RankState = {
        "cpu_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state() if runtime.device.type == "cuda" else None,
        "buffers": {name: value.detach().cpu() for name, value in model.named_buffers()},
    }
    states: list[RankState | None] = [None] * runtime.world_size
    if runtime.world_size > 1:
        dist.all_gather_object(states, local)
    else:
        states[0] = local
    return states


def restore_rank_state(model: nn.Module, state: RankState) -> None:
    """Restore local randomness and normalization state after model loading."""
    torch.set_rng_state(state["cpu_rng"])
    if state["cuda_rng"] is not None:
        torch.cuda.set_rng_state(state["cuda_rng"])
    with torch.no_grad():
        for name, buffer in model.named_buffers():
            _ = buffer.copy_(state["buffers"][name])


def atomic_save(payload: Checkpoint, path: Path) -> None:
    """Readers see a complete old or new checkpoint, never a partial file."""
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    _ = temporary.replace(path)
