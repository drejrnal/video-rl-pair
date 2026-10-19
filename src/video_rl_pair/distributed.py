# Copyright (c) 2026 video-rl-pair contributors.
"""One process per GPU via torchrun; Gloo also permits CPU smoke tests."""

import os
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta

import torch
from torch import Tensor
from torch import distributed as dist

from video_rl_pair.config import TrainConfig


@dataclass(frozen=True, slots=True)
class Runtime:
    """Rank identity and collectives needed by the training procedure."""

    device: torch.device
    rank: int
    world_size: int

    def mean(self, value: Tensor) -> Tensor:
        """Reduce detached metrics without changing the gradient scale."""
        result = value.detach().clone()
        if self.world_size > 1:
            _ = dist.all_reduce(result)
            result /= self.world_size
        return result

    def require_all(self, condition: bool, message: str) -> None:
        """Make every rank fail together before a mismatched backward call."""
        valid = torch.tensor(int(condition), device=self.device)
        if self.world_size > 1:
            _ = dist.all_reduce(valid, op=dist.ReduceOp.MIN)
        if not bool(valid.item()):
            raise RuntimeError(message)


@contextmanager
def distributed_runtime(config: TrainConfig) -> Generator[Runtime, None, None]:
    """Initialize from torchrun's environment and always close the group."""
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_cuda = config.device == "cuda" or (config.device == "auto" and torch.cuda.is_available())
    device = torch.device("cuda", local_rank) if use_cuda else torch.device("cpu")
    if use_cuda:
        torch.cuda.set_device(device)
    if config.precision != "fp32" and not use_cuda:
        message = "Use precision=fp32 for CPU runs."
        raise ValueError(message)
    if config.precision == "bf16" and not torch.cuda.is_bf16_supported():
        message = "This GPU does not support bf16; select fp32 or fp16."
        raise ValueError(message)
    if world_size > 1:
        dist.init_process_group(
            backend="nccl" if use_cuda else "gloo", timeout=timedelta(minutes=10)
        )
    try:
        yield Runtime(device=device, rank=rank, world_size=world_size)
    finally:
        if world_size > 1:
            dist.destroy_process_group()
