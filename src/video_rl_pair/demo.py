# Copyright (c) 2026 video-rl-pair contributors.
"""Synthetic inputs for tracing the computation; never a training dataset."""

from collections.abc import Iterator

import torch
from torch import Tensor

from video_rl_pair.config import ExperimentConfig
from video_rl_pair.distributed import Runtime


def synthetic_batches(
    settings: ExperimentConfig, runtime: Runtime, start_step: int
) -> Iterator[Tensor]:
    """Yield reproducible random clips without a dataset or dataloader.

    A separate generator keeps input generation independent of dropout RNG.
    Microstep/rank seeds let a resumed demo reproduce its next input exactly.
    """
    cfg = settings.input
    shape = (
        settings.train.batch_size,
        cfg.positions,
        cfg.context_frames,
        3,
        cfg.image_size,
        cfg.image_size,
    )
    for microstep in range(
        start_step * settings.train.accumulate,
        settings.train.max_steps * settings.train.accumulate,
    ):
        generator = torch.Generator().manual_seed(
            settings.train.seed + microstep * runtime.world_size + runtime.rank
        )
        yield torch.rand(shape, generator=generator) * 2 - 1
