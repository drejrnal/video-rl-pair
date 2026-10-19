# Copyright (c) 2026 video-rl-pair contributors.
"""Decode only sampled MP4 images, keeping cached metadata small."""

from dataclasses import dataclass
from functools import lru_cache
from itertools import pairwise
from pathlib import Path
from typing import assert_never

import av
import torch
from torch import Tensor

from video_rl_pair.config import ExperimentConfig


@dataclass(frozen=True, slots=True)
class SampledVideo:
    """Context clips [T,K,3,H,W] and their anchor timestamps in seconds."""

    clips: Tensor
    times: Tensor


@lru_cache(maxsize=64)
def _timestamps(identity: tuple[Path, int, int]) -> tuple[float, ...]:
    path, _, _ = identity
    with av.open(str(path)) as container:
        if not container.streams.video:
            message = f"No video stream in {path}"
            raise ValueError(message)
        stream = container.streams.video[0]
        rate = float(stream.average_rate or 30)
        times = tuple(
            float(frame.time) if frame.pts is not None else index / rate
            for index, frame in enumerate(container.decode(stream))
        )
    if len(times) < 2 or any(right <= left for left, right in pairwise(times)):
        message = f"Video needs two frames and strictly increasing timestamps: {path}"
        raise ValueError(message)
    return times


def sample_video(path: Path, settings: ExperimentConfig, seed: int | None = None) -> SampledVideo:
    """Sample sorted anchor frames and past context with clamping at frame zero."""
    stat = path.stat()
    times = _timestamps((path.resolve(), stat.st_size, stat.st_mtime_ns))
    count = len(times)
    positions = settings.input.positions
    if seed is None:
        positions = min(positions, count)
        anchors = torch.linspace(0, count - 1, positions).round().long()
    else:
        generator = torch.Generator().manual_seed(seed)
        anchors = (
            (
                (torch.arange(positions) + torch.rand(positions, generator=generator))
                * count
                / positions
            )
            .long()
            .clamp_max(count - 1)
        )
    offsets = torch.arange(settings.input.context_frames - 1, -1, -1)
    indices = (anchors[:, None] - offsets * settings.input.context_stride).clamp_min(0)
    flat_indices = [int(index.item()) for index in indices.flatten()]
    wanted = set(flat_indices)
    frames: dict[int, Tensor] = {}
    size = settings.input.image_size
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in wanted:
                rgb = frame.reformat(width=size, height=size, format="rgb24").to_ndarray()
                frames[index] = torch.tensor(rgb).permute(2, 0, 1).float().div_(255)
            if len(frames) == len(wanted):
                break
    if len(frames) != len(wanted):
        message = f"Video changed or could not decode all sampled frames: {path}"
        raise ValueError(message)
    clips = torch.stack([frames[index] for index in flat_indices])
    clips = clips.reshape(positions, settings.input.context_frames, 3, size, size)
    clips = _normalize(clips, settings)
    sampled_times = torch.tensor([times[int(anchor.item())] - times[0] for anchor in anchors])
    return SampledVideo(clips=clips, times=sampled_times)


def _normalize(clips: Tensor, settings: ExperimentConfig) -> Tensor:
    match settings.model.backbone:
        case "resnet50_v2":
            return clips.mul(2).sub(1)
        case "resnet50_v1":
            mean = clips.new_tensor([0.485, 0.456, 0.406]).reshape(1, 1, 3, 1, 1)
            std = clips.new_tensor([0.229, 0.224, 0.225]).reshape(1, 1, 3, 1, 1)
            return (clips - mean) / std
    assert_never(settings.model.backbone)
