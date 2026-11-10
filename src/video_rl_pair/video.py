# Copyright (c) 2026 video-rl-pair contributors.
"""Endpoint-preserving video sampling and DINOv3 RGB preprocessing."""

from dataclasses import dataclass
from functools import lru_cache
from itertools import pairwise
from pathlib import Path

import av
import torch
from PIL import Image
from torch import Tensor
from torch.nn import functional

from video_rl_pair.config import ExperimentConfig


@dataclass(frozen=True, slots=True)
class SampledVideo:
    """Frames [T,3,H,W]; only the first valid_length entries are real samples."""

    clips: Tensor
    times: Tensor
    images: tuple[Image.Image, ...]
    valid_length: int


@lru_cache(maxsize=64)
def _timestamps(identity: tuple[Path, int, int]) -> tuple[float, ...]:
    path, _, _ = identity
    with av.open(str(path)) as container:
        if not container.streams.video:
            msg = f"No video stream: {path}"
            raise ValueError(msg)
        stream = container.streams.video[0]
        rate = float(stream.average_rate or 30)
        times = tuple(
            float(frame.time) if frame.pts is not None else index / rate
            for index, frame in enumerate(container.decode(stream))
        )
    if len(times) < 2 or any(right <= left for left, right in pairwise(times)):
        msg = f"Video needs two frames and increasing timestamps: {path}"
        raise ValueError(msg)
    return times


def sample_indices(count: int, positions: int, seed: int | None) -> tuple[Tensor, int]:
    """Keep endpoints; randomize distinct interior frames only during training."""
    if count < 2 or positions < 2:
        msg = "Videos and sampling grids need at least two frames."
        raise ValueError(msg)
    valid_length = min(count, positions)
    if seed is None:
        return torch.linspace(0, count - 1, valid_length).round().long(), valid_length
    if count < positions:
        # Tail padding is excluded from both directional costs and cycle loss.
        return torch.cat((torch.arange(count), torch.full((positions - count,), count - 1))), count
    generator = torch.Generator().manual_seed(seed)
    interior = positions - 2
    anchors = [0]
    for index in range(interior):
        low = 1 + index * (count - 2) // interior
        high = 1 + (index + 1) * (count - 2) // interior
        anchors.append(int(torch.randint(low, high, (), generator=generator)))
    anchors.append(count - 1)
    return torch.tensor(anchors), positions


def preprocess_rgb(rgb: Tensor, image_size: int) -> Tensor:
    """Match DINOv3 web-model preprocessing: rescale, bilinear resize, ImageNet norm."""
    pixels = rgb.permute(2, 0, 1).float().div(255).unsqueeze(0)
    pixels = functional.interpolate(
        pixels, size=(image_size, image_size), mode="bilinear", align_corners=False, antialias=True
    )[0]
    mean = pixels.new_tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = pixels.new_tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    return (pixels - mean) / std


def sample_video(path: Path, settings: ExperimentConfig, seed: int | None = None) -> SampledVideo:
    """Decode selected frames; evaluation retains original RGB images for inspection."""
    stat = path.stat()
    times = _timestamps((path.resolve(), stat.st_size, stat.st_mtime_ns))
    anchors, valid_length = sample_indices(len(times), settings.input.positions, seed)
    wanted = set(anchors.tolist())
    frames: dict[int, Tensor] = {}
    images: dict[int, Image.Image] = {}
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in wanted:
                rgb = frame.to_ndarray(format="rgb24")
                frames[index] = preprocess_rgb(
                    torch.from_numpy(rgb.copy()), settings.input.image_size
                )
                if seed is None:
                    images[index] = Image.fromarray(rgb)
            if len(frames) == len(wanted):
                break
    if len(frames) != len(wanted):
        msg = f"Video changed or sampled frames could not be decoded: {path}"
        raise ValueError(msg)
    return SampledVideo(
        clips=torch.stack([frames[int(i)] for i in anchors]),
        times=torch.tensor([times[int(i)] - times[0] for i in anchors], dtype=torch.float64),
        images=tuple(images[int(i)] for i in anchors) if seed is None else (),
        valid_length=valid_length,
    )
