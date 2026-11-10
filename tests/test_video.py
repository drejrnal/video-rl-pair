# Copyright (c) 2026 video-rl-pair contributors.
"""Video sampling preserves endpoints, excludes padding, and matches DINOv3 preprocessing."""

from collections.abc import Iterator
from pathlib import Path

import av
import numpy as np
import pytest
import torch
from transformers import DINOv3ViTImageProcessorFast

from video_rl_pair.config import ExperimentConfig, InputConfig, LossConfig, ModelConfig, TrainConfig
from video_rl_pair.video import preprocess_rgb, sample_indices, sample_video


@pytest.fixture(autouse=True)
def single_cpu_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def write_video(path: Path, count: int) -> None:
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width, stream.height = 48, 32
        stream.pix_fmt = "yuv420p"
        for index in range(count):
            array = np.zeros((32, 48, 3), dtype=np.uint8)
            array[:, :, 0] = 40 + index * 30
            array[:, :, 1] = np.arange(48, dtype=np.uint8)[None, :] * 4
            array[:, :, 2] = np.arange(32, dtype=np.uint8)[:, None] * 6
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def settings() -> ExperimentConfig:
    return ExperimentConfig(
        model=ModelConfig(),
        input=InputConfig(positions=6, image_size=32),
        loss=LossConfig(),
        train=TrainConfig(),
    )


def test_training_samples_are_distinct_ordered_seeded_and_cover_endpoints() -> None:
    first, length = sample_indices(100, 32, seed=42)
    second, _ = sample_indices(100, 32, seed=42)
    other, _ = sample_indices(100, 32, seed=43)
    assert length == 32
    assert first[0] == 0
    assert first[-1] == 99
    assert bool((first[1:] > first[:-1]).all())
    torch.testing.assert_close(first, second)
    assert not torch.equal(first, other)
    assert sample_indices(40, 2, seed=1)[0].tolist() == [0, 39]


@pytest.mark.parametrize("count", [2, 4])
def test_short_video_training_padding_and_unique_evaluation(tmp_path: Path, count: int) -> None:
    path = tmp_path / f"short-{count}.mp4"
    write_video(path, count)
    training = sample_video(path, settings(), seed=7)
    assert training.clips.shape == (6, 3, 32, 32)
    assert training.valid_length == count
    assert training.images == ()
    for index in range(count, 6):
        torch.testing.assert_close(training.clips[index], training.clips[count - 1])
        assert training.times[index] == training.times[count - 1]
    evaluation = sample_video(path, settings())
    assert evaluation.clips.shape == (count, 3, 32, 32)
    assert evaluation.valid_length == count
    assert len(evaluation.images) == count
    assert all(image.size == (48, 32) for image in evaluation.images)
    assert bool((evaluation.times[1:] > evaluation.times[:-1]).all())
    torch.testing.assert_close(evaluation.times, torch.arange(count, dtype=torch.float64) / 10)
    torch.testing.assert_close(evaluation.clips, training.clips[:count])
    with av.open(str(path)) as container:
        originals = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    for image, original in zip(evaluation.images, originals, strict=True):
        np.testing.assert_array_equal(np.asarray(image), original)


def test_preprocessing_agrees_with_official_dinov3_processor() -> None:
    rgb = torch.randint(
        0, 256, (48, 64, 3), dtype=torch.uint8, generator=torch.Generator().manual_seed(9)
    )
    processor = DINOv3ViTImageProcessorFast(size={"height": 32, "width": 32})
    expected = processor(rgb.permute(2, 0, 1), return_tensors="pt").pixel_values[0]
    torch.testing.assert_close(preprocess_rgb(rgb, 32), expected, rtol=1e-5, atol=1e-6)
