# Copyright (c) 2026 video-rl-pair contributors.
"""Annotation-free task-local MP4 pairs and resumable distributed batches."""

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import combinations, islice
from pathlib import Path
from typing import Literal, Self, assert_never

import torch
from pydantic import Field, model_validator
from torch import Tensor
from torch.utils.data import BatchSampler, DataLoader, Dataset, DistributedSampler
from typing_extensions import override

from video_rl_pair.config import ExperimentConfig, Settings
from video_rl_pair.distributed import Runtime
from video_rl_pair.video import sample_video


@dataclass(frozen=True, slots=True)
class VideoPair:
    """Two demonstrations from a single task."""

    task_id: str
    pair_id: str
    first: Path
    second: Path


class PairDataConfig(Settings):
    """Explicit held-out tasks prevent accidental cross-task evaluation."""

    root: Path
    train_tasks: tuple[str, ...] = ()
    eval_tasks: tuple[str, ...] = ()
    num_workers: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_tasks(self) -> Self:
        """Reject unsafe, duplicate, overlapping or missing task selections."""
        tasks = self.train_tasks + self.eval_tasks
        if not self.train_tasks or not self.eval_tasks:
            message = "Both train_tasks and eval_tasks must be nonempty."
            raise ValueError(message)
        if len(set(tasks)) != len(tasks):
            message = "Task selections must be unique and train/eval disjoint."
            raise ValueError(message)
        for task in tasks:
            if not task or task in {".", ".."} or "/" in task or "\\" in task:
                message = f"Task IDs must be simple child names: {task!r}"
                raise ValueError(message)
            if not (self.root / task).is_dir():
                message = f"Missing task directory: {self.root / task}"
                raise ValueError(message)
        train = _discover(self.root, self.train_tasks)
        evaluation = _discover(self.root, self.eval_tasks)
        train_files = {path.resolve() for pair in train for path in (pair.first, pair.second)}
        eval_files = {path.resolve() for pair in evaluation for path in (pair.first, pair.second)}
        if train_files & eval_files:
            message = "Train/eval video paths overlap after resolving symlinks."
            raise ValueError(message)
        return self


def _videos(directory: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(
            path for path in directory.iterdir() if path.suffix.lower() == ".mp4" and path.is_file()
        )
    )


def _discover(root: Path, tasks: tuple[str, ...]) -> tuple[VideoPair, ...]:
    pairs: list[VideoPair] = []
    for task in tasks:
        directory = root / task
        direct = _videos(directory)
        groups = tuple(sorted(path for path in directory.iterdir() if path.is_dir()))
        if direct and groups:
            message = f"Mixed direct MP4 and pair-folder layouts: {directory}"
            raise ValueError(message)
        if direct:
            if len(direct) < 2:
                message = f"Task requires at least two MP4 videos: {directory}"
                raise ValueError(message)
            candidates: list[tuple[str, Path, Path]] = [
                (f"{first.name}--{second.name}", first, second)
                for first, second in combinations(direct, 2)
            ]
        else:
            candidates = []
            for group in groups:
                videos = _videos(group)
                if len(videos) != 2 or any(path.is_dir() for path in group.iterdir()):
                    message = f"Pair folder requires exactly two MP4 videos: {group}"
                    raise ValueError(message)
                candidates.append((group.name, videos[0], videos[1]))
        if not candidates:
            message = f"No MP4 pairs found in task: {directory}"
            raise ValueError(message)
        for pair_id, first, second in candidates:
            if first.resolve() == second.resolve():
                message = f"Pair points to the same video twice: {directory / pair_id}"
                raise ValueError(message)
            pairs.append(VideoPair(task, pair_id, first, second))
    return tuple(pairs)


def discover_pairs(
    config: PairDataConfig, split: Literal["train", "eval"]
) -> tuple[VideoPair, ...]:
    """Return stable task-local pairs from either supported directory layout."""
    match split:
        case "train":
            return _discover(config.root, config.train_tasks)
        case "eval":
            return _discover(config.root, config.eval_tasks)
    assert_never(split)


@dataclass(frozen=True)
class _PairDataset(Dataset[Tensor]):
    pairs: tuple[VideoPair, ...]
    settings: ExperimentConfig
    epoch: int

    def __len__(self) -> int:
        return len(self.pairs)

    @override
    def __getitem__(self, index: int) -> Tensor:
        pair = self.pairs[index]
        seed = self.settings.train.seed + 2 * (self.epoch * len(self.pairs) + index)
        return torch.stack(
            (
                sample_video(pair.first, self.settings, seed).clips,
                sample_video(pair.second, self.settings, seed + 1).clips,
            )
        )


def make_pair_batches(
    settings: ExperimentConfig, runtime: Runtime, start_step: int, *, data: PairDataConfig
) -> Iterable[Tensor]:
    """Shard complete pairs, drop partial batches, and resume at a microstep boundary."""
    if settings.train.batch_size % 2:
        message = "train.batch_size is videos per rank and must be even."
        raise ValueError(message)
    if not 0 <= start_step <= settings.train.max_steps:
        message = "start_step must lie between zero and max_steps."
        raise ValueError(message)
    pairs = discover_pairs(data, "train")
    pairs_per_batch = settings.train.batch_size // 2
    batches_per_epoch = len(pairs) // runtime.world_size // pairs_per_batch
    if not batches_per_epoch:
        message = "Insufficient training pairs for one full batch on every rank."
        raise ValueError(message)
    microstep = start_step * settings.train.accumulate
    stop = settings.train.max_steps * settings.train.accumulate
    while microstep < stop:
        epoch, offset = divmod(microstep, batches_per_epoch)
        dataset = _PairDataset(pairs, settings, epoch)
        sampler = DistributedSampler[int](
            dataset,
            num_replicas=runtime.world_size,
            rank=runtime.rank,
            seed=settings.train.seed,
            drop_last=True,
        )
        sampler.set_epoch(epoch)
        batches = BatchSampler(sampler, pairs_per_batch, drop_last=True)
        loader: Iterable[Tensor] = DataLoader(
            dataset,
            batch_sampler=list(islice(batches, offset, None)),
            num_workers=data.num_workers,
            generator=torch.Generator().manual_seed(settings.train.seed + epoch),
        )
        remaining = min(batches_per_epoch - offset, stop - microstep)
        yield from islice(loader, remaining)
        microstep += remaining
