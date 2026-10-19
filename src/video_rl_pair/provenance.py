# Copyright (c) 2026 video-rl-pair contributors.
"""Persist pair identities and lightweight file metadata with training checkpoints."""

from pathlib import Path

from video_rl_pair.config import Settings
from video_rl_pair.data import PairDataConfig, discover_pairs


class PairIdentity(Settings):
    """Resolved files and pair order control deterministic loader resume."""

    task_id: str
    pair_id: str
    first: str
    second: str


class VideoIdentity(Settings):
    """Path, size and modification time detect routine dataset edits without file hashes."""

    path: str
    size: int
    modified_ns: int


class DataProvenance(Settings):
    """Record the actual training inputs, separate from the later evaluation set."""

    train_tasks: tuple[str, ...]
    pairs: tuple[PairIdentity, ...]
    videos: tuple[VideoIdentity, ...]


def snapshot_data(data: PairDataConfig) -> DataProvenance:
    """Record stable pair ordering and metadata before training begins."""
    pairs = tuple(
        PairIdentity(
            task_id=p.task_id,
            pair_id=p.pair_id,
            first=str(p.first.resolve()),
            second=str(p.second.resolve()),
        )
        for p in discover_pairs(data, "train")
    )
    paths = sorted({path for pair in pairs for path in (pair.first, pair.second)})
    videos: list[VideoIdentity] = []
    for path in paths:
        stat = Path(path).stat()
        videos.append(VideoIdentity(path=path, size=stat.st_size, modified_ns=stat.st_mtime_ns))
    return DataProvenance(train_tasks=data.train_tasks, pairs=pairs, videos=tuple(videos))


def check_held_out(provenance: DataProvenance, data: PairDataConfig) -> None:
    """Reject task or resolved-file leakage against the checkpoint's recorded training set."""
    eval_paths = {
        str(path.resolve())
        for pair in discover_pairs(data, "eval")
        for path in (pair.first, pair.second)
    }
    train_paths = {video.path for video in provenance.videos}
    if set(provenance.train_tasks) & set(data.eval_tasks) or train_paths & eval_paths:
        message = "Evaluation overlaps the checkpoint's recorded training tasks or videos."
        raise ValueError(message)
    if set(provenance.train_tasks) != set(data.train_tasks):
        message = "Configured train_tasks differ from the checkpoint's training provenance."
        raise ValueError(message)
