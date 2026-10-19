# Copyright (c) 2026 video-rl-pair contributors.
"""Serializable frame sequences with independently computed bidirectional matches."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import Field

from video_rl_pair.config import Settings

if TYPE_CHECKING:
    from collections.abc import Sequence

    from video_rl_pair.data import VideoPair
    from video_rl_pair.evaluation_types import PairResult
    from video_rl_pair.inference import EncodedVideo


class ProgressionVideo(Settings):
    """Sample times and embedded images for one video in the offline viewer."""

    path: str
    times: tuple[float, ...]
    images: tuple[str, ...]


class ProgressionMatch(Settings):
    """Forward and reverse maps retain their own sampled source lengths."""

    method: Literal["nearest", "dtw", "time"]
    forward: tuple[int, ...]
    reverse: tuple[int, ...]


class ProgressionPair(Settings):
    """One held-out task pair shared across all displayed matching methods."""

    task_id: str
    pair_id: str
    source: ProgressionVideo
    target: ProgressionVideo
    matches: tuple[ProgressionMatch, ...]


class ProgressionReport(Settings):
    """Bounded selection of pairs for visual inspection of retrieved progression."""

    horizon: int = Field(ge=1)
    pairs: tuple[ProgressionPair, ...]
    checkpoint: str | None
    initialization_baseline: bool


def make_progression_pair(
    pair: VideoPair,
    videos: tuple[EncodedVideo, EncodedVideo],
    results: Sequence[PairResult],
) -> ProgressionPair:
    """Reuse evaluation maps verbatim; reverse retrieval is never an inverted forward map."""
    first, second = videos
    by_direction = {
        (result.method, result.source, result.target): result.indices for result in results
    }
    return ProgressionPair(
        task_id=pair.task_id,
        pair_id=pair.pair_id,
        source=ProgressionVideo(
            path=str(first.path),
            times=tuple(float(time) for time in first.times),
            images=first.images,
        ),
        target=ProgressionVideo(
            path=str(second.path),
            times=tuple(float(time) for time in second.times),
            images=second.images,
        ),
        matches=tuple(
            ProgressionMatch(
                method=result.method,
                forward=result.indices,
                reverse=by_direction[(result.method, str(second.path), str(first.path))],
            )
            for result in results
            if result.source == str(first.path) and result.target == str(second.path)
        ),
    )
