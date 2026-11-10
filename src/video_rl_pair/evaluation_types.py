# Copyright (c) 2026 video-rl-pair contributors.
"""Configuration and inspectable outputs for the single DTW alignment task."""

from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from video_rl_pair.config import Settings


class EvaluationConfig(Settings):
    """Align an explicit pair or deterministic pairs from held-out task folders."""

    enabled: bool = True
    checkpoint: Path | None = None
    source: Path | None = None
    target: Path | None = None
    output_dir: Path = Path("runs/evaluation")
    positions: int = Field(default=32, ge=2)
    chunk_size: int = Field(default=8, ge=1)
    max_pairs_per_task: int = Field(default=2, ge=1)
    image_size: int = Field(default=512, ge=64)
    seed: int = Field(default=42, ge=0)

    @model_validator(mode="after")
    def validate_pair(self) -> Self:
        """Explicit evaluation requires two distinct existing video files."""
        if (self.source is None) != (self.target is None):
            message = "Set both evaluation.source and evaluation.target, or neither."
            raise ValueError(message)
        if self.source is not None and self.target is not None:
            for path in (self.source, self.target):
                if not path.is_file():
                    message = f"Evaluation video does not exist: {path}"
                    raise ValueError(message)
            if self.source.resolve() == self.target.resolve():
                message = "Source and target must be different video files."
                raise ValueError(message)
        return self


class PathStep(Settings):
    """One ordered pair of sampled frames on the full DTW path."""

    source_index: int
    target_index: int
    source_time: float
    target_time: float
    cosine: float
    cost: float
    image: str


class PairResult(Settings):
    """Matrices use source rows and target columns; PNG paths are report-relative."""

    task_id: str
    pair_id: str
    source: str
    target: str
    source_times: tuple[float, ...]
    target_times: tuple[float, ...]
    similarity: tuple[tuple[float, ...], ...]
    cost: tuple[tuple[float, ...], ...]
    path: tuple[PathStep, ...]
    total_cost: float


class EvaluationReport(Settings):
    """Human-inspectable alignments; path cost is not a semantic accuracy score."""

    checkpoint: str | None
    mode: Literal["baseline", "stage1", "stage2"]
    training_provenance_checked: bool
    provenance_check: Literal["not_available", "files", "tasks_and_files"]
    options: EvaluationConfig
    selected_pairs: dict[str, int]
    pairs: tuple[PairResult, ...]
