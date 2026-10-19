# Copyright (c) 2026 video-rl-pair contributors.
"""Serializable matching diagnostics; no annotation-based accuracy is inferred."""

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from statistics import mean
from typing import Literal

from pydantic import Field

from video_rl_pair.config import Settings


class EvaluationConfig(Settings):
    """Deterministic held-out evaluation, optionally after the final training step."""

    enabled: bool = True
    checkpoint: Path | None = None
    allow_initialization: bool = False
    output_dir: Path = Path("runs/evaluation")
    methods: tuple[Literal["nearest", "dtw"], ...] = Field(default=("nearest", "dtw"), min_length=1)
    positions: int = Field(default=40, ge=2)
    chunk_size: int = Field(default=8, ge=1)
    max_pairs_per_task: int = Field(default=20, ge=1)
    visual_pairs_per_task: int = Field(default=2, ge=0)
    progression_horizon: int = Field(default=8, ge=1)
    seed: int = Field(default=42, ge=0)


class Diagnostics(Settings):
    """Order, coverage and self-return statistics; none establish semantic correctness."""

    mean_cosine: float
    tau_b: float | None
    unique_target_fraction: float
    cycle_time_error: float


class PairResult(Settings):
    """One direction of one pair, including indices needed to inspect a match."""

    task_id: str
    pair_id: str
    source: str
    target: str
    method: Literal["nearest", "dtw", "time"]
    source_times: tuple[float, ...]
    target_times: tuple[float, ...]
    indices: tuple[int, ...]
    diagnostics: Diagnostics


class Summary(Settings):
    """Average diagnostics and explicit coverage for undefined ordering scores."""

    count: int
    tau_defined_count: int
    diagnostics: Diagnostics


class EvaluationReport(Settings):
    """Task means have equal weight in macro, regardless of task size."""

    checkpoint: str | None
    initialization_baseline: bool
    training_provenance_checked: bool
    train_tasks: tuple[str, ...]
    eval_tasks: tuple[str, ...]
    options: EvaluationConfig
    available_pairs: dict[str, int]
    selected_pairs: dict[str, int]
    embedding_spread: dict[str, float]
    by_task: dict[str, dict[str, Summary]]
    macro: dict[str, Summary]
    pairs: tuple[PairResult, ...]


def summarize(values: Sequence[Diagnostics]) -> Summary:
    """Average evaluated units; missing tau is excluded with coverage retained."""
    tau = [value.tau_b for value in values if value.tau_b is not None]
    return Summary(
        count=len(values),
        tau_defined_count=len(tau),
        diagnostics=Diagnostics(
            mean_cosine=mean(value.mean_cosine for value in values),
            tau_b=mean(tau) if tau else None,
            unique_target_fraction=mean(value.unique_target_fraction for value in values),
            cycle_time_error=mean(value.cycle_time_error for value in values),
        ),
    )


def aggregate(
    results: Sequence[PairResult],
) -> tuple[dict[str, dict[str, Summary]], dict[str, Summary]]:
    """Average directions/pairs within tasks, then average tasks equally."""
    groups: dict[str, dict[str, list[Diagnostics]]] = {}
    for result in results:
        groups.setdefault(result.task_id, {}).setdefault(result.method, []).append(
            result.diagnostics
        )
    tasks = {
        task: {method: summarize(values) for method, values in methods.items()}
        for task, methods in groups.items()
    }
    macro_values: dict[str, list[Diagnostics]] = defaultdict(list)
    for methods in tasks.values():
        for method, summary in methods.items():
            macro_values[method].append(summary.diagnostics)
    return tasks, {method: summarize(values) for method, values in macro_values.items()}
