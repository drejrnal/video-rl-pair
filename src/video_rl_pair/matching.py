# Copyright (c) 2026 video-rl-pair contributors.
"""Detached CPU matching and annotation-free embedding diagnostics.

Rows are source frames and columns are target frames. These diagnostics do
not establish semantic accuracy. Hard DTW here is an evaluation alignment,
not the differentiable smoothDTW objective used for training.
"""

import math
from dataclasses import dataclass
from typing import Literal, assert_never

import torch
from torch import Tensor
from typing_extensions import override


@dataclass(frozen=True, slots=True)
class MatchingInputError(ValueError):
    """An input violates the matching boundary contract."""

    reason: str

    @override
    def __str__(self) -> str:
        """Describe the invalid input."""
        return self.reason


@dataclass(frozen=True, slots=True)
class MatchingResult:
    """CPU indices [M], float64 cosine similarity [M,N], optional DTW path."""

    indices: Tensor
    similarity: Tensor
    path: tuple[tuple[int, int], ...] | None = None


@dataclass(frozen=True, slots=True)
class MatchDiagnostics:
    """Cosine, temporal order, and target coverage; not semantic accuracy."""

    mean_cosine: float
    tau_b: float | None
    unique_target_fraction: float


def _unit_embeddings(embeddings: Tensor) -> Tensor:
    """Parse finite, nonzero floating rows into detached float64 unit vectors."""
    if embeddings.ndim != 2 or min(embeddings.shape) < 1:
        msg = "Expected nonempty embeddings [frames, dimensions]."
        raise MatchingInputError(msg)
    if not embeddings.is_floating_point() or not bool(torch.isfinite(embeddings).all()):
        msg = "Embeddings must be finite floating-point values."
        raise MatchingInputError(msg)
    values = embeddings.detach().to(device="cpu", dtype=torch.float64)
    scales = values.abs().amax(dim=1, keepdim=True)
    if bool((scales == 0).any()):
        msg = "Embedding rows must have nonzero norm."
        raise MatchingInputError(msg)
    scaled = values / scales
    return scaled / scaled.square().sum(dim=1, keepdim=True).sqrt()


def _similarity(source: Tensor, target: Tensor) -> Tensor:
    """Normalize two sequences and compare all frame pairs."""
    source_units, target_units = _unit_embeddings(source), _unit_embeddings(target)
    if source_units.shape[1] != target_units.shape[1]:
        msg = "Source and target embedding dimensions must agree."
        raise MatchingInputError(msg)
    return (source_units @ target_units.T).clamp(-1, 1)


def _dtw_path(cost: Tensor) -> tuple[tuple[int, int], ...]:
    """Minimize total endpoint-to-endpoint cost; ties prefer diagonal/above/left."""
    rows, columns = cost.shape
    accumulated = [[math.inf] * (columns + 1) for _ in range(rows + 1)]
    accumulated[0][0] = 0.0
    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            accumulated[row][column] = float(cost[row - 1, column - 1]) + min(
                accumulated[row - 1][column - 1],
                accumulated[row - 1][column],
                accumulated[row][column - 1],
            )
    row, column = rows, columns
    reverse_path: list[tuple[int, int]] = []
    while row > 0 and column > 0:
        reverse_path.append((row - 1, column - 1))
        candidates = ((row - 1, column - 1), (row - 1, column), (row, column - 1))
        row, column = min(candidates, key=lambda cell: accumulated[cell[0]][cell[1]])
    return tuple(reversed(reverse_path))


def match_embeddings(
    x: Tensor, y: Tensor, method: Literal["nearest", "dtw"] = "nearest"
) -> MatchingResult:
    """Match floating [M,D] to [N,D], allowing unequal or singleton lengths.

    Nearest ties choose the earliest target. DTW minimizes sum(1-cosine)
    with fixed endpoints and unit diagonal/vertical/horizontal steps. Each
    source chooses the highest-cosine target on its path, earliest on ties.
    Outputs are detached CPU tensors; inputs are unchanged.
    """
    similarity = _similarity(x, y)
    match method:
        case "nearest":
            return MatchingResult(similarity.argmax(dim=1), similarity)
        case "dtw":
            path = _dtw_path(1 - similarity)
            candidates = torch.full_like(similarity, -math.inf)
            for row, column in path:
                candidates[row, column] = similarity[row, column]
            return MatchingResult(candidates.argmax(dim=1), similarity, path)
        case _:
            assert_never(method)


def _normalized_times(times: Tensor) -> Tensor:
    """Parse strictly increasing timestamps and normalize their actual range."""
    if times.ndim != 1 or times.numel() < 2 or times.is_complex() or times.dtype == torch.bool:
        msg = "Expected at least two real timestamps in a vector."
        raise MatchingInputError(msg)
    values = times.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(values).all()) or not bool((values[1:] > values[:-1]).all()):
        msg = "Timestamps must be finite and strictly increasing."
        raise MatchingInputError(msg)
    span = values[-1] - values[0]
    if not bool(torch.isfinite(span)):
        msg = "Timestamp range must be finite."
        raise MatchingInputError(msg)
    return (values - values[0]) / span


def normalized_time_matches(source_times: Tensor, target_times: Tensor) -> Tensor:
    """Match normalized timestamps by absolute distance; ties choose earliest."""
    source, target = _normalized_times(source_times), _normalized_times(target_times)
    return (source[:, None] - target[None, :]).abs().argmin(dim=1)


def kendall_tau_b(indices: Tensor) -> float | None:
    """Compare source order to target indices, correcting target ties.

    Source order has no ties. Return None for a singleton or all-tied target
    matches, where tau-b is undefined. This measures order, not correctness.
    """
    if indices.ndim != 1 or indices.numel() < 1 or indices.dtype != torch.long:
        msg = "Expected a nonempty long vector of target indices."
        raise MatchingInputError(msg)
    if bool((indices < 0).any()):
        msg = "Target indices must be nonnegative."
        raise MatchingInputError(msg)
    values = indices.detach().cpu()
    concordant, discordant = 0, 0
    for row in range(values.numel() - 1):
        later = values[row + 1 :]
        concordant += int((later > values[row]).sum())
        discordant += int((later < values[row]).sum())
    comparable = concordant + discordant
    if comparable == 0:
        return None
    pairs = values.numel() * (values.numel() - 1) // 2
    return (concordant - discordant) / math.sqrt(pairs * comparable)


def summarize_matches(source: Tensor, target: Tensor, indices: Tensor) -> MatchDiagnostics:
    """Summarize selected cosine, order, and fraction of target frames visited."""
    similarity = _similarity(source, target)
    tau = kendall_tau_b(indices)
    selected = indices.detach().cpu()
    if selected.numel() != similarity.shape[0] or bool((selected >= similarity.shape[1]).any()):
        msg = "Expected one in-range target index per source frame."
        raise MatchingInputError(msg)
    scores = similarity[torch.arange(selected.numel()), selected]
    return MatchDiagnostics(
        mean_cosine=float(scores.mean()),
        tau_b=tau,
        unique_target_fraction=len({int(index) for index in selected}) / similarity.shape[1],
    )


def embedding_spread(embeddings: Tensor) -> float:
    """Mean squared distance to the unit-embedding centroid (zero at collapse)."""
    units = _unit_embeddings(embeddings)
    return float((units - units.mean(dim=0)).square().sum(dim=1).mean())
