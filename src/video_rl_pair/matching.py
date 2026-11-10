# Copyright (c) 2026 video-rl-pair contributors.
"""Cosine matching costs and complete, endpoint-constrained hard DTW paths."""

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class MatchingResult:
    """Every cell in the optimal path is retained, including repeated frame indices."""

    similarity: Tensor
    cost: Tensor
    path: tuple[tuple[int, int], ...]
    total_cost: float


def _unit_embeddings(embeddings: Tensor) -> Tensor:
    """Normalize finite, nonzero frame features without overflow."""
    if embeddings.ndim != 2 or min(embeddings.shape) < 1:
        message = "Expected nonempty embeddings shaped [frames, dimensions]."
        raise ValueError(message)
    if not embeddings.is_floating_point() or not bool(torch.isfinite(embeddings).all()):
        message = "Embeddings must contain finite floating-point values."
        raise ValueError(message)
    values = embeddings.detach().to(device="cpu", dtype=torch.float64)
    scales = values.abs().amax(dim=1, keepdim=True)
    if bool((scales == 0).any()):
        message = "Every frame embedding must have a nonzero norm."
        raise ValueError(message)
    scaled = values / scales
    return scaled / scaled.square().sum(dim=1, keepdim=True).sqrt()


def dtw_path(cost: Tensor) -> tuple[tuple[int, int], ...]:
    """Minimize total cost; ties prefer diagonal, then above, then left.

    Paths begin at (0, 0), end at (M-1, N-1), and use unit diagonal,
    vertical or horizontal steps. This is hard DTW for inspection, not
    the differentiable smoothDTW objective used during training.
    """
    if cost.ndim != 2 or min(cost.shape) < 1 or not bool(torch.isfinite(cost).all()):
        message = "DTW requires a nonempty, finite cost matrix."
        raise ValueError(message)
    values = cost.detach().to(device="cpu", dtype=torch.float64)
    rows, columns = values.shape
    accumulated = [[math.inf] * (columns + 1) for _ in range(rows + 1)]
    accumulated[0][0] = 0.0
    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            accumulated[row][column] = float(values[row - 1, column - 1]) + min(
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


def match_embeddings(source: Tensor, target: Tensor) -> MatchingResult:
    """Return cosine similarity, 1-cosine cost, and the complete optimal path."""
    first, second = _unit_embeddings(source), _unit_embeddings(target)
    if first.shape[1] != second.shape[1]:
        message = "Source and target embedding dimensions must agree."
        raise ValueError(message)
    similarity = (first @ second.T).clamp(-1, 1)
    cost = 1 - similarity
    path = dtw_path(cost)
    return MatchingResult(
        similarity=similarity,
        cost=cost,
        path=path,
        total_cost=sum(float(cost[row, column]) for row, column in path),
    )
