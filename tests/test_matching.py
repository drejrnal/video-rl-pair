# Copyright (c) 2026 video-rl-pair contributors.
"""Hard DTW is checked against exhaustive paths, not a duplicate recurrence."""

import itertools
from collections.abc import Iterator

import pytest
import torch

from video_rl_pair.matching import dtw_path, match_embeddings


def _all_paths(
    rows: int, columns: int, start: tuple[int, int] = (0, 0)
) -> Iterator[tuple[tuple[int, int], ...]]:
    if start == (rows - 1, columns - 1):
        yield (start,)
        return
    for dr, dc in ((1, 1), (1, 0), (0, 1)):
        next_cell = (start[0] + dr, start[1] + dc)
        if next_cell[0] < rows and next_cell[1] < columns:
            for tail in _all_paths(rows, columns, next_cell):
                yield (start, *tail)


@pytest.mark.parametrize(("rows", "columns"), [(1, 3), (3, 1), (2, 3), (3, 3)])
def test_dtw_finds_global_minimum(rows: int, columns: int) -> None:
    for values in itertools.islice(itertools.product((0.0, 0.7, 2.0), repeat=rows * columns), 40):
        cost = torch.tensor(values, dtype=torch.float64).reshape(rows, columns)
        path = dtw_path(cost)
        optimum = min(sum(float(cost[i, j]) for i, j in p) for p in _all_paths(rows, columns))
        assert sum(float(cost[i, j]) for i, j in path) == pytest.approx(optimum)
        assert path[0] == (0, 0)
        assert path[-1] == (rows - 1, columns - 1)


def test_complete_path_keeps_one_to_many_matches() -> None:
    source = torch.eye(2)
    target = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    result = match_embeddings(source, target)
    assert result.path == ((0, 0), (0, 1), (1, 2))
    assert result.total_cost == 0.0
    torch.testing.assert_close(result.cost, 1 - result.similarity)


def test_ties_are_deterministic() -> None:
    assert dtw_path(torch.zeros(3, 3)) == ((0, 0), (1, 1), (2, 2))


@pytest.mark.parametrize("invalid", [torch.zeros(2, 3), torch.full((2, 3), float("nan"))])
def test_invalid_embeddings_fail(invalid: torch.Tensor) -> None:
    with pytest.raises(ValueError, match=r"nonzero|finite"):
        match_embeddings(invalid, torch.ones(3, 3))
