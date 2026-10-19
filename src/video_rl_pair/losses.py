# Copyright (c) 2026 video-rl-pair contributors.
"""The paper equations: contrastive costs -> smoothDTW -> global cycles.

Rows always index the source sequence; columns index the destination.
The dynamic program deliberately uses ordinary PyTorch operations so its
entire derivative, including the SmoothMin weights, is visible to autograd.
"""

from dataclasses import dataclass
from typing import final

import torch
from torch import Tensor, nn
from torch.nn import functional
from typing_extensions import override

from video_rl_pair.config import LossConfig


def matching_cost(x: Tensor, y: Tensor, beta: float) -> Tensor:
    """Eq. 8: C[i,j] = -log softmax_j(cosine(x_i, y_j) / beta).

    Inputs: [..., M, D] and [..., N, D]. Output: [..., M, N].
    Log-softmax avoids explicitly forming tiny probabilities.
    """
    similarity = functional.normalize(x, dim=-1) @ functional.normalize(y, dim=-1).transpose(-1, -2)
    return -functional.log_softmax(similarity / beta, dim=-1)


def smooth_min(predecessors: Tensor, gamma: float) -> Tensor:
    """Eq. 5: softmax(-r/gamma) weighted MEAN of r, not log-sum-exp.

    Invalid boundary predecessors are +inf. Masking them out of the weighted
    values prevents 0 * inf from producing NaN. At least one must be finite.
    """
    weights = functional.softmax(-predecessors / gamma, dim=-1)
    values = predecessors.masked_fill(torch.isinf(predecessors), 0)
    return (weights * values).sum(dim=-1)


def smooth_dtw(cost: Tensor, gamma: float) -> Tensor:
    """Return the accumulated table R, with the same [..., M, N] shape as C.

    R(0,0)=0 in the padded recurrence; all other padded border cells are
    unreachable. Here the border is implicit. Every valid cell has up to
    three predecessors: diagonal, above, and left. No in-place graph edits.
    """
    rows, columns = cost.shape[-2:]
    zero = torch.zeros_like(cost[..., 0, 0])
    infinity = torch.full_like(zero, float("inf"))
    previous = [zero, *([infinity] * columns)]
    table: list[Tensor] = []
    for row in range(rows):
        current = [infinity]
        for column in range(columns):
            neighbors = torch.stack((previous[column], previous[column + 1], current[-1]), dim=-1)
            current.append(cost[..., row, column] + smooth_min(neighbors, gamma))
        table.append(torch.stack(current[1:], dim=-1))
        previous = current
    return torch.stack(table, dim=-2)


def global_cycle_loss(forward: Tensor, reverse: Tensor, alpha: float) -> Tensor:
    """Eqs. 10-13: sum_i -log sum_j p(Y_j|X_i) p(X_i|Y_j).

    Inputs are accumulated DTW costs. Return one value per pair. Computing
    the diagonal of the round-trip matrix in log space avoids underflow.
    """
    log_xy = functional.log_softmax(-forward / alpha, dim=-1)
    log_yx = functional.log_softmax(-reverse / alpha, dim=-1)
    log_return = torch.logsumexp(log_xy + log_yx.transpose(-1, -2), dim=-1)
    return -log_return.sum(dim=-1)


@dataclass(frozen=True, slots=True)
class LossTerms:
    """Pair-averaged losses, before any head-weight regularization."""

    total: Tensor
    alignment: Tensor
    cycle: Tensor


@final
class AlignmentLoss(nn.Module):
    """Bidirectional alignment plus the X -> Y -> X cycle in the text."""

    def __init__(self, config: LossConfig) -> None:
        """Set the three temperatures and two loss coefficients."""
        super().__init__()
        self.config = config

    @override
    def forward(self, x: Tensor, y: Tensor) -> LossTerms:
        """Compare corresponding pairs [..., M, D] and [..., N, D]."""
        cfg = self.config
        xy = smooth_dtw(matching_cost(x, y, cfg.beta), cfg.gamma)
        yx = smooth_dtw(matching_cost(y, x, cfg.beta), cfg.gamma)
        alignment = (xy[..., -1, -1] + yx[..., -1, -1]).mean()
        cycle = global_cycle_loss(xy, yx, cfg.alpha).mean()
        return LossTerms(
            total=cfg.alignment_weight * alignment + cfg.cycle_weight * cycle,
            alignment=alignment,
            cycle=cycle,
        )

    def from_batch(self, embeddings: Tensor) -> LossTerms:
        """Pair sequence 0 against each other same-action sequence.

        A local batch [B,T,D] yields B-1 pairs. DDP averages losses across
        ranks; it does not create pairs between different ranks.
        """
        if embeddings.ndim != 3 or embeddings.shape[0] < 2:
            message = "Expected embeddings [B,T,D] with at least two same-action videos."
            raise ValueError(message)
        anchors = embeddings[:1].expand(embeddings.shape[0] - 1, -1, -1)
        terms: LossTerms = self(anchors, embeddings[1:])
        return terms

    def from_pairs(self, embeddings: Tensor) -> LossTerms:
        """Compare only supplied pairs [P,2,T,D], even across different tasks."""
        if embeddings.ndim != 4 or embeddings.shape[0] < 1 or embeddings.shape[1] != 2:
            message = "Expected paired embeddings [P,2,T,D]."
            raise ValueError(message)
        terms: LossTerms = self(embeddings[:, 0], embeddings[:, 1])
        return terms
