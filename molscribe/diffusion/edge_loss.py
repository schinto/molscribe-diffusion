"""Isolated loss helpers for discrete edge diffusion.

The loss is used only by the optional ``GraphLoss`` edge-diffusion path and
computes cross-entropy on sparse targets prepared by ``edge_training``.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .edge_utils import EDGE_IGNORE_INDEX, EDGE_NUM_CLASSES


class EdgeDiffusionLoss(nn.Module):
    """Cross-entropy loss for sparse edge-diffusion denoising targets.

    Args:
        class_weights: Optional tensor with shape ``[7]`` for public edge
            classes.  When omitted, all public classes are weighted equally.
        ignore_index: Target value ignored by the loss.  Defaults to
            ``EDGE_IGNORE_INDEX``.
        empty_loss_requires_grad: When true, batches with no valid denoising
            targets return a differentiable zero scalar connected to
            ``edge_logits``.

    Input shapes:
        ``edge_logits``: ``[B, 7, K, K]`` public-class logits.
        ``loss_targets``: ``[B, K, K]`` sparse targets.  Entries selected for
        denoising contain public classes ``0`` through ``6``.  All ignored
        entries, including diagonal, lower triangle, and padding, should be
        ``EDGE_IGNORE_INDEX``.

    Returns:
        Scalar loss.  Only ``loss_targets != ignore_index`` contributes.

    Memory complexity:
        Flattens logits to ``[B * K * K, 7]`` and targets to ``[B * K * K]``;
        memory and work are ``O(B * K^2)``.
    """

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: int = EDGE_IGNORE_INDEX,
        empty_loss_requires_grad: bool = True,
    ) -> None:
        super(EdgeDiffusionLoss, self).__init__()
        self.ignore_index = ignore_index
        self.empty_loss_requires_grad = empty_loss_requires_grad
        if class_weights is not None:
            if class_weights.shape != (EDGE_NUM_CLASSES,):
                raise ValueError("class_weights must have shape [7]")
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

    def forward(self, edge_logits: torch.Tensor, loss_targets: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy over non-ignored sparse edge targets."""

        _validate_shapes(edge_logits, loss_targets)
        active_targets = loss_targets != self.ignore_index
        if not active_targets.any():
            if self.empty_loss_requires_grad:
                return edge_logits.sum() * 0.0
            return edge_logits.new_zeros(())

        invalid_targets = active_targets & ((loss_targets < 0) | (loss_targets >= EDGE_NUM_CLASSES))
        if invalid_targets.any():
            raise ValueError("loss_targets must contain public classes 0 through 6 or ignore_index")

        logits = edge_logits.permute(0, 2, 3, 1).reshape(-1, EDGE_NUM_CLASSES)
        targets = loss_targets.reshape(-1).long()
        return F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights,
            ignore_index=self.ignore_index,
        )


def edge_diffusion_loss(edge_logits: torch.Tensor, loss_targets: torch.Tensor) -> torch.Tensor:
    """Functional wrapper for ``EdgeDiffusionLoss`` with default settings."""

    return EdgeDiffusionLoss()(edge_logits, loss_targets)


def _validate_shapes(edge_logits: torch.Tensor, loss_targets: torch.Tensor) -> None:
    if edge_logits.dim() != 4:
        raise ValueError("edge_logits must have shape [B, 7, K, K]")
    if edge_logits.size(1) != EDGE_NUM_CLASSES:
        raise ValueError("edge_logits must have seven public edge classes")
    if edge_logits.size(2) != edge_logits.size(3):
        raise ValueError("edge_logits must be square in its last two dimensions")
    expected_targets_shape = (edge_logits.size(0), edge_logits.size(2), edge_logits.size(3))
    if loss_targets.shape != expected_targets_shape:
        raise ValueError("loss_targets must have shape [B, K, K] matching edge_logits")
