"""Minimal discrete edge diffusion predictor.

This module is intentionally not wired into ``Decoder``, ``Criterion``, or the
existing ``GraphPredictor``.  It provides a standalone model skeleton for
experiments behind future configuration flags.
"""

from __future__ import annotations

from typing import Optional, Union

import torch
from torch import nn

from .edge_utils import EDGE_MASK_ID, EDGE_NUM_CLASSES


class EdgeDiffusionPredictor(nn.Module):
    """Predict denoised edge classes from atom states and noisy edge states.

    Args:
        hidden_size: Input atom hidden-state dimension ``D``.
        pair_hidden_size: Internal pair feature size ``H``.
        timestep_size: Dimension of learned timestep embeddings.
        max_timesteps: Number of supported discrete timesteps.

    Input shapes:
        ``hidden``: ``[B, K, D]``
        ``noisy_edges``: ``[B, K, K]`` with classes ``0`` through
        ``EDGE_MASK_ID``.
        ``timestep``: scalar or ``[B]``.
        ``valid_edge_mask``: optional ``[B, K, K]`` boolean mask.  Invalid logits
        are zeroed for hygiene; downstream losses should still use their own
        ignore mask.

    Output shape:
        ``edge_logits``: ``[B, 7, K, K]`` for public classes only.

    Memory complexity:
        The first implementation materializes pair tensors of shape
        ``[B, K, K, H]`` and therefore uses ``O(B * K^2 * H)`` memory and
        ``O(B * K^2 * H)`` MLP work.
    """

    def __init__(
        self,
        hidden_size: int,
        pair_hidden_size: Optional[int] = None,
        timestep_size: Optional[int] = None,
        max_timesteps: int = 1024,
    ) -> None:
        super(EdgeDiffusionPredictor, self).__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if max_timesteps <= 0:
            raise ValueError("max_timesteps must be positive")

        pair_hidden_size = pair_hidden_size or hidden_size
        timestep_size = timestep_size or hidden_size
        self.max_timesteps = max_timesteps
        self.hidden_proj = nn.Linear(hidden_size, pair_hidden_size)
        self.edge_embedding = nn.Embedding(EDGE_MASK_ID + 1, pair_hidden_size)
        self.timestep_embedding = nn.Embedding(max_timesteps, timestep_size)
        mlp_input_size = pair_hidden_size * 5 + timestep_size
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_size, pair_hidden_size),
            nn.GELU(),
            nn.Linear(pair_hidden_size, pair_hidden_size),
            nn.GELU(),
            nn.Linear(pair_hidden_size, EDGE_NUM_CLASSES),
        )

    def _prepare_timestep(
        self,
        timestep: Union[int, torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        timestep_tensor = torch.as_tensor(timestep, dtype=torch.long, device=device)
        if timestep_tensor.dim() == 0:
            timestep_tensor = timestep_tensor.expand(batch_size)
        elif timestep_tensor.shape != (batch_size,):
            raise ValueError("timestep must be a scalar or have shape [B]")
        if (timestep_tensor < 0).any() or (timestep_tensor >= self.max_timesteps).any():
            raise ValueError("timestep values must be in [0, max_timesteps - 1]")
        return timestep_tensor

    def forward(
        self,
        hidden: torch.Tensor,
        noisy_edges: torch.Tensor,
        timestep: Union[int, torch.Tensor],
        valid_edge_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return coupled public edge logits with shape ``[B, 7, K, K]``."""

        if hidden.dim() != 3:
            raise ValueError("hidden must have shape [B, K, D]")
        if noisy_edges.dim() != 3:
            raise ValueError("noisy_edges must have shape [B, K, K]")
        batch, size, _ = hidden.shape
        if noisy_edges.shape != (batch, size, size):
            raise ValueError("noisy_edges must have shape [B, K, K] matching hidden")
        if noisy_edges.min().item() < 0 or noisy_edges.max().item() > EDGE_MASK_ID:
            raise ValueError("noisy_edges must contain classes 0 through EDGE_MASK_ID")

        projected = self.hidden_proj(hidden)
        h_i = projected.unsqueeze(2).expand(batch, size, size, -1)
        h_j = projected.unsqueeze(1).expand(batch, size, size, -1)
        edge_features = self.edge_embedding(noisy_edges.long())
        timestep_ids = self._prepare_timestep(timestep, batch, hidden.device)
        timestep_features = self.timestep_embedding(timestep_ids).view(batch, 1, 1, -1)
        timestep_features = timestep_features.expand(batch, size, size, -1)
        pair_features = torch.cat(
            [
                h_i,
                h_j,
                torch.abs(h_i - h_j),
                h_i * h_j,
                edge_features,
                timestep_features,
            ],
            dim=-1,
        )

        logits = self.mlp(pair_features)
        logits = self._couple_directional_logits(logits)

        diag = torch.eye(size, dtype=torch.bool, device=hidden.device)
        logits = logits.masked_fill(diag.view(1, size, size, 1), 0.0)
        if valid_edge_mask is not None:
            if valid_edge_mask.shape != (batch, size, size):
                raise ValueError("valid_edge_mask must have shape [B, K, K]")
            logits = logits.masked_fill(~valid_edge_mask.to(dtype=torch.bool).unsqueeze(-1), 0.0)
        return logits.permute(0, 3, 1, 2).contiguous()

    def _couple_directional_logits(self, logits: torch.Tensor) -> torch.Tensor:
        coupled = logits.clone()
        reverse = logits.transpose(1, 2)
        for cls in range(5):
            value = 0.5 * (logits[..., cls] + reverse[..., cls])
            coupled[..., cls] = value
        cls_5 = 0.5 * (logits[..., 5] + reverse[..., 6])
        cls_6 = 0.5 * (logits[..., 6] + reverse[..., 5])
        coupled[..., 5] = cls_5
        coupled[..., 6] = cls_6
        return coupled
