"""Training preparation helpers for discrete edge diffusion.

These helpers are intentionally not integrated into ``Criterion`` or
``Decoder`` yet.  They convert full edge targets ``[B, K, K]`` into noisy edge
inputs and sparse denoising targets for independent upper-triangle pairs.
"""

from __future__ import annotations

from typing import Callable, Dict, Union

import torch

from .edge_utils import EDGE_IGNORE_INDEX, corrupt_edges
from .schedules import cosine_mask_schedule, linear_mask_schedule

Schedule = Union[str, Callable[[Union[int, torch.Tensor], int], torch.Tensor]]


def edge_diffusion_loss_targets(
    targets: torch.Tensor,
    corrupted_pair_mask: torch.Tensor,
) -> torch.Tensor:
    """Build sparse denoising targets for selected independent edge pairs.

    Args:
        targets: Full edge target matrix with shape ``[B, K, K]``.
        corrupted_pair_mask: Boolean mask with shape ``[B, K, K]``.  True values
            mark independent upper-triangle pairs ``i < j`` selected for the
            denoising loss.

    Returns:
        Tensor with shape ``[B, K, K]``.  Entries selected by
        ``corrupted_pair_mask`` contain the original public edge class from
        ``targets``.  Every other entry, including diagonal, lower triangle, and
        padding, is ``EDGE_IGNORE_INDEX``.  This keeps the later loss restricted
        to the independent pair set.
    """

    if targets.shape != corrupted_pair_mask.shape:
        raise ValueError("targets and corrupted_pair_mask must have the same shape [B, K, K]")
    if targets.dim() != 3 or targets.size(1) != targets.size(2):
        raise ValueError("targets must have shape [B, K, K]")

    loss_targets = torch.full_like(targets, EDGE_IGNORE_INDEX)
    loss_targets[corrupted_pair_mask.to(dtype=torch.bool)] = targets[corrupted_pair_mask.to(dtype=torch.bool)]
    return loss_targets


def prepare_edge_diffusion_training_batch(
    targets: torch.Tensor,
    timestep: Union[int, torch.Tensor],
    num_steps: int,
    schedule: Schedule = "linear",
    generator: torch.Generator | None = None,
) -> Dict[str, torch.Tensor]:
    """Prepare noisy edge inputs and sparse targets for edge diffusion training.

    Args:
        targets: Full edge target matrix with shape ``[B, K, K]``.  Public edge
            classes are ``0`` through ``6``; ``EDGE_IGNORE_INDEX`` marks padding
            or ignored entries.
        timestep: Scalar or per-example tensor ``[B]`` using zero-based
            diffusion steps.
        num_steps: Total number of diffusion steps passed to the mask schedule.
        schedule: ``"linear"``, ``"cosine"``, or a callable returning a mask
            ratio tensor in ``[0, 1]``.
        generator: Optional PyTorch generator for reproducible corruption.

    Returns:
        A dictionary of tensors:

        ``noisy_edges``:
            ``[B, K, K]`` targets with selected pairs replaced by
            ``EDGE_MASK_ID`` in both directions.
        ``loss_targets``:
            ``[B, K, K]`` sparse targets with original classes only at selected
            independent upper-triangle pairs; all other entries are
            ``EDGE_IGNORE_INDEX``.
        ``corrupted_pair_mask``:
            ``[B, K, K]`` boolean mask for selected independent pairs ``i < j``.
        ``valid_pair_mask``:
            ``[B, K, K]`` boolean mask for all valid independent pairs.
        ``mask_ratio``:
            Scalar or ``[B]`` tensor returned by the schedule.

    Memory complexity:
        Stores several ``[B, K, K]`` tensors, so memory is ``O(B * K^2)``.
    """

    mask_ratio = _resolve_schedule(schedule)(timestep, num_steps)
    noisy_edges, corrupted_pair_mask, valid_pair_mask = corrupt_edges(
        targets,
        mask_ratio,
        generator=generator,
    )
    loss_targets = edge_diffusion_loss_targets(targets, corrupted_pair_mask)
    return {
        "noisy_edges": noisy_edges,
        "loss_targets": loss_targets,
        "corrupted_pair_mask": corrupted_pair_mask,
        "valid_pair_mask": valid_pair_mask,
        "mask_ratio": mask_ratio,
    }


def _resolve_schedule(schedule: Schedule) -> Callable[[Union[int, torch.Tensor], int], torch.Tensor]:
    if schedule == "linear":
        return linear_mask_schedule
    if schedule == "cosine":
        return cosine_mask_schedule
    if callable(schedule):
        return schedule
    raise ValueError("schedule must be 'linear', 'cosine', or a callable")
