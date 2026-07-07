"""Mask schedules for discrete diffusion.

Timesteps are zero-based.  For ``num_steps`` total diffusion steps, timestep
``0`` returns a mask ratio of ``0`` and timestep ``num_steps - 1`` returns
``1``.  Intermediate ratios are monotonically increasing.
"""

from __future__ import annotations

import math
from typing import Union

import torch

Number = Union[int, float]


def _validate_inputs(timestep: Union[Number, torch.Tensor], num_steps: int) -> torch.Tensor:
    if not isinstance(num_steps, int):
        raise TypeError("num_steps must be an integer")
    if num_steps < 2:
        raise ValueError("num_steps must be at least 2 so first and last steps are defined")

    timestep_tensor = torch.as_tensor(timestep)
    if not torch.is_floating_point(timestep_tensor):
        timestep_tensor = timestep_tensor.to(dtype=torch.float32)
    if not torch.isfinite(timestep_tensor).all():
        raise ValueError("timestep must contain only finite values")
    if (timestep_tensor < 0).any() or (timestep_tensor > num_steps - 1).any():
        raise ValueError("timestep values must be in the inclusive range [0, num_steps - 1]")
    return timestep_tensor


def linear_mask_schedule(timestep: Union[Number, torch.Tensor], num_steps: int) -> torch.Tensor:
    """Return a linear mask ratio for a zero-based diffusion timestep.

    Args:
        timestep: Scalar or tensor of timesteps in ``[0, num_steps - 1]``.
        num_steps: Total number of diffusion steps.  Must be at least 2.

    Returns:
        Tensor with the same broadcast-free shape as ``timestep`` and values in
        ``[0, 1]``.  Step 0 is unmasked; the final step is fully masked.
    """

    timestep_tensor = _validate_inputs(timestep, num_steps)
    return timestep_tensor / float(num_steps - 1)


def cosine_mask_schedule(timestep: Union[Number, torch.Tensor], num_steps: int) -> torch.Tensor:
    """Return a cosine mask ratio for a zero-based diffusion timestep.

    Args:
        timestep: Scalar or tensor of timesteps in ``[0, num_steps - 1]``.
        num_steps: Total number of diffusion steps.  Must be at least 2.

    Returns:
        Tensor with values in ``[0, 1]`` using ``1 - cos(progress * pi / 2)``.
        Step 0 is unmasked; the final step is fully masked.
    """

    timestep_tensor = _validate_inputs(timestep, num_steps)
    progress = timestep_tensor / float(num_steps - 1)
    return 1.0 - torch.cos(progress * (math.pi / 2.0))
