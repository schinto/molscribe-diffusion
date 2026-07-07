"""Utilities for discrete edge classes used by edge diffusion.

Edge targets use shape ``[B, K, K]``.  Public edge classes are ``0`` through
``6``.  ``EDGE_MASK_ID`` is an internal noising state only, and
``EDGE_IGNORE_INDEX`` marks padding or ignored pairs.  Independent denoising
pairs are represented by the upper triangular entries ``i < j``.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch

EDGE_NUM_CLASSES = 7
EDGE_MASK_ID = 7
EDGE_IGNORE_INDEX = -100


def _check_edge_tensor(edges: torch.Tensor, name: str = "edges") -> None:
    if edges.dim() != 3:
        raise ValueError(f"{name} must have shape [B, K, K]")
    if edges.size(1) != edges.size(2):
        raise ValueError(f"{name} must be square in its last two dimensions")


def transpose_edge_classes(edges: torch.Tensor) -> torch.Tensor:
    """Return edge classes with directional stereo classes inverted.

    Classes ``0`` through ``4`` are unchanged, class ``5`` maps to ``6``, and
    class ``6`` maps to ``5``.  ``EDGE_MASK_ID`` and ``EDGE_IGNORE_INDEX`` are
    preserved.  The operation is elementwise and does not transpose matrix axes.
    """

    transposed = edges.clone()
    transposed = torch.where(edges == 5, torch.full_like(edges, 6), transposed)
    transposed = torch.where(edges == 6, torch.full_like(edges, 5), transposed)
    return transposed


def _upper_triangular_mask(size: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1)


def valid_edge_pair_mask(targets: torch.Tensor, upper_only: bool = True) -> torch.Tensor:
    """Build a boolean mask for valid edge pairs from ``targets``.

    Args:
        targets: Edge target tensor with shape ``[B, K, K]``.
        upper_only: When true, only independent pairs ``i < j`` are returned.

    Returns:
        Boolean tensor with shape ``[B, K, K]``.  ``EDGE_IGNORE_INDEX``, invalid
        public classes, padding rows/columns inferred from ignored pairs, and
        the diagonal are false.
    """

    _check_edge_tensor(targets, "targets")
    valid_public = (targets >= 0) & (targets < EDGE_NUM_CLASSES)
    pair_mask = valid_public & valid_public.transpose(1, 2)

    size = targets.size(1)
    eye = torch.eye(size, dtype=torch.bool, device=targets.device).unsqueeze(0)
    pair_mask = pair_mask & ~eye
    if upper_only:
        pair_mask = pair_mask & _upper_triangular_mask(size, targets.device).unsqueeze(0)
    return pair_mask


def expand_upper_triangular_edges(
    upper_edges: torch.Tensor,
    diagonal_value: int = EDGE_IGNORE_INDEX,
) -> torch.Tensor:
    """Expand independent upper-triangle edge classes to a full matrix.

    Args:
        upper_edges: Tensor ``[B, K, K]``.  Entries ``i < j`` carry the
            independent edge classes.  Lower-triangle entries are ignored.
        diagonal_value: Value written to every diagonal entry.

    Returns:
        Full ``[B, K, K]`` edge matrix.  Classes ``0`` through ``4`` are mirrored
        symmetrically.  Class ``5`` is mirrored as ``6`` and class ``6`` as ``5``.
    """

    _check_edge_tensor(upper_edges, "upper_edges")
    full = upper_edges.clone()
    size = upper_edges.size(1)
    upper = _upper_triangular_mask(size, upper_edges.device)
    row, col = upper.nonzero(as_tuple=True)
    full[:, col, row] = transpose_edge_classes(upper_edges[:, row, col])

    diag = torch.arange(size, device=upper_edges.device)
    full[:, diag, diag] = diagonal_value
    return full


def check_edge_matrix_consistency(
    edges: torch.Tensor,
    atom_mask: Optional[torch.Tensor] = None,
) -> bool:
    """Check whether a full edge matrix has valid final edge semantics.

    Args:
        edges: Full edge matrix with shape ``[B, K, K]``.
        atom_mask: Optional boolean tensor ``[B, K]`` where true marks real
            atoms.  ``EDGE_IGNORE_INDEX`` is allowed only on diagonal entries
            and pairs touching padded atoms.

    Returns:
        True when the matrix has no self-bonds, no final ``EDGE_MASK_ID``, valid
        symmetry for classes ``0`` through ``4``, valid directional inversion for
        classes ``5`` and ``6``, and ignore indices only in allowed positions.
    """

    _check_edge_tensor(edges)
    batch, size, _ = edges.shape
    if atom_mask is None:
        atom_mask = torch.ones(batch, size, dtype=torch.bool, device=edges.device)
    else:
        if atom_mask.shape != (batch, size):
            raise ValueError("atom_mask must have shape [B, K]")
        atom_mask = atom_mask.to(dtype=torch.bool, device=edges.device)

    diag = torch.eye(size, dtype=torch.bool, device=edges.device).unsqueeze(0)
    active_pairs = atom_mask.unsqueeze(2) & atom_mask.unsqueeze(1) & ~diag
    ignored_allowed = ~active_pairs

    if (edges == EDGE_MASK_ID).any():
        return False
    if ((edges == EDGE_IGNORE_INDEX) & ~ignored_allowed).any():
        return False
    if ((edges != EDGE_IGNORE_INDEX) & ignored_allowed).any():
        return False
    public_or_ignore = ((edges >= 0) & (edges < EDGE_NUM_CLASSES)) | (edges == EDGE_IGNORE_INDEX)
    if not public_or_ignore.all():
        return False

    upper = _upper_triangular_mask(size, edges.device).unsqueeze(0) & active_pairs
    lhs = edges[upper]
    rhs = transpose_edge_classes(edges.transpose(1, 2))[upper]
    return bool(torch.equal(lhs, rhs))


def _mask_ratio_per_batch(
    mask_ratio: Union[float, torch.Tensor],
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    ratio = torch.as_tensor(mask_ratio, dtype=torch.float32, device=device)
    if ratio.dim() == 0:
        ratio = ratio.expand(batch_size)
    elif ratio.shape != (batch_size,):
        raise ValueError("mask_ratio must be a scalar or have shape [B]")
    if not torch.isfinite(ratio).all():
        raise ValueError("mask_ratio must contain only finite values")
    if (ratio < 0).any() or (ratio > 1).any():
        raise ValueError("mask_ratio values must be in [0, 1]")
    return ratio


def corrupt_edges(
    targets: torch.Tensor,
    mask_ratio: Union[float, torch.Tensor],
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply masked discrete corruption to independent edge pairs.

    Args:
        targets: Edge classes with shape ``[B, K, K]``.  Valid public classes are
            ``0`` through ``6``; ``EDGE_IGNORE_INDEX`` marks ignored pairs.
        mask_ratio: Scalar or per-example tensor ``[B]`` in ``[0, 1]``.
        generator: Optional PyTorch generator for reproducible pair sampling.

    Returns:
        ``noisy_edges`` with shape ``[B, K, K]`` where selected pairs are set to
        ``EDGE_MASK_ID`` in both directions, ``corrupted_pair_mask`` with shape
        ``[B, K, K]`` marking selected independent pairs ``i < j``, and
        ``valid_pair_mask`` with shape ``[B, K, K]`` marking all valid
        independent pairs.  Diagonal and padding entries are never changed.
    """

    _check_edge_tensor(targets, "targets")
    batch, size, _ = targets.shape
    valid_targets = ((targets >= 0) & (targets < EDGE_NUM_CLASSES)) | (targets == EDGE_IGNORE_INDEX)
    if not valid_targets.all():
        raise ValueError("targets must contain public classes 0 through 6 or EDGE_IGNORE_INDEX")
    ratio = _mask_ratio_per_batch(mask_ratio, batch, targets.device)
    valid_pair_mask = valid_edge_pair_mask(targets, upper_only=True)
    corrupted_pair_mask = torch.zeros_like(valid_pair_mask)

    for batch_idx in range(batch):
        positions = valid_pair_mask[batch_idx].nonzero(as_tuple=False)
        pair_count = positions.size(0)
        if pair_count == 0:
            continue
        if ratio[batch_idx].item() == 0:
            selected_count = 0
        elif ratio[batch_idx].item() == 1:
            selected_count = pair_count
        else:
            selected_count = int(torch.ceil(ratio[batch_idx] * pair_count).item())
        if selected_count == 0:
            continue
        perm = torch.randperm(pair_count, generator=generator, device=targets.device)
        selected = positions[perm[:selected_count]]
        corrupted_pair_mask[batch_idx, selected[:, 0], selected[:, 1]] = True

    noisy_edges = targets.clone()
    selected_batch, selected_rows, selected_cols = corrupted_pair_mask.nonzero(as_tuple=True)
    if selected_batch.numel() > 0:
        noisy_edges[selected_batch, selected_rows, selected_cols] = EDGE_MASK_ID
        noisy_edges[selected_batch, selected_cols, selected_rows] = EDGE_MASK_ID
    return noisy_edges, corrupted_pair_mask, valid_pair_mask
