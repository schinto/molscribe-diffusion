import pytest
import torch
import torch.nn.functional as F

from molscribe.diffusion.edge_loss import EdgeDiffusionLoss, edge_diffusion_loss
from molscribe.diffusion.edge_utils import EDGE_IGNORE_INDEX


def test_edge_diffusion_loss_matches_cross_entropy_on_active_targets():
    logits = torch.randn(2, 7, 3, 3, requires_grad=True)
    targets = torch.full((2, 3, 3), EDGE_IGNORE_INDEX)
    targets[0, 0, 1] = 2
    targets[1, 1, 2] = 5

    loss = edge_diffusion_loss(logits, targets)

    flat_logits = logits.permute(0, 2, 3, 1).reshape(-1, 7)
    flat_targets = targets.reshape(-1).long()
    expected = F.cross_entropy(flat_logits, flat_targets, ignore_index=EDGE_IGNORE_INDEX)
    assert torch.allclose(loss, expected)


def test_edge_diffusion_loss_backward_has_finite_gradients_only_from_active_targets():
    logits = torch.randn(1, 7, 3, 3, requires_grad=True)
    targets = torch.full((1, 3, 3), EDGE_IGNORE_INDEX)
    targets[0, 0, 2] = 6

    loss = EdgeDiffusionLoss()(logits, targets)
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[:, :, 0, 2].abs().sum().item() > 0
    ignored_grad = logits.grad.clone()
    ignored_grad[:, :, 0, 2] = 0
    assert ignored_grad.abs().sum().item() == pytest.approx(0.0)


def test_edge_diffusion_loss_returns_differentiable_zero_without_active_targets():
    logits = torch.randn(2, 7, 2, 2, requires_grad=True)
    targets = torch.full((2, 2, 2), EDGE_IGNORE_INDEX)

    loss = EdgeDiffusionLoss()(logits, targets)
    loss.backward()

    assert loss.item() == pytest.approx(0.0)
    assert logits.grad is not None
    assert logits.grad.abs().sum().item() == pytest.approx(0.0)


def test_edge_diffusion_loss_can_return_plain_zero_without_active_targets():
    logits = torch.randn(1, 7, 2, 2)
    targets = torch.full((1, 2, 2), EDGE_IGNORE_INDEX)

    loss = EdgeDiffusionLoss(empty_loss_requires_grad=False)(logits, targets)

    assert loss.shape == ()
    assert loss.item() == pytest.approx(0.0)


def test_edge_diffusion_loss_rejects_invalid_public_targets():
    logits = torch.randn(1, 7, 2, 2)
    targets = torch.full((1, 2, 2), EDGE_IGNORE_INDEX)
    targets[0, 0, 1] = 7

    with pytest.raises(ValueError, match="public classes"):
        EdgeDiffusionLoss()(logits, targets)


def test_edge_diffusion_loss_rejects_invalid_shapes():
    logits = torch.randn(1, 7, 2, 2)
    targets = torch.full((1, 2, 2), EDGE_IGNORE_INDEX)

    with pytest.raises(ValueError, match="seven"):
        EdgeDiffusionLoss()(logits[:, :6], targets)
    with pytest.raises(ValueError, match="matching"):
        EdgeDiffusionLoss()(logits, targets[:, :1])


def test_edge_diffusion_loss_accepts_class_weights():
    logits = torch.randn(1, 7, 2, 2)
    targets = torch.full((1, 2, 2), EDGE_IGNORE_INDEX)
    targets[0, 0, 1] = 3
    weights = torch.arange(1, 8, dtype=torch.float32)

    loss = EdgeDiffusionLoss(class_weights=weights)(logits, targets)

    expected = F.cross_entropy(
        logits.permute(0, 2, 3, 1).reshape(-1, 7),
        targets.reshape(-1).long(),
        weight=weights,
        ignore_index=EDGE_IGNORE_INDEX,
    )
    assert torch.allclose(loss, expected)


def test_edge_diffusion_loss_rejects_bad_class_weights():
    with pytest.raises(ValueError, match="shape"):
        EdgeDiffusionLoss(class_weights=torch.ones(6))
