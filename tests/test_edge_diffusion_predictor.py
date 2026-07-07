import pytest
import torch

from molscribe.diffusion.edge_diffusion import EdgeDiffusionPredictor
from molscribe.diffusion.edge_utils import EDGE_IGNORE_INDEX, EDGE_MASK_ID, expand_upper_triangular_edges


@pytest.mark.parametrize("size", [1, 4])
def test_edge_diffusion_predictor_forward_backward_cpu(size):
    batch_size = 2
    hidden_size = 8
    hidden = torch.randn(batch_size, size, hidden_size, requires_grad=True)
    noisy_edges = _noisy_edges(batch_size, size)
    valid_mask = noisy_edges != EDGE_IGNORE_INDEX

    predictor = EdgeDiffusionPredictor(hidden_size, pair_hidden_size=12, max_timesteps=8)
    logits = predictor(hidden, noisy_edges.clamp_min(0), torch.tensor([1, 3]), valid_mask)

    assert logits.shape == (batch_size, 7, size, size)
    assert torch.isfinite(logits).all()
    assert not torch.isnan(logits).any()

    loss = logits.square().mean()
    loss.backward()

    assert hidden.grad is not None
    assert torch.isfinite(hidden.grad).all()
    for parameter in predictor.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_edge_diffusion_predictor_accepts_scalar_timestep_and_no_valid_pairs():
    hidden = torch.randn(2, 1, 6)
    noisy_edges = torch.zeros(2, 1, 1, dtype=torch.long)
    valid_mask = torch.zeros(2, 1, 1, dtype=torch.bool)
    predictor = EdgeDiffusionPredictor(6, max_timesteps=4)

    logits = predictor(hidden, noisy_edges, 0, valid_mask)

    assert logits.shape == (2, 7, 1, 1)
    assert torch.equal(logits, torch.zeros_like(logits))


def test_edge_diffusion_predictor_couples_symmetric_and_stereo_logits():
    hidden = torch.randn(2, 4, 8)
    noisy_edges = _noisy_edges(2, 4).clamp_min(0)
    predictor = EdgeDiffusionPredictor(8, pair_hidden_size=10, max_timesteps=8)

    logits = predictor(hidden, noisy_edges, torch.tensor([2, 3]))

    for cls in range(5):
        assert torch.allclose(logits[:, cls], logits[:, cls].transpose(1, 2), atol=1e-6)
    assert torch.allclose(logits[:, 5], logits[:, 6].transpose(1, 2), atol=1e-6)
    assert torch.allclose(logits[:, 6], logits[:, 5].transpose(1, 2), atol=1e-6)


def test_edge_diffusion_predictor_rejects_invalid_noisy_class_and_timestep():
    hidden = torch.randn(1, 2, 4)
    noisy_edges = torch.zeros(1, 2, 2, dtype=torch.long)
    predictor = EdgeDiffusionPredictor(4, max_timesteps=2)

    with pytest.raises(ValueError, match="classes"):
        predictor(hidden, noisy_edges + EDGE_MASK_ID + 1, 0)
    with pytest.raises(ValueError, match="max_timesteps"):
        predictor(hidden, noisy_edges, 2)


def _noisy_edges(batch_size, size):
    upper = torch.full((batch_size, size, size), EDGE_IGNORE_INDEX)
    for row in range(size):
        for col in range(row + 1, size):
            upper[:, row, col] = (row + col) % 7
    full = expand_upper_triangular_edges(upper)
    if size > 1:
        full[:, 0, 1] = EDGE_MASK_ID
        full[:, 1, 0] = EDGE_MASK_ID
    return full
