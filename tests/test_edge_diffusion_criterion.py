from types import SimpleNamespace

import pytest
import torch

from molscribe.diffusion.edge_loss import EdgeDiffusionLoss
from molscribe.loss import Criterion, GraphLoss


def test_criterion_keeps_graph_loss_as_default_for_edges():
    criterion = Criterion(_args(use_edge_diffusion_loss=False), tokenizer={})

    assert isinstance(criterion.criterion["edges"], GraphLoss)
    assert criterion.criterion["edges"].edge_diffusion_criterion is None


def test_criterion_accepts_args_without_edge_diffusion_loss_flag():
    args = SimpleNamespace(formats=["edges"], label_smoothing=0.0)

    criterion = Criterion(args, tokenizer={})

    assert isinstance(criterion.criterion["edges"], GraphLoss)
    assert criterion.criterion["edges"].edge_diffusion_criterion is None


def test_criterion_enables_edge_diffusion_loss_behind_flag():
    criterion = Criterion(_args(use_edge_diffusion_loss=True), tokenizer={})

    assert isinstance(criterion.criterion["edges"].edge_diffusion_criterion, EdgeDiffusionLoss)


def test_graph_loss_computes_baseline_edges_unchanged_with_flag_disabled():
    outputs = {"edges": torch.randn(2, 7, 3, 3, requires_grad=True)}
    targets = {"edges": torch.randint(0, 7, (2, 3, 3))}
    default_loss = GraphLoss()(outputs, targets)["edges"]
    flagged_loss = GraphLoss(use_edge_diffusion_loss=True)(outputs, targets)["edges"]

    assert torch.allclose(default_loss, flagged_loss)


def test_graph_loss_computes_optional_edge_diffusion_loss_with_flag():
    outputs = {
        "edge_diffusion": torch.randn(2, 7, 3, 3, requires_grad=True),
    }
    targets = {
        "edge_diffusion": torch.full((2, 3, 3), -100),
    }
    targets["edge_diffusion"][0, 0, 1] = 2
    targets["edge_diffusion"][1, 1, 2] = 5

    losses = GraphLoss(use_edge_diffusion_loss=True)(outputs, targets)
    loss = losses["edge_diffusion"]
    loss.backward()

    assert torch.isfinite(loss)
    assert outputs["edge_diffusion"].grad is not None
    assert torch.isfinite(outputs["edge_diffusion"].grad).all()


def test_graph_loss_rejects_edge_diffusion_outputs_without_flag():
    outputs = {"edge_diffusion": torch.randn(1, 7, 2, 2)}
    targets = {"edge_diffusion": torch.full((1, 2, 2), -100)}

    with pytest.raises(ValueError, match="use_edge_diffusion_loss"):
        GraphLoss()(outputs, targets)


def test_criterion_forward_returns_edge_diffusion_loss_for_matching_inputs():
    criterion = Criterion(_args(use_edge_diffusion_loss=True), tokenizer={})
    outputs = {"edge_diffusion": torch.randn(1, 7, 2, 2, requires_grad=True)}
    targets = {"edge_diffusion": torch.full((1, 2, 2), -100)}
    targets["edge_diffusion"][0, 0, 1] = 3
    results = {"edges": (outputs, targets)}

    losses = criterion(results, refs={})

    assert set(losses.keys()) == {"edge_diffusion"}
    assert torch.isfinite(losses["edge_diffusion"])


def _args(use_edge_diffusion_loss):
    return SimpleNamespace(
        formats=["edges"],
        label_smoothing=0.0,
        use_edge_diffusion_loss=use_edge_diffusion_loss,
    )
