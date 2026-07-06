import importlib.util
import sys
import types
from pathlib import Path

import torch


def load_graph_predictor():
    repo_root = Path(__file__).resolve().parents[1]
    package = types.ModuleType("molscribe")
    package.__path__ = [str(repo_root / "molscribe")]
    sys.modules.setdefault("molscribe", package)

    spec = importlib.util.spec_from_file_location(
        "molscribe.model",
        repo_root / "molscribe" / "model.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["molscribe.model"] = module
    spec.loader.exec_module(module)
    return module.GraphPredictor


def test_graph_predictor_forward_backward_with_atom_indices():
    GraphPredictor = load_graph_predictor()
    batch_size = 2
    seq_len = 8
    hidden_size = 16
    atom_count = 3

    hidden = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
    atom_indices = torch.tensor(
        [
            [1, 3, 6],
            [0, 4, 7],
        ],
        dtype=torch.long,
    )

    predictor = GraphPredictor(hidden_size)
    outputs = predictor(hidden, indices=atom_indices)
    logits = outputs["edges"]

    assert logits.shape == (batch_size, 7, atom_count, atom_count)
    assert torch.isfinite(logits).all()

    loss = logits.square().mean()
    loss.backward()

    assert hidden.grad is not None
    assert torch.isfinite(hidden.grad).all()
    for parameter in predictor.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
