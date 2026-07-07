from types import SimpleNamespace

import torch
from torch import nn

from molscribe.loss import Criterion
from molscribe.model import Decoder


class FakeAtomDecoder(nn.Module):

    def __init__(self, hidden):
        super(FakeAtomDecoder, self).__init__()
        self.register_buffer("hidden", hidden)

    def forward(self, encoder_out, labels, label_lengths):
        return None, None, self.hidden


def test_decoder_forward_keeps_default_edges_output_without_prepared_refs():
    decoder, hidden = _decoder(use_edge_diffusion=True, use_edge_diffusion_loss=True)
    refs = _refs(hidden)
    refs.pop("edge_diffusion")

    results = decoder(torch.empty(2, 1, 8), None, refs)
    predictions, targets = results["edges"]

    assert set(predictions.keys()) == {"edges"}
    assert set(targets.keys()) == {"edges"}


def test_decoder_forward_adds_edge_diffusion_outputs_for_prepared_refs():
    decoder, hidden = _decoder(use_edge_diffusion=True, use_edge_diffusion_loss=True)
    refs = _refs(hidden)

    results = decoder(torch.empty(2, 1, 8), None, refs)
    predictions, targets = results["edges"]

    assert predictions["edges"].shape == (2, 7, 3, 3)
    assert predictions["edge_diffusion"].shape == (2, 7, 3, 3)
    assert torch.equal(targets["edges"], refs["edges"])
    assert torch.equal(targets["edge_diffusion"], refs["edge_diffusion"]["loss_targets"])


def test_decoder_forward_edge_diffusion_outputs_work_with_criterion():
    decoder, hidden = _decoder(use_edge_diffusion=True, use_edge_diffusion_loss=True)
    criterion = Criterion(_args(use_edge_diffusion=True, use_edge_diffusion_loss=True), tokenizer={})
    refs = _refs(hidden)

    results = decoder(torch.empty(2, 1, 8), None, refs)
    losses = criterion({"edges": results["edges"]}, refs)
    loss = sum(losses.values())
    loss.backward()

    assert set(losses.keys()) == {"edges", "edge_diffusion"}
    assert torch.isfinite(loss)


def test_decoder_forward_ignores_prepared_refs_without_loss_flag():
    decoder, hidden = _decoder(use_edge_diffusion=True, use_edge_diffusion_loss=False)
    refs = _refs(hidden)

    results = decoder(torch.empty(2, 1, 8), None, refs)
    predictions, targets = results["edges"]

    assert "edge_diffusion" not in predictions
    assert "edge_diffusion" not in targets


def _decoder(use_edge_diffusion, use_edge_diffusion_loss):
    hidden = torch.randn(2, 5, 8)
    decoder = Decoder(_args(use_edge_diffusion, use_edge_diffusion_loss), tokenizer={})
    decoder.formats = ["atomtok_coords", "edges"]
    decoder.decoder["atomtok_coords"] = FakeAtomDecoder(hidden)
    return decoder, hidden


def _args(use_edge_diffusion, use_edge_diffusion_loss):
    return SimpleNamespace(
        formats=["edges"],
        dec_hidden_size=8,
        continuous_coords=False,
        compute_confidence=False,
        use_edge_diffusion=use_edge_diffusion,
        use_edge_diffusion_loss=use_edge_diffusion_loss,
        edge_diffusion_hidden_size=8,
        edge_diffusion_steps=8,
        label_smoothing=0.0,
    )


def _refs(hidden):
    atom_indices = torch.tensor([[0, 2, 4], [1, 3, 4]])
    edge_targets = torch.randint(0, 7, (2, 3, 3))
    loss_targets = torch.full((2, 3, 3), -100)
    loss_targets[0, 0, 1] = 2
    loss_targets[1, 1, 2] = 5
    noisy_edges = edge_targets.clone()
    valid_pair_mask = torch.zeros(2, 3, 3, dtype=torch.bool)
    valid_pair_mask[:, 0, 1] = True
    valid_pair_mask[:, 0, 2] = True
    valid_pair_mask[:, 1, 2] = True
    return {
        "atomtok_coords": [torch.zeros(2, 1, dtype=torch.long), torch.ones(2, dtype=torch.long)],
        "atom_indices": [atom_indices],
        "edges": edge_targets,
        "edge_diffusion": {
            "noisy_edges": noisy_edges,
            "timestep": torch.tensor([1, 2]),
            "loss_targets": loss_targets,
            "valid_pair_mask": valid_pair_mask,
        },
    }
