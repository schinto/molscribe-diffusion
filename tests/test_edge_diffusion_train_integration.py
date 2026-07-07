from types import SimpleNamespace

import torch
from torch import nn

from molscribe.diffusion.edge_training import prepare_optional_edge_diffusion_refs
from molscribe.diffusion.edge_utils import EDGE_IGNORE_INDEX, EDGE_MASK_ID, expand_upper_triangular_edges
from molscribe.model import Decoder


class FakeAtomDecoder(nn.Module):

    def __init__(self, hidden):
        super(FakeAtomDecoder, self).__init__()
        self.register_buffer("hidden", hidden)

    def forward(self, encoder_out, labels, label_lengths):
        return None, None, self.hidden


def test_prepare_optional_edge_diffusion_refs_is_noop_without_both_flags():
    refs = {"edges": _targets(batch_size=1)}

    assert prepare_optional_edge_diffusion_refs(
        refs,
        _args(use_edge_diffusion=False, use_edge_diffusion_loss=True),
    ) is refs
    assert prepare_optional_edge_diffusion_refs(
        refs,
        _args(use_edge_diffusion=True, use_edge_diffusion_loss=False),
    ) is refs
    assert "edge_diffusion" not in refs


def test_prepare_optional_edge_diffusion_refs_adds_training_batch():
    refs = {"edges": _targets(batch_size=2)}
    generator = torch.Generator().manual_seed(13)

    prepared = prepare_optional_edge_diffusion_refs(
        refs,
        _args(use_edge_diffusion=True, use_edge_diffusion_loss=True, edge_diffusion_steps=2),
        generator=generator,
    )
    edge_refs = prepared["edge_diffusion"]

    assert prepared is not refs
    assert "edge_diffusion" not in refs
    assert edge_refs["timestep"].shape == (2,)
    assert edge_refs["noisy_edges"].shape == refs["edges"].shape
    assert edge_refs["loss_targets"].shape == refs["edges"].shape
    assert edge_refs["valid_pair_mask"].shape == refs["edges"].shape
    assert edge_refs["noisy_edges"].min().item() >= 0
    assert edge_refs["noisy_edges"].max().item() <= EDGE_MASK_ID
    assert torch.equal(edge_refs["loss_targets"][~edge_refs["corrupted_pair_mask"]], torch.full_like(
        edge_refs["loss_targets"][~edge_refs["corrupted_pair_mask"]],
        EDGE_IGNORE_INDEX,
    ))


def test_prepare_optional_edge_diffusion_refs_keeps_existing_prepared_refs():
    existing = {"noisy_edges": _targets(batch_size=1)}
    refs = {
        "edges": _targets(batch_size=1),
        "edge_diffusion": existing,
    }

    prepared = prepare_optional_edge_diffusion_refs(
        refs,
        _args(use_edge_diffusion=True, use_edge_diffusion_loss=True),
    )

    assert prepared is refs
    assert prepared["edge_diffusion"] is existing


def test_prepared_training_refs_feed_decoder_forward_hook():
    hidden = torch.randn(2, 5, 8)
    refs = {
        "atomtok_coords": [torch.zeros(2, 1, dtype=torch.long), torch.ones(2, dtype=torch.long)],
        "atom_indices": [torch.tensor([[0, 2, 4], [1, 3, 4]])],
        "edges": _targets(batch_size=2),
    }
    args = _args(use_edge_diffusion=True, use_edge_diffusion_loss=True, edge_diffusion_steps=2)
    prepared = prepare_optional_edge_diffusion_refs(
        refs,
        args,
        generator=torch.Generator().manual_seed(7),
    )
    decoder = Decoder(args, tokenizer={})
    decoder.formats = ["atomtok_coords", "edges"]
    decoder.decoder["atomtok_coords"] = FakeAtomDecoder(hidden)

    results = decoder(torch.empty(2, 1, 8), None, prepared)
    predictions, targets = results["edges"]

    assert predictions["edge_diffusion"].shape == (2, 7, 3, 3)
    assert torch.equal(targets["edge_diffusion"], prepared["edge_diffusion"]["loss_targets"])


def _args(
    use_edge_diffusion,
    use_edge_diffusion_loss,
    edge_diffusion_steps=4,
):
    return SimpleNamespace(
        formats=["edges"],
        dec_hidden_size=8,
        continuous_coords=False,
        compute_confidence=False,
        use_edge_diffusion=use_edge_diffusion,
        use_edge_diffusion_loss=use_edge_diffusion_loss,
        edge_diffusion_hidden_size=8,
        edge_diffusion_steps=edge_diffusion_steps,
        edge_diffusion_schedule="linear",
        label_smoothing=0.0,
    )


def _targets(batch_size):
    upper = torch.full((batch_size, 3, 3), EDGE_IGNORE_INDEX)
    upper[:, 0, 1] = 1
    upper[:, 0, 2] = 5
    upper[:, 1, 2] = 6
    return expand_upper_triangular_edges(upper)
