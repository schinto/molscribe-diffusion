from types import SimpleNamespace

import torch

from molscribe.diffusion.edge_diffusion import EdgeDiffusionPredictor
from molscribe.model import Decoder, GraphPredictor


def test_decoder_keeps_graph_predictor_as_default_edge_decoder():
    decoder = Decoder(_args(use_edge_diffusion=False), tokenizer={})

    assert list(decoder.decoder.keys()) == ["edges"]
    assert isinstance(decoder.decoder["edges"], GraphPredictor)


def test_decoder_instantiates_optional_edge_diffusion_predictor_behind_flag():
    decoder = Decoder(
        _args(
            use_edge_diffusion=True,
            edge_diffusion_hidden_size=12,
            edge_diffusion_steps=16,
        ),
        tokenizer={},
    )

    assert isinstance(decoder.decoder["edges"], GraphPredictor)
    assert isinstance(decoder.decoder["edge_diffusion"], EdgeDiffusionPredictor)
    assert decoder.decoder["edge_diffusion"].max_timesteps == 16


def test_decoder_is_compatible_with_args_without_edge_diffusion_fields():
    args = SimpleNamespace(
        formats=["edges"],
        dec_hidden_size=8,
        continuous_coords=False,
        compute_confidence=False,
    )

    decoder = Decoder(args, tokenizer={})

    assert list(decoder.decoder.keys()) == ["edges"]
    assert isinstance(decoder.decoder["edges"], GraphPredictor)


def test_optional_edge_diffusion_does_not_change_graph_predictor_outputs():
    torch.manual_seed(7)
    baseline_decoder = Decoder(_args(use_edge_diffusion=False), tokenizer={})
    diffusion_decoder = Decoder(_args(use_edge_diffusion=True), tokenizer={})
    diffusion_decoder.decoder["edges"].load_state_dict(baseline_decoder.decoder["edges"].state_dict())

    hidden = torch.randn(2, 5, 8)
    atom_indices = torch.tensor([[0, 2, 4], [1, 3, 4]])

    baseline_edges = baseline_decoder.decoder["edges"](hidden, indices=atom_indices)["edges"]
    diffusion_edges = diffusion_decoder.decoder["edges"](hidden, indices=atom_indices)["edges"]

    assert torch.equal(baseline_edges, diffusion_edges)


def test_optional_decoder_loads_baseline_state_dict_with_strict_false():
    baseline_decoder = Decoder(_args(use_edge_diffusion=False), tokenizer={})
    diffusion_decoder = Decoder(_args(use_edge_diffusion=True), tokenizer={})

    incompatible = diffusion_decoder.load_state_dict(baseline_decoder.state_dict(), strict=False)

    assert incompatible.unexpected_keys == []
    assert incompatible.missing_keys
    assert all(key.startswith("decoder.edge_diffusion.") for key in incompatible.missing_keys)


def test_baseline_decoder_loads_optional_state_dict_with_strict_false():
    baseline_decoder = Decoder(_args(use_edge_diffusion=False), tokenizer={})
    diffusion_decoder = Decoder(_args(use_edge_diffusion=True), tokenizer={})

    incompatible = baseline_decoder.load_state_dict(diffusion_decoder.state_dict(), strict=False)

    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys
    assert all(key.startswith("decoder.edge_diffusion.") for key in incompatible.unexpected_keys)


def _args(
    use_edge_diffusion,
    edge_diffusion_hidden_size=None,
    edge_diffusion_steps=1024,
):
    return SimpleNamespace(
        formats=["edges"],
        dec_hidden_size=8,
        continuous_coords=False,
        compute_confidence=False,
        use_edge_diffusion=use_edge_diffusion,
        edge_diffusion_hidden_size=edge_diffusion_hidden_size,
        edge_diffusion_steps=edge_diffusion_steps,
    )
