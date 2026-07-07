from types import SimpleNamespace

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
