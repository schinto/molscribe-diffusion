import pytest
import torch

from molscribe.diffusion.edge_training import (
    edge_diffusion_loss_targets,
    prepare_edge_diffusion_training_batch,
)
from molscribe.diffusion.edge_utils import (
    EDGE_IGNORE_INDEX,
    EDGE_MASK_ID,
    expand_upper_triangular_edges,
)


def test_edge_diffusion_loss_targets_keep_only_corrupted_upper_pairs():
    targets = _targets()
    corrupted_pair_mask = torch.zeros_like(targets, dtype=torch.bool)
    corrupted_pair_mask[0, 0, 2] = True

    loss_targets = edge_diffusion_loss_targets(targets, corrupted_pair_mask)

    assert loss_targets[0, 0, 2].item() == targets[0, 0, 2].item()
    assert loss_targets[0, 2, 0].item() == EDGE_IGNORE_INDEX
    assert torch.equal(torch.diagonal(loss_targets[0]), torch.full((3,), EDGE_IGNORE_INDEX))


def test_prepare_training_batch_first_linear_step_masks_nothing():
    targets = _targets()

    batch = prepare_edge_diffusion_training_batch(targets, timestep=0, num_steps=4)

    assert batch["mask_ratio"].item() == pytest.approx(0.0)
    assert torch.equal(batch["noisy_edges"], targets)
    assert not batch["corrupted_pair_mask"].any()
    assert torch.equal(batch["loss_targets"], torch.full_like(targets, EDGE_IGNORE_INDEX))
    assert batch["valid_pair_mask"].sum().item() == 3


def test_prepare_training_batch_last_linear_step_masks_all_valid_pairs():
    targets = _targets()

    batch = prepare_edge_diffusion_training_batch(targets, timestep=3, num_steps=4)

    assert batch["mask_ratio"].item() == pytest.approx(1.0)
    assert torch.equal(batch["corrupted_pair_mask"], batch["valid_pair_mask"])
    assert batch["noisy_edges"][0, 0, 1].item() == EDGE_MASK_ID
    assert batch["noisy_edges"][0, 1, 0].item() == EDGE_MASK_ID
    assert batch["loss_targets"][0, 0, 1].item() == targets[0, 0, 1].item()
    assert batch["loss_targets"][0, 1, 0].item() == EDGE_IGNORE_INDEX


def test_prepare_training_batch_cosine_and_generator_are_reproducible():
    targets = torch.cat([_targets(), _targets()], dim=0)
    timestep = torch.tensor([1, 2])
    generator_a = torch.Generator().manual_seed(99)
    generator_b = torch.Generator().manual_seed(99)

    batch_a = prepare_edge_diffusion_training_batch(
        targets,
        timestep=timestep,
        num_steps=4,
        schedule="cosine",
        generator=generator_a,
    )
    batch_b = prepare_edge_diffusion_training_batch(
        targets,
        timestep=timestep,
        num_steps=4,
        schedule="cosine",
        generator=generator_b,
    )

    assert torch.equal(batch_a["noisy_edges"], batch_b["noisy_edges"])
    assert torch.equal(batch_a["corrupted_pair_mask"], batch_b["corrupted_pair_mask"])
    assert batch_a["mask_ratio"].shape == (2,)
    assert torch.all(batch_a["mask_ratio"] >= 0)
    assert torch.all(batch_a["mask_ratio"] <= 1)


def test_prepare_training_batch_handles_no_valid_pairs():
    targets = torch.full((1, 2, 2), EDGE_IGNORE_INDEX)

    batch = prepare_edge_diffusion_training_batch(targets, timestep=3, num_steps=4)

    assert torch.equal(batch["noisy_edges"], targets)
    assert not batch["corrupted_pair_mask"].any()
    assert not batch["valid_pair_mask"].any()
    assert torch.equal(batch["loss_targets"], targets)


def test_prepare_training_batch_rejects_unknown_schedule():
    with pytest.raises(ValueError, match="schedule"):
        prepare_edge_diffusion_training_batch(_targets(), timestep=0, num_steps=4, schedule="bad")


def _targets():
    upper = torch.full((1, 3, 3), EDGE_IGNORE_INDEX)
    upper[0, 0, 1] = 1
    upper[0, 0, 2] = 5
    upper[0, 1, 2] = 6
    return expand_upper_triangular_edges(upper)
