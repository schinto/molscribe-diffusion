import pytest
import torch

from molscribe.diffusion.schedules import cosine_mask_schedule, linear_mask_schedule


@pytest.mark.parametrize("schedule", [linear_mask_schedule, cosine_mask_schedule])
def test_schedule_range_monotonic_first_and_last(schedule):
    values = schedule(torch.arange(10), 10)

    assert torch.all(values >= 0)
    assert torch.all(values <= 1)
    assert torch.all(values[1:] >= values[:-1])
    assert values[0].item() == pytest.approx(0.0)
    assert values[-1].item() == pytest.approx(1.0)


def test_linear_schedule_scalar_and_tensor_values():
    assert linear_mask_schedule(2, 5).item() == pytest.approx(0.5)

    values = linear_mask_schedule(torch.tensor([0, 2, 4]), 5)
    assert torch.allclose(values, torch.tensor([0.0, 0.5, 1.0]))


def test_cosine_schedule_scalar_and_tensor_values():
    assert cosine_mask_schedule(0, 5).item() == pytest.approx(0.0)

    values = cosine_mask_schedule(torch.tensor([0, 4]), 5)
    assert torch.allclose(values, torch.tensor([0.0, 1.0]), atol=1e-6)


@pytest.mark.parametrize("schedule", [linear_mask_schedule, cosine_mask_schedule])
def test_schedule_invalid_arguments(schedule):
    with pytest.raises(ValueError, match="at least 2"):
        schedule(0, 1)
    with pytest.raises(ValueError, match="inclusive range"):
        schedule(-1, 4)
    with pytest.raises(ValueError, match="inclusive range"):
        schedule(4, 4)
    with pytest.raises(ValueError, match="finite"):
        schedule(torch.tensor(float("nan")), 4)
    with pytest.raises(TypeError, match="integer"):
        schedule(0, 4.0)
