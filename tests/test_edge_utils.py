import torch

from molscribe.diffusion.edge_utils import (
    EDGE_IGNORE_INDEX,
    EDGE_MASK_ID,
    corrupt_edges,
    check_edge_matrix_consistency,
    expand_upper_triangular_edges,
    transpose_edge_classes,
    valid_edge_pair_mask,
)


def test_transpose_edge_classes_preserves_and_inverts_expected_classes():
    edges = torch.tensor([0, 1, 2, 3, 4, 5, 6, EDGE_MASK_ID, EDGE_IGNORE_INDEX])

    transposed = transpose_edge_classes(edges)

    assert torch.equal(transposed[:5], edges[:5])
    assert transposed[5].item() == 6
    assert transposed[6].item() == 5
    assert transposed[7].item() == EDGE_MASK_ID
    assert transposed[8].item() == EDGE_IGNORE_INDEX


def test_valid_edge_pair_mask_excludes_ignore_padding_diagonal_and_lower_triangle():
    targets = torch.tensor(
        [
            [
                [EDGE_IGNORE_INDEX, 1, 5, EDGE_IGNORE_INDEX],
                [1, EDGE_IGNORE_INDEX, 2, EDGE_IGNORE_INDEX],
                [6, 2, EDGE_IGNORE_INDEX, EDGE_IGNORE_INDEX],
                [EDGE_IGNORE_INDEX, EDGE_IGNORE_INDEX, EDGE_IGNORE_INDEX, EDGE_IGNORE_INDEX],
            ]
        ]
    )

    mask = valid_edge_pair_mask(targets)

    expected = torch.tensor(
        [
            [
                [False, True, True, False],
                [False, False, True, False],
                [False, False, False, False],
                [False, False, False, False],
            ]
        ]
    )
    assert torch.equal(mask, expected)


def test_expand_upper_triangular_edges_mirrors_symmetry_and_stereo_direction():
    upper = torch.full((1, 3, 3), EDGE_IGNORE_INDEX)
    upper[0, 0, 1] = 2
    upper[0, 0, 2] = 5
    upper[0, 1, 2] = 6

    full = expand_upper_triangular_edges(upper)

    assert full[0, 1, 0].item() == 2
    assert full[0, 2, 0].item() == 6
    assert full[0, 2, 1].item() == 5
    assert torch.equal(torch.diagonal(full[0]), torch.full((3,), EDGE_IGNORE_INDEX))
    assert check_edge_matrix_consistency(full)


def test_check_edge_matrix_consistency_detects_invalid_final_matrices():
    upper = torch.full((1, 3, 3), EDGE_IGNORE_INDEX)
    upper[0, 0, 1] = 3
    upper[0, 0, 2] = 5
    upper[0, 1, 2] = 1
    full = expand_upper_triangular_edges(upper)

    assert check_edge_matrix_consistency(full)

    asymmetric = full.clone()
    asymmetric[0, 1, 0] = 4
    assert not check_edge_matrix_consistency(asymmetric)

    masked = full.clone()
    masked[0, 0, 1] = EDGE_MASK_ID
    assert not check_edge_matrix_consistency(masked)

    self_bond = full.clone()
    self_bond[0, 0, 0] = 1
    assert not check_edge_matrix_consistency(self_bond)

    with_padding = torch.full((1, 3, 3), EDGE_IGNORE_INDEX)
    with_padding[0, 0, 1] = 2
    with_padding[0, 1, 0] = 2
    atom_mask = torch.tensor([[True, True, False]])
    assert check_edge_matrix_consistency(with_padding, atom_mask=atom_mask)


def test_corrupt_edges_mask_ratio_zero_preserves_targets():
    targets = _full_targets()
    noisy_edges, corrupted_mask, valid_mask = corrupt_edges(targets, 0.0)

    assert torch.equal(noisy_edges, targets)
    assert not corrupted_mask.any()
    assert valid_mask.sum().item() == 3


def test_corrupt_edges_mask_ratio_one_masks_all_valid_independent_pairs():
    targets = _full_targets()
    noisy_edges, corrupted_mask, valid_mask = corrupt_edges(targets, 1.0)

    assert torch.equal(corrupted_mask, valid_mask)
    assert noisy_edges[0, 0, 1].item() == EDGE_MASK_ID
    assert noisy_edges[0, 1, 0].item() == EDGE_MASK_ID
    assert noisy_edges[0, 0, 2].item() == EDGE_MASK_ID
    assert noisy_edges[0, 2, 0].item() == EDGE_MASK_ID
    assert torch.equal(torch.diagonal(noisy_edges[0]), torch.full((3,), EDGE_IGNORE_INDEX))


def test_corrupt_edges_reproducible_with_generator_and_batch_size_greater_than_one():
    targets = torch.cat([_full_targets(), _full_targets()], dim=0)
    generator_a = torch.Generator().manual_seed(123)
    generator_b = torch.Generator().manual_seed(123)

    noisy_a, corrupted_a, valid_a = corrupt_edges(targets, 0.5, generator=generator_a)
    noisy_b, corrupted_b, valid_b = corrupt_edges(targets, 0.5, generator=generator_b)

    assert torch.equal(noisy_a, noisy_b)
    assert torch.equal(corrupted_a, corrupted_b)
    assert torch.equal(valid_a, valid_b)
    assert corrupted_a.shape == targets.shape


def test_corrupt_edges_preserves_padding_diagonal_and_handles_no_valid_pairs():
    targets = torch.full((1, 2, 2), EDGE_IGNORE_INDEX)

    noisy_edges, corrupted_mask, valid_mask = corrupt_edges(targets, 1.0)

    assert torch.equal(noisy_edges, targets)
    assert not corrupted_mask.any()
    assert not valid_mask.any()


def test_corrupt_edges_rejects_non_target_classes():
    targets = _full_targets()
    targets[0, 0, 1] = EDGE_MASK_ID

    try:
        corrupt_edges(targets, 1.0)
    except ValueError as error:
        assert "public classes" in str(error)
    else:
        raise AssertionError("corrupt_edges accepted an internal mask class as a target")


def test_corrupt_edges_preserves_stereo_consistency_after_expansion():
    upper = torch.full((1, 3, 3), EDGE_IGNORE_INDEX)
    upper[0, 0, 1] = 5
    upper[0, 0, 2] = 6
    upper[0, 1, 2] = 1
    targets = expand_upper_triangular_edges(upper)

    noisy_edges, corrupted_mask, _ = corrupt_edges(targets, 0.0)

    assert not corrupted_mask.any()
    assert check_edge_matrix_consistency(noisy_edges)


def _full_targets():
    upper = torch.full((1, 3, 3), EDGE_IGNORE_INDEX)
    upper[0, 0, 1] = 1
    upper[0, 0, 2] = 5
    upper[0, 1, 2] = 6
    return expand_upper_triangular_edges(upper)
