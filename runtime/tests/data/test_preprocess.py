"""CPU tests for normal-map rotation augmentation."""

from __future__ import annotations

import pytest
import torch


def test_axis_angle_produces_a_proper_rotation() -> None:
    from vforge.data.preprocess import rotation_matrix_axis_angle

    rotation = rotation_matrix_axis_angle([0.0, 1.0, 0.0], 30.0)

    assert torch.allclose(rotation @ rotation.T, torch.eye(3), atol=1e-6)
    assert float(torch.det(rotation)) == pytest.approx(1.0, abs=1e-6)


def test_a_zero_axis_is_rejected_rather_than_normalized_to_nothing() -> None:
    from vforge.data.preprocess import rotation_matrix_axis_angle

    with pytest.raises(ValueError, match="non-zero"):
        rotation_matrix_axis_angle([0.0, 0.0, 0.0], 30.0)


def test_random_rotations_are_proper_and_never_reflections() -> None:
    """A reflection would flip handedness, which is not a legal normal transform."""
    from vforge.data.preprocess import random_rotation_matrix

    torch.manual_seed(0)
    for _ in range(20):
        rotation = random_rotation_matrix()
        assert torch.allclose(rotation @ rotation.T, torch.eye(3), atol=1e-5)
        assert float(torch.det(rotation)) == pytest.approx(1.0, abs=1e-5)


def test_random_axes_are_unit_length() -> None:
    from vforge.data.preprocess import random_axis

    torch.manual_seed(0)
    axes = torch.stack([random_axis() for _ in range(16)])

    assert torch.allclose(torch.linalg.norm(axes, dim=1), torch.ones(16), atol=1e-5)


def test_rotating_a_normal_clip_preserves_unit_length() -> None:
    from vforge.data.preprocess import (
        random_rotation_matrix,
        rotate_normal_frames,
    )

    torch.manual_seed(0)
    direction = torch.tensor([0.0, 0.0, 1.0]).reshape(1, 3, 1, 1)
    clip = ((direction + 1.0) * 0.5).expand(4, 3, 2, 2).contiguous()

    rotated = rotate_normal_frames(clip, random_rotation_matrix())
    decoded = rotated * 2.0 - 1.0

    assert rotated.shape == clip.shape
    assert torch.allclose(torch.linalg.norm(decoded, dim=1), torch.ones(4, 2, 2), atol=1e-3)


def test_neutral_grey_is_a_fixed_point_of_any_rotation() -> None:
    """Occluded regions are grey; if rotation moved them they would turn colored."""
    from vforge.data.preprocess import (
        random_rotation_matrix,
        rotate_normal_frames,
    )

    torch.manual_seed(1)
    grey = torch.full((3, 3, 2, 2), 0.5)

    rotated = rotate_normal_frames(grey, random_rotation_matrix())

    assert torch.allclose(rotated, grey, atol=1e-5)


def test_rotation_is_constant_across_time_within_one_clip() -> None:
    """A per-frame draw would make the normal condition flicker."""
    from vforge.data.preprocess import (
        random_rotation_matrix,
        rotate_normal_frames,
    )

    torch.manual_seed(2)
    frame = torch.rand(1, 3, 2, 2)
    clip = frame.expand(5, 3, 2, 2).contiguous()

    rotated = rotate_normal_frames(clip, random_rotation_matrix())

    assert torch.allclose(rotated, rotated[0:1].expand_as(rotated), atol=1e-6)


def test_rotate_normal_frames_rejects_a_non_normal_shape() -> None:
    from vforge.data.preprocess import rotate_normal_frames

    with pytest.raises(ValueError, match=r"\[T, 3, H, W\]"):
        rotate_normal_frames(torch.rand(4, 1, 2, 2), torch.eye(3))


def test_zero_probability_leaves_every_clip_alone() -> None:
    from vforge.data.preprocess import NormalRotationAugmentor

    torch.manual_seed(3)
    clip = torch.rand(2, 3, 2, 2)
    augmentor = NormalRotationAugmentor(prob=0.0)

    assert torch.equal(augmentor(clip), clip)


def test_a_non_positive_angle_cap_falls_back_to_full_so3() -> None:
    """Capping at zero degrees would silently disable the augmentation."""
    from vforge.data.preprocess import NormalRotationAugmentor

    augmentor = NormalRotationAugmentor(max_angle_deg=0.0, full_so3=False)

    assert augmentor.full_so3
    assert "full-SO3" in augmentor.describe()


def test_a_capped_augmentor_stays_within_its_angle() -> None:
    from vforge.data.preprocess import NormalRotationAugmentor

    torch.manual_seed(4)
    augmentor = NormalRotationAugmentor(max_angle_deg=10.0, full_so3=False)

    for _ in range(20):
        rotation = augmentor.sample_matrix()
        # trace = 1 + 2*cos(theta) for any rotation matrix.
        angle = torch.acos(((torch.trace(rotation) - 1.0) / 2.0).clamp(-1.0, 1.0))
        assert float(torch.rad2deg(angle)) <= 10.0 + 1e-3


def test_sp_peers_drawing_from_the_global_rng_stay_in_lockstep() -> None:
    """A private Generator here would desynchronize ranks that share a dp_rank."""
    from vforge.data.preprocess import NormalRotationAugmentor

    augmentor = NormalRotationAugmentor()
    torch.manual_seed(7)
    first = augmentor.sample_matrix()
    torch.manual_seed(7)
    second = augmentor.sample_matrix()

    assert torch.equal(first, second)
