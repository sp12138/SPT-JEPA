"""Predict observed pose effects under the product action R^3 x SO(3)."""

import math

import torch


# Scales of the commands in Windows, expressed in metres and radians.
ROTATION = slice(10, 13)
RADIANS_PER_UNIT = 0.5


def rotation_matrix(rotvec):
    """Rodrigues' formula with its finite value and gradient at zero."""
    x, y, z = rotvec.unbind(dim=-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1)
    skew = skew.unflatten(-1, (3, 3))
    turns = rotvec.norm(dim=-1) / math.pi
    identity = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device)
    return identity + torch.sinc(turns)[..., None, None] * skew + (
        0.5 * torch.sinc(turns / 2).square()[..., None, None] * (skew @ skew)
    )


def rigid_prediction_error(current, future, command, basis):
    """Squared factual prediction error in the Euclidean and SO(3) factors.

    The encoder's rotation vector is a chart for an SO(3) state. Left-compose
    the observed command because angular inputs are expressed in the base
    frame. The other coordinates retain their additive prediction. Chordal
    rotation error has the same infinitesimal units as the original angular
    residual.
    """
    dtype = torch.float64 if current.dtype == torch.float64 else torch.float32
    with torch.autocast(current.device.type, enabled=False):
        current = current.to(dtype)
        future = future.to(dtype)
        command = command.to(dtype)
        basis = basis.to(dtype)
        rotation_basis = basis[:, ROTATION]

        residual = future - current - command @ basis.T
        euclidean = residual - (residual @ rotation_basis) @ rotation_basis.T

        current_pose = rotation_matrix(RADIANS_PER_UNIT * (current @ rotation_basis))
        future_pose = rotation_matrix(RADIANS_PER_UNIT * (future @ rotation_basis))
        command_rotation = rotation_matrix(RADIANS_PER_UNIT * command[..., ROTATION])
        rotation = future_pose - command_rotation @ current_pose
        rotation_square = rotation.square().sum((-2, -1)) / (2 * RADIANS_PER_UNIT**2)

        return euclidean.square().sum(-1) + rotation_square, rotation_square
