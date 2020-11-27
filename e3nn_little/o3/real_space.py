# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring, bare-except
import math

import torch


def rot_z(gamma, dtype=None, device=None):
    """
    Rotation around Z axis
    """
    if dtype is None:
        dtype = torch.get_default_dtype()
    if not torch.is_tensor(gamma):
        gamma = torch.tensor(gamma, dtype=dtype, device=device)
    else:
        gamma = gamma.to(dtype=dtype, device=device)

    return torch.stack([
        torch.stack([gamma.cos(),
                     -gamma.sin(),
                     gamma.new_zeros(gamma.shape)], dim=-1),
        torch.stack([gamma.sin(),
                     gamma.cos(),
                     gamma.new_zeros(gamma.shape)], dim=-1),
        torch.stack([gamma.new_zeros(gamma.shape),
                     gamma.new_zeros(gamma.shape),
                     gamma.new_ones(gamma.shape)], dim=-1)
    ], dim=-2)


def rot_y(beta, dtype=None, device=None):
    """
    Rotation around Y axis
    """
    if dtype is None:
        dtype = torch.get_default_dtype()
    if not torch.is_tensor(beta):
        beta = torch.tensor(beta, dtype=dtype, device=device)
    else:
        beta = beta.to(dtype=dtype, device=device)

    return torch.stack([
        torch.stack([beta.cos(),
                     beta.new_zeros(beta.shape),
                     beta.sin()], dim=-1),
        torch.stack([beta.new_zeros(beta.shape),
                     beta.new_ones(beta.shape),
                     beta.new_zeros(beta.shape)], dim=-1),
        torch.stack([-beta.sin(),
                     beta.new_zeros(beta.shape),
                     beta.cos()], dim=-1),
    ], dim=-2)


# The following two functions (rot and xyz_to_angles) satisfies that
# rot(*xyz_to_angles([x, y, z]), 0) @ np.array([[0], [0], [1]])
# is proportional to
# [x, y, z]

def rot(alpha, beta, gamma, dtype=None, device=None):
    """
    ZYZ Euler angles rotation
    """
    return rot_z(alpha, dtype, device) @ rot_y(beta, dtype, device) @ rot_z(gamma, dtype, device)


def rand_rot():
    """
    random rotation matrix
    """
    return rot(*rand_angles())


def rand_angles():
    """
    random rotation angles
    """
    alpha, gamma = 2 * math.pi * torch.rand(2)
    beta = torch.rand(()).mul(2).sub(1).acos()
    return alpha, beta, gamma


def angles_to_xyz(alpha, beta):
    """
    Convert (alpha, beta) into point (x, y, z) on the sphere
    """
    x = torch.sin(beta) * torch.cos(alpha)
    y = torch.sin(beta) * torch.sin(alpha)
    z = torch.cos(beta)
    return torch.stack([x, y, z], dim=-1)


def xyz_to_angles(x, y=None, z=None):
    """
    Convert point (x, y, z) on the sphere into (alpha, beta)
    """
    if y is not None and z is not None:
        x = torch.stack([x, y, z], dim=-1)

    x = torch.nn.functional.normalize(x, p=2, dim=-1)  # forward 0's instead of nan for zero-radius
    x.masked_fill_(x < -1., -1.)                       # mitigate numerical inaccuracies from normalization
    x.masked_fill_(x > 1., 1.)

    beta = torch.acos(x[..., 2])
    alpha = torch.atan2(x[..., 1], x[..., 0])
    return alpha, beta


def rot_to_abc(R):
    """
    Convert rotation matrix into (alpha, beta, gamma)
    """
    x = R @ R.new_tensor([0, 0, 1])
    a, b = xyz_to_angles(x)
    R = rot(a, b, a.new_zeros(a.shape)).transpose(-1, -2) @ R
    c = torch.atan2(R[..., 1, 0], R[..., 0, 0])
    return a, b, c


def compose(a1, b1, c1, a2, b2, c2):
    """
    (a, b, c) = (a1, b1, c1) composed with (a2, b2, c2)
    """
    comp = rot(a1, b1, c1) @ rot(a2, b2, c2)
    xyz = comp @ torch.tensor([0, 0, 1.])
    a, b = xyz_to_angles(xyz)
    rotz = rot(0, -b, -a) @ comp
    c = torch.atan2(rotz[1, 0], rotz[0, 0])
    return a, b, c
