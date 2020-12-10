# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring, bare-except
import os

import torch
from e3nn_little import o3
from e3nn_little.math import direct_sum

_Jd, _W3j = torch.load(os.path.join(os.path.dirname(__file__), 'constants.pt'))


def _z_rot_mat(angle, l):
    """
    Create the matrix representation of a z-axis rotation by the given angle,
    in the irrep l of dimension 2 * l + 1, in the basis of real centered
    spherical harmonics (RC basis in rep_bases.py).

    Note: this function is easy to use, but inefficient: only the entries
    on the diagonal and anti-diagonal are non-zero, so explicitly constructing
    this matrix is unnecessary.
    """
    M = torch.zeros(2 * l + 1, 2 * l + 1, dtype=torch.float64)
    inds = torch.arange(0, 2 * l + 1, 1)
    reversed_inds = torch.arange(2 * l, -1, -1)
    frequencies = torch.arange(l, -l - 1, -1, dtype=torch.float64)
    M[inds, reversed_inds] = torch.sin(frequencies * angle)
    M[inds, inds] = torch.cos(frequencies * angle)
    return M


def irrep(l, alpha, beta, gamma):
    J = _Jd[l]
    Xa = _z_rot_mat(alpha, l)
    Xb = _z_rot_mat(beta, l)
    Xc = _z_rot_mat(gamma, l)
    return Xa @ J @ Xb @ J @ Xc


def rep(Rs, alpha, beta, gamma, parity=None):
    """
    Representation of O(3). Parity applied (-1)**parity times.
    """
    Rs = o3.simplify(Rs)
    if parity is None:
        return direct_sum(*[irrep(l, alpha, beta, gamma) for mul, l, _ in Rs for _ in range(mul)])
    else:
        assert all(parity != 0 for _, _, parity in Rs)
        return direct_sum(*[(p ** parity) * irrep(l, alpha, beta, gamma) for mul, l, p in Rs for _ in range(mul)])


def wigner_3j(l1, l2, l3):
    assert abs(l2 - l3) <= l1 <= l2 + l3

    if l1 <= l2 <= l3:
        return _W3j[(l1, l2, l3)].clone()
    if l1 <= l3 <= l2:
        return _W3j[(l1, l3, l2)].transpose(1, 2).mul((-1) ** (l1 + l2 + l3)).clone()
    if l2 <= l1 <= l3:
        return _W3j[(l2, l1, l3)].transpose(0, 1).mul((-1) ** (l1 + l2 + l3)).clone()
    if l3 <= l2 <= l1:
        return _W3j[(l3, l2, l1)].transpose(0, 2).mul((-1) ** (l1 + l2 + l3)).clone()
    if l2 <= l3 <= l1:
        return _W3j[(l2, l3, l1)].transpose(0, 2).transpose(1, 2).clone()
    if l3 <= l1 <= l2:
        return _W3j[(l3, l1, l2)].transpose(0, 2).transpose(0, 1).clone()
