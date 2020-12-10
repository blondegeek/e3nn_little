# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring
import math
from functools import partial

import torch
from e3nn_little import o3


def test_sh_equivariance1():
    """test
    - compose
    - spherical_harmonics_alpha_beta
    - irrep
    """
    torch.set_default_dtype(torch.float64)
    for l in range(7 + 1):
        a, b, _ = o3.rand_angles()
        alpha, beta, gamma = o3.rand_angles()

        ra, rb, _ = o3.compose(alpha, beta, gamma, a, b, 0)
        Yrx = o3.spherical_harmonics_alpha_beta([l], ra, rb)

        Y = o3.spherical_harmonics_alpha_beta([l], a, b)
        DrY = o3.irrep(l, alpha, beta, gamma) @ Y

        assert (Yrx - DrY).abs().max() < 1e-10 * Y.abs().max()


def test_sh_equivariance2():
    """test
    - rot
    - rep
    - spherical_harmonics
    """
    torch.set_default_dtype(torch.float64)
    Rs = [0, 1, 2, 3, 4, 5, 6]

    abc = o3.rand_angles()
    R = o3.rot(*abc)
    D = o3.rep(Rs, *abc)

    x = torch.randn(10, 3)

    y1 = o3.spherical_harmonics(Rs, x @ R.T)
    y2 = o3.spherical_harmonics(Rs, x) @ D.T

    assert (y1 - y2).abs().max() < 1e-10


def test_sh_is_in_irrep():
    torch.set_default_dtype(torch.float64)
    for l in range(4 + 1):
        a, b, _ = o3.rand_angles()
        Y = o3.spherical_harmonics_alpha_beta([l], a, b) * math.sqrt(4 * math.pi) / math.sqrt(2 * l + 1) * (-1) ** l
        D = o3.irrep(l, a, b, 0)
        assert (Y - D[:, l]).abs().max() < 1e-10


def test_rsh_backwardable():
    torch.set_default_dtype(torch.float64)
    lmax = 10
    Rs = [(1, l) for l in range(lmax + 1)]

    xyz = torch.tensor([
        [0., 0., 1.],
        [1.0, 0, 0],
        [0.0, 10.0, 0],
        [0.435, 0.7644, 0.023],
    ], requires_grad=True, dtype=torch.float64)
    assert torch.autograd.gradcheck(partial(o3.spherical_harmonics, Rs), (xyz,), check_undefined_grad=False)


def test_sh_normalization():
    torch.set_default_dtype(torch.float64)
    for l in range(11 + 1):
        n = o3.spherical_harmonics([l], torch.randn(3), 'integral').pow(2).mean()
        assert abs(n - 1 / (4 * math.pi)) < 1e-10

        n = o3.spherical_harmonics([l], torch.randn(3), 'norm').norm()
        assert abs(n - 1) < 1e-10

        n = o3.spherical_harmonics([l], torch.randn(3), 'component').pow(2).mean()
        assert abs(n - 1) < 1e-10


def test_sh_closure():
    """
    integral of Ylm * Yjn = delta_lj delta_mn
    integral of 1 over the unit sphere = 4 pi
    """
    torch.set_default_dtype(torch.float64)
    x = torch.randn(300_000, 3)
    Ys = [o3.spherical_harmonics([l], x) for l in range(0, 3 + 1)]
    for l1, Y1 in enumerate(Ys):
        for l2, Y2 in enumerate(Ys):
            m = Y1[:, :, None] * Y2[:, None, :]
            m = m.mean(0) * 4 * math.pi
            if l1 == l2:
                i = torch.eye(2 * l1 + 1)
                assert (m - i).abs().max() < 0.01
            else:
                assert m.abs().max() < 0.01


def test_sh_parity():
    """
    (-1)^l Y(x) = Y(-x)
    """
    torch.set_default_dtype(torch.float64)
    for l in range(11 + 1):
        x = torch.randn(3)
        Y1 = (-1) ** l * o3.spherical_harmonics([l], x)
        Y2 = o3.spherical_harmonics([l], -x)
        assert (Y1 - Y2).abs().max() < 1e-10
