# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring, bare-except
import math
from functools import lru_cache
from typing import List, Tuple

import torch
from sympy import Integer, Poly, diff, factorial, pi, sqrt, symbols

from e3nn_little.util import eval_code
from e3nn_little import o3



def spherical_harmonics(Rs, pos, normalization='none'):
    """
    spherical harmonics

    :param Rs: list of L's
    :param pos: tensor of shape [..., 3]
    :return: tensor of shape [..., m]
    """
    with torch.autograd.profiler.record_function(f'spherical_harmonics({o3.format_Rs(Rs)}, {pos.shape[:-1]})'):
        Rs = o3.simplify(Rs)
        *size, _ = pos.shape
        pos = pos.reshape(-1, 3)
        d = torch.norm(pos, 2, dim=1)
        pos = pos[d > 0]
        pos = pos / d[d > 0, None]

        # if z > x, rotate x-axis with z-axis
        s = pos[:, 2].abs() > pos[:, 0].abs()
        pos[s] = pos[s] @ pos.new_tensor([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])

        alpha = torch.atan2(pos[:, 1], pos[:, 0])
        z = pos[:, 2]
        y = pos[:, :2].norm(dim=1)

        sh = _spherical_harmonics_alpha_z_y(Rs, alpha, z, y)

        # rotate back
        sh[s] = sh[s] @ _rep_zx(tuple(Rs), pos.dtype, pos.device)

        if len(d) > len(sh):
            out = sh.new_zeros(len(d), sh.shape[1])
            out[d == 0] = math.sqrt(1 / (4 * math.pi)) * torch.cat([sh.new_ones(1) if l == 0 else sh.new_zeros(2 * l + 1) for mul, l, p in Rs for _ in range(mul)])
            out[d > 0] = sh
            sh = out

        if normalization == 'component':
            sh.mul_(math.sqrt(4 * math.pi))
        if normalization == 'norm':
            sh.mul_(torch.cat([math.sqrt(4 * math.pi / (2 * l + 1)) * sh.new_ones(2 * l + 1) for mul, l, p in Rs for _ in range(mul)]))
        return sh.reshape(*size, sh.shape[1])


@lru_cache()
def _rep_zx(Rs, dtype, device):
    return o3.rep(Rs, 0, -math.pi / 2, 0).to(device=device, dtype=dtype)


def _spherical_harmonics_alpha_z_y(Rs, alpha, z, y):
    """
    spherical harmonics
    """
    Rs = o3.simplify(Rs)
    sha = _spherical_harmonics_alpha(o3.lmax(Rs), alpha.flatten())  # [z, m]
    shz = _spherical_harmonics_z(Rs, z.flatten(), y.flatten())  # [z, l * m]
    out = mul_m_lm(Rs, sha, shz)
    return out.reshape(alpha.shape + (shz.shape[1],))


@torch.jit.script
def _spherical_harmonics_alpha(l: int, alpha: torch.Tensor) -> torch.Tensor:  # pragma: no cover
    """
    the alpha (x, y) component of the spherical harmonics
    (useful to perform fourier transform)

    :param alpha: tensor of shape [...]
    :return: tensor of shape [..., m]
    """
    alpha = alpha.unsqueeze(-1)  # [..., 1]

    m = torch.arange(1, l + 1, dtype=alpha.dtype, device=alpha.device)  # [1, 2, 3, ..., l]
    cos = torch.cos(m * alpha)  # [..., m]

    m = torch.arange(l, 0, -1, dtype=alpha.dtype, device=alpha.device)  # [l, l-1, l-2, ..., 1]
    sin = torch.sin(m * alpha)  # [..., m]

    out = torch.cat([
        math.sqrt(2) * sin,
        torch.ones_like(alpha),
        math.sqrt(2) * cos,
    ], dim=alpha.ndim-1)

    return out  # [..., m]


def _spherical_harmonics_z(Rs, z, y=None):
    """
    the z component of the spherical harmonics
    (useful to perform fourier transform)

    :param z: tensor of shape [...]
    :return: tensor of shape [..., l * m]
    """
    Rs = o3.simplify(Rs)
    for _, l, p in Rs:
        assert p in [0, (-1)**l]
    ls = [l for mul, l, _ in Rs]
    return legendre(ls, z, y)  # [..., l * m]


def legendre(ls, z, y=None):
    """
    associated Legendre polynomials

    :param ls: list
    :param z: tensor of shape [...]
    :return: tensor of shape [..., l * m]
    """
    if y is None:
        y = (1 - z**2).relu().sqrt()

    return _legendre_genjit(tuple(ls))(z, y)


_legendre_code = """
import torch

@torch.jit.script
def main(z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = z.new_zeros(z.shape + (lsize,))

# fill out

    return out
"""


@lru_cache()
def _legendre_genjit(ls):
    ls = list(ls)
    fill = ""
    i = 0
    for l in ls:
        for m in range(l + 1):
            p = _poly_legendre(l, m)
            formula = " + ".join("{:.25f} * z**{} * y**{}".format(c, zn, yn) for (zn, yn), c in p.items())
            fill += "    l{} = {}\n".format(m, formula)

        for m in range(-l, l + 1):
            fill += "    out[..., {}] = l{}\n".format(i, abs(m))
            i += 1

    code = _legendre_code
    code = code.replace("lsize", str(sum(2 * l + 1 for l in ls)))
    code = code.replace("# fill out", fill)
    return eval_code(code).main


def _poly_legendre(l, m):
    """
    polynomial coefficients of legendre

    y = sqrt(1 - z^2)
    """
    z, y = symbols('z y', real=True)
    return Poly(_sympy_legendre(l, m), domain='R', gens=(z, y)).as_dict()


def _sympy_legendre(l, m):
    """
    en.wikipedia.org/wiki/Associated_Legendre_polynomials
    - remove two times (-1)^m
    - use another normalization such that P(l, -m) = P(l, m)

    y = sqrt(1 - z^2)
    """
    l = Integer(l)
    m = Integer(abs(m))
    z, y = symbols('z y', real=True)
    ex = 1 / (2**l * factorial(l)) * y**m * diff((z**2 - 1)**l, z, l + m)
    ex *= (-1)**l * sqrt((2 * l + 1) / (4 * pi) * factorial(l - m) / factorial(l + m))
    return ex


@torch.jit.script
def mul_m_lm(Rs: List[Tuple[int, int, int]], x_m: torch.Tensor, x_lm: torch.Tensor) -> torch.Tensor:  # pragma: no cover
    """
    multiply tensor [..., l * m] by [..., m]
    """
    l_max = x_m.shape[-1] // 2
    out = []
    i = 0
    for mul, l, _ in Rs:
        d = mul * (2 * l + 1)
        x1 = x_lm[..., i: i + d]  # [..., mul * m]
        x1 = x1.reshape(x1.shape[:-1] + (mul, 2 * l + 1))  # [..., mul, m]
        x2 = x_m[..., l_max - l: l_max + l + 1]  # [..., m]
        x2 = x2.reshape(x2.shape[:-1] + (1, 2 * l + 1))  # [..., mul=1, m]
        x = x1 * x2
        x = x.reshape(x.shape[:-2] + (d,))
        out.append(x)
        i += d
    return torch.cat(out, dim=-1)
