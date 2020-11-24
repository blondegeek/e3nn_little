# pylint: disable=arguments-differ, no-member, missing-docstring, invalid-name, line-too-long
from functools import partial

import torch
from torch.autograd import profiler

from e3nn_little.util import normalize2mom


class ConstantRadialModel(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(d))

    def forward(self, _radii):
        batch = _radii.size(0)
        return self.weight.reshape(1, -1).expand(batch, -1)


class FiniteElementModel(torch.nn.Module):
    def __init__(self, out_dim, position, basis, Model):
        '''
        :param out_dim: output dimension
        :param position: tensor [i, ...]
        :param basis: scalar function: tensor [a, ...] -> [a]
        :param Model: Class(d1, d2), trainable model: R^d1 -> R^d2
        '''
        super().__init__()
        self.register_buffer('position', position)
        self.basis = basis
        self.f = Model(len(position), out_dim)

    def forward(self, x):
        """
        :param x: tensor [batch, ...]
        :return: tensor [batch, dim]
        """
        diff = x.unsqueeze(1) - self.position.unsqueeze(0)  # [batch, i, ...]
        batch, n, *rest = diff.size()
        x = self.basis(diff.reshape(-1, *rest)).reshape(batch, n)  # [batch, i]
        return self.f(x)


class FC(torch.nn.Module):
    def __init__(self, d1, d2, h, L, act):
        super().__init__()

        self.h = h
        self.L = L

        weights = []

        hh = d1
        for _ in range(L):
            weights.append(torch.nn.Parameter(torch.randn(h, hh)))
            hh = h

        weights.append(torch.nn.Parameter(torch.randn(d2, hh)))
        self.weights = torch.nn.ParameterList(weights)
        self.act = normalize2mom(act)

    def __repr__(self):
        return f"{self.__class__.__name__}(L={self.L} h={self.h})"

    def forward(self, x):
        with profiler.record_function(repr(self)):
            if self.L == 0:
                W = self.weights[0]
                h = x.size(1)
                return x @ W.t()

            for i, W in enumerate(self.weights):
                h = x.size(1)

                if i == 0:
                    # note: normalization assumes that the sum of the inputs is 1
                    x = self.act(x @ W.t())
                elif i < self.L:
                    x = self.act(x @ (W.t() / h ** 0.5))
                else:
                    # we aim for a gaussian output at initialisation
                    x = x @ (W.t() / h ** 0.5)

            return x


def FiniteElementFCModel(out_dim, position, basis, h, L, act):
    Model = partial(FC, h=h, L=L, act=act)
    return FiniteElementModel(out_dim, position, basis, Model)


def GaussianRadialModel(out_dim, max_radius, number_of_basis, h, L, act, min_radius=0.):
    """exp(-x^2 /spacing)"""
    spacing = (max_radius - min_radius) / (number_of_basis - 1)
    radii = torch.linspace(min_radius, max_radius, number_of_basis)
    sigma = 0.8 * spacing

    def basis(x):
        return x.div(sigma).pow(2).neg().exp().div(1.423085244900308)
    return FiniteElementFCModel(out_dim, radii, basis, h, L, act)
