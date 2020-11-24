# pylint: disable=arguments-differ, no-member, missing-docstring, invalid-name, line-too-long
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
    def __init__(self, hs, act):
        super().__init__()
        assert isinstance(hs, tuple)
        self.hs = hs
        weights = []

        for h1, h2 in zip(self.hs, self.hs[1:]):
            weights.append(torch.nn.Parameter(torch.randn(h1, h2)))

        self.weights = torch.nn.ParameterList(weights)
        self.act = normalize2mom(act)

    def __repr__(self):
        return f"{self.__class__.__name__}{self.hs}"

    def forward(self, x):
        with profiler.record_function(repr(self)):
            for i, W in enumerate(self.weights):
                # first layer
                if i == 0:
                    x = x @ W
                else:
                    x = x @ (W / x.shape[1]**0.5)

                # not last layer
                if i < len(self.weights) - 1:
                    x = self.act(x)

            return x


def FiniteElementFCModel(out_dim, position, basis, hs, act):
    def Model(d_in, d_out):
        return FC((d_in,) + hs + (d_out,), act)
    return FiniteElementModel(out_dim, position, basis, Model)


def GaussianRadialModel(out_dim, max_radius, number_of_basis, hs, act, min_radius=0.):
    """exp(-x^2 /spacing)"""
    spacing = (max_radius - min_radius) / (number_of_basis - 1)
    radii = torch.linspace(min_radius, max_radius, number_of_basis)
    sigma = 0.8 * spacing

    def basis(x):
        return x.div(sigma).pow(2).neg().exp().div(1.423085244900308)
    return FiniteElementFCModel(out_dim, radii, basis, hs, act)
