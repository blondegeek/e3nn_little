# pylint: disable=arguments-differ, no-member, missing-docstring, invalid-name, line-too-long
import torch

from e3nn_little.util import normalize2mom


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
        with torch.autograd.profiler.record_function(repr(self)):
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


class FCrelu(torch.nn.Module):
    def __init__(self, hs):
        super().__init__()
        assert isinstance(hs, tuple)
        self.hs = hs
        weights = []

        for h1, h2 in zip(self.hs, self.hs[1:]):
            weights.append(torch.nn.Parameter(torch.randn(h1, h2)))

        self.weights = torch.nn.ParameterList(weights)

    def __repr__(self):
        return f"{self.__class__.__name__}{self.hs}"

    def forward(self, x):
        with torch.autograd.profiler.record_function(repr(self)):
            for i, W in enumerate(self.weights):
                # first layer
                if i == 0:
                    x = x @ W
                else:
                    x = x @ W.mul(2**0.5 / x.shape[1]**0.5)

                # not last layer
                if i < len(self.weights) - 1:
                    x.relu_()

            return x


class GaussianRadialModel(torch.nn.Module):
    def __init__(self, number_of_basis, max_radius, min_radius=0.):
        '''
        :param out_dim: output dimension
        :param position: tensor [i, ...]
        :param basis: scalar function: tensor [a, ...] -> [a]
        :param Model: Class(d1, d2), trainable model: R^d1 -> R^d2
        '''
        super().__init__()
        spacing = (max_radius - min_radius) / (number_of_basis - 1)
        radii = torch.linspace(min_radius, max_radius, number_of_basis)
        self.sigma = 0.8 * spacing

        self.register_buffer('radii', radii)

    def forward(self, x):
        """
        :param x: tensor [batch]
        :return: tensor [batch, dim]
        """
        x = x[:, None] - self.radii[None, :]  # [batch, i]
        x = x.div(self.sigma).pow(2).neg().exp().div(1.423085244900308)
        return x  # [batch, i]
