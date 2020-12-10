# pylint: disable=arguments-differ, no-member, missing-docstring, invalid-name, line-too-long
import torch

from e3nn_little.math import normalize2mom


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
