# pylint: disable=arguments-differ, no-member, missing-docstring, invalid-name, line-too-long
import torch


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
