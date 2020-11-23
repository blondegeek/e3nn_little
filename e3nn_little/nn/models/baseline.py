# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring, bare-except, abstract-method, arguments-differ
import math
from functools import partial

import ase
import torch
from torch.nn import Embedding
from torch_geometric.nn import MessagePassing, radius_graph
from torch_scatter import scatter

from e3nn_little import o3
from e3nn_little.nn import (GatedBlockParity, GaussianRadialModel,
                            GroupedWeightedTensorProduct, Linear, swish)


qm9_target_dict = {
    0: 'dipole_moment',
    1: 'isotropic_polarizability',
    2: 'homo',
    3: 'lumo',
    4: 'gap',
    5: 'electronic_spatial_extent',
    6: 'zpve',
    7: 'energy_U0',
    8: 'energy_U',
    9: 'enthalpy_H',
    10: 'free_energy',
    11: 'heat_capacity',
}


class Network(torch.nn.Module):
    def __init__(self, mul=10, lmax=1,
                 num_layers=2, cutoff=10.0, rad_gaussians=3,
                 rad_h=200, rad_layers=2, num_neighbors=20,
                 readout='add', dipole=False, mean=None, std=None, scale=None,
                 atomref=None):
        super(Network, self).__init__()

        assert readout in ['add', 'sum', 'mean']
        self.readout = readout
        self.cutoff = cutoff
        self.dipole = dipole
        self.readout = 'add' if self.dipole else self.readout
        self.mean = mean
        self.std = std
        self.scale = scale
        self.num_neighbors = num_neighbors

        atomic_mass = torch.from_numpy(ase.data.atomic_masses)
        self.register_buffer('atomic_mass', atomic_mass)

        self.embedding = Embedding(100, mul)

        RadialModel = partial(
            GaussianRadialModel,
            max_radius=cutoff,
            number_of_basis=rad_gaussians,
            h=rad_h,
            L=rad_layers,
            act=swish
        )
        Rs_sh = [(1, l, (-1)**l) for l in range(lmax + 1)]  # spherical harmonics representation

        modules = []

        Rs = [(mul, 0, 1)]
        for _ in range(num_layers):
            act = GatedBlockParity.make_gated_block(Rs, mul, lmax)
            lay = Conv(Rs, act.Rs_in, Rs_sh, RadialModel)

            Rs = act.Rs_out

            modules += [torch.nn.ModuleList([lay, act])]

        self.layers = torch.nn.ModuleList(modules)

        Rs_out = [(1, 0, 1)]
        self.layers.append(Conv(Rs, Rs_out, Rs_sh, RadialModel))

        self.register_buffer('initial_atomref', atomref)
        self.atomref = None
        if atomref is not None:
            self.atomref = Embedding(100, 1)
            self.atomref.weight.data.copy_(atomref)

    def forward(self, z, pos, batch=None):
        assert z.dim() == 1 and z.dtype == torch.long
        batch = torch.zeros_like(z) if batch is None else batch

        h = self.embedding(z)

        edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
        row, col = edge_index
        edge_vec = pos[row] - pos[col]

        for lay, act in self.layers[:-1]:
            h = lay(h, edge_index, edge_vec, n_norm=self.num_neighbors)
            h = act(h)

        h = self.layers[-1](h, edge_index, edge_vec, n_norm=self.num_neighbors)


        if self.dipole:
            # Get center of mass.
            mass = self.atomic_mass[z].view(-1, 1)
            c = scatter(mass * pos, batch, dim=0) / scatter(mass, batch, dim=0)
            h = h * (pos - c[batch])

        if not self.dipole and self.mean is not None and self.std is not None:
            h = h * self.std + self.mean

        if not self.dipole and self.atomref is not None:
            h = h + self.atomref(z)

        out = scatter(h, batch, dim=0, reduce=self.readout)

        if self.dipole:
            out = torch.norm(out, dim=-1, keepdim=True)

        if self.scale is not None:
            out = self.scale * out

        return out


class Conv(MessagePassing):
    def __init__(self, Rs_in, Rs_out, Rs_sh, RadialModel, groups=math.inf, normalization='component'):
        super().__init__(aggr='add')
        self.Rs_in = o3.simplify(Rs_in)
        self.Rs_out = o3.simplify(Rs_out)

        self.lin1 = Linear(Rs_in, Rs_out)
        self.tp = GroupedWeightedTensorProduct(Rs_in, Rs_sh, Rs_out, groups=groups, normalization=normalization, own_weight=False)
        self.rm = RadialModel(self.tp.nweight)
        self.lin2 = Linear(Rs_out, Rs_out)
        self.Rs_sh = Rs_sh
        self.normalization = normalization

    def forward(self, x, edge_index, edge_vec, sh=None, size=None, n_norm=1):
        # x = [num_atoms, dim(Rs_in)]
        if sh is None:
            sh = o3.spherical_harmonics(self.Rs_sh, edge_vec, self.normalization)  # [num_messages, dim(Rs_sh)]
        sh = sh / n_norm**0.5

        w = self.rm(edge_vec.norm(dim=1))  # [num_messages, nweight]

        self_interation = self.lin1(x)
        x = self.propagate(edge_index, size=size, x=x, sh=sh, w=w)
        x = self.lin2(x)
        si = self.lin1.output_mask
        return 0.5**0.5 * self_interation + (1 + (0.5**0.5 - 1) * si) * x

    def message(self, x_j, sh, w):
        return self.tp(x_j, sh, w)
