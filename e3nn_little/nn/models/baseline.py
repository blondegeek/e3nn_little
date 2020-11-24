# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring, bare-except, abstract-method, arguments-differ
from functools import partial

import ase
import torch
from torch.autograd import profiler
from torch.nn import Embedding
from torch_geometric.nn import MessagePassing, radius_graph
from torch_scatter import scatter

from e3nn_little import o3
from e3nn_little.nn import (GatedBlockParity, GaussianRadialModel,
                            WeightedTensorProduct, Linear, swish)


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
    def __init__(self, mul=30, lmax=1,
                 num_layers=1, cutoff=10.0, rad_gaussians=40,
                 rad_h=500, rad_layers=4, num_neighbors=20,
                 readout='add', dipole=False, mean=None, std=None, scale=None,
                 atomref=None, options=""):
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
        self.options = options

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
        self.Rs_sh = [(1, l, (-1)**l) for l in range(lmax + 1)]  # spherical harmonics representation

        modules = []

        Rs = [(mul, 0, 1)]
        for _ in range(num_layers):
            act = GatedBlockParity.make_gated_block(Rs, mul, lmax)
            lay = Conv(Rs, act.Rs_in, self.Rs_sh, RadialModel)
            extra = Linear(act.Rs_out, act.Rs_out) if 'extra' in self.options else None
            shortcut = Linear(Rs, act.Rs_out) if 'res' in self.options else None

            Rs = o3.simplify(act.Rs_out)

            modules += [torch.nn.ModuleList([lay, act, extra, shortcut])]

        self.layers = torch.nn.ModuleList(modules)

        Rs_out = [(1, 0, 1)]
        self.layers.append(Conv(Rs, Rs_out, self.Rs_sh, RadialModel))

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
        sh = o3.spherical_harmonics(self.Rs_sh, edge_vec, 'component') / self.num_neighbors**0.5

        for lay, act, extra, shortcut in self.layers[:-1]:
            if shortcut:
                s = shortcut(h)

            h = lay(h, edge_index, edge_vec, sh)  # convolution
            h = act(h)  # gate non linearity

            if extra:
                h = extra(h)  # optional extra linear layer

            if shortcut:
                m = shortcut.output_mask
                h = 0.5**0.5 * s + (1 * (1-m) + 0.5**0.5 * m) * h

        h = self.layers[-1](h, edge_index, edge_vec, sh)

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
    def __init__(self, Rs_in, Rs_out, Rs_sh, RadialModel, normalization='component'):
        super().__init__(aggr='add')
        self.Rs_in = o3.simplify(Rs_in)
        self.Rs_out = o3.simplify(Rs_out)

        self.si = Linear(Rs_in, Rs_out)
        self.tp = WeightedTensorProduct(Rs_in, Rs_sh, Rs_out, normalization=normalization, own_weight=False)
        self.rm = RadialModel(self.tp.nweight)

        self.Rs_sh = Rs_sh
        self.normalization = normalization

    def forward(self, x, edge_index, edge_vec, sh, size=None):
        with profiler.record_function("Conv"):
            # x = [num_atoms, dim(Rs_in)]
            s = self.si(x)

            w = self.rm(edge_vec.norm(dim=1))  # [num_messages, nweight]
            x = self.propagate(edge_index, size=size, x=x, sh=sh, w=w)

            m = self.si.output_mask
            return 0.5**0.5 * s + (1 + (0.5**0.5 - 1) * m) * x

    def message(self, x_j, sh, w):
        return self.tp(x_j, sh, w)
