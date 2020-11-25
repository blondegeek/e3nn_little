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
                            WeightedTensorProduct, Linear)
from e3nn_little.util import swish


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
    def __init__(self, muls=(30, 10, 5), lmax=1,
                 num_layers=1, cutoff=10.0, rad_gaussians=40,
                 rad_hs=(500, 500, 50), num_neighbors=20,
                 readout='add', dipole=False, mean=None, std=None, scale=None,
                 atomref=None, options=""):
        super().__init__()

        assert readout in ['add', 'sum', 'mean']
        self.readout = readout
        self.cutoff = cutoff
        self.dipole = dipole
        self.mean = mean
        self.std = std
        self.scale = scale
        self.num_neighbors = num_neighbors
        self.options = options

        atomic_mass = torch.from_numpy(ase.data.atomic_masses)
        self.register_buffer('atomic_mass', atomic_mass)

        self.embedding = Embedding(100, muls[0])
        Rs = [(muls[0], 0, 1)]

        RadialModel = partial(
            GaussianRadialModel,
            max_radius=cutoff,
            number_of_basis=rad_gaussians,
            hs=rad_hs,
            act=swish
        )
        self.Rs_sh = [(1, l, (-1)**l) for l in range(lmax + 1)]  # spherical harmonics representation

        modules = []
        for _ in range(num_layers):
            act = make_gated_block(Rs, muls, self.Rs_sh)
            conv = Conv(Rs, act.Rs_in, self.Rs_sh, RadialModel)
            extra = Linear(act.Rs_out, act.Rs_out) if 'extra' in self.options else None
            shortcut = Linear(Rs, act.Rs_out) if 'res' in self.options else None

            Rs = o3.simplify(act.Rs_out)

            modules += [torch.nn.ModuleList([conv, act, extra, shortcut])]

        self.layers = torch.nn.ModuleList(modules)

        Rs_out = [(1, 0, 1), (1, 0, -1)]
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

        for conv, act, extra, shortcut in self.layers[:-1]:
            if shortcut:
                s = shortcut(h)

            h = conv(h, edge_index, edge_vec, sh)  # convolution
            h = act(h)  # gate non linearity

            if extra:
                h = extra(h)  # optional extra linear layer

            if shortcut:
                m = shortcut.output_mask
                h = 0.5**0.5 * s + (1 * (1-m) + 0.5**0.5 * m) * h

        h = self.layers[-1](h, edge_index, edge_vec, sh)

        # even + odd^2 = even
        assert h.shape[1] == 2
        h = h[:, 0] + h[:, 1].pow(2)
        h = h.view(-1, 1)

        if self.mean is not None and self.std is not None:
            h = h * self.std + self.mean

        if self.atomref is not None:
            h = h + self.atomref(z)

        out = scatter(h, batch, dim=0, reduce=self.readout)

        if self.scale is not None:
            out = self.scale * out

        return out


def make_gated_block(Rs_in, muls, Rs_sh):
    """
    Make a `GatedBlockParity` assuming many things
    """
    Rs_available = [
        (l, p_in * p_sh)
        for _, l_in, p_in in o3.simplify(Rs_in)
        for _, l_sh, p_sh in Rs_sh
        for l in range(abs(l_in - l_sh), l_in + l_sh + 1)
    ]

    scalars = [(mul, l, p) for mul, l, p in [(muls[0], 0, +1), (muls[0], 0, -1)] if (l, p) in Rs_available]
    act_scalars = [(mul, swish if p == 1 else torch.tanh) for mul, l, p in scalars]

    nonscalars = [(muls[l], l, p) for l in range(1, len(muls)) for p in [+1, -1] if (l, p) in Rs_available]
    if (0, +1) in Rs_available:
        gates = [(o3.mul_dim(nonscalars), 0, +1)]
        act_gates = [(-1, torch.sigmoid)]
    else:
        gates = [(o3.mul_dim(nonscalars), 0, -1)]
        act_gates = [(-1, torch.tanh)]

    return GatedBlockParity(scalars, act_scalars, gates, act_gates, nonscalars)


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
