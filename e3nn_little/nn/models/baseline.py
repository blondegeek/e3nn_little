# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring, bare-except, abstract-method, arguments-differ
from functools import partial

import torch
from torch.nn import Embedding
from torch_geometric.nn import MessagePassing, radius_graph
from torch_scatter import scatter

from e3nn_little import o3
from e3nn_little.nn import (GatedBlockParity, GaussianRadialModel,
                            GroupedWeightedTensorProduct, Linear)
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
    def __init__(self, muls=(50, 10, 0), ps=(1, -1), lmax=1,
                 num_layers=2, cutoff=10.0, rad_gaussians=40,
                 rad_hs=(300, 300, 300, 300, 50), num_neighbors=20, groups=2,
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

        self.embedding = Embedding(100, muls[0])
        self.Rs_in = [(muls[0], 0, 1)]

        RadialModel = partial(
            GaussianRadialModel,
            max_radius=cutoff,
            number_of_basis=rad_gaussians,
            hs=rad_hs,
            act=None if 'relu' in options else swish
        )
        self.Rs_sh = [(1, l, (-1)**l) for l in range(lmax + 1)]  # spherical harmonics representation

        Rs = self.Rs_in
        modules = []
        for _ in range(num_layers):
            act = make_gated_block(Rs, muls, ps, self.Rs_sh)
            conv = Conv(Rs, act.Rs_in, self.Rs_sh, RadialModel, groups)
            shortcut = Linear(Rs, act.Rs_out) if 'res' in self.options else None

            Rs = o3.simplify(act.Rs_out)

            modules += [torch.nn.ModuleList([conv, act, shortcut])]

        self.layers = torch.nn.ModuleList(modules)

        self.Rs_out = [(1, 0, p) for p in ps]
        self.layers.append(Conv(Rs, self.Rs_out, self.Rs_sh, RadialModel))

        self.register_buffer('initial_atomref', atomref)
        self.atomref = None
        if atomref is not None:
            self.atomref = Embedding(100, 1)
            self.atomref.weight.data.copy_(atomref)

    def forward(self, z, pos, batch=None):
        assert z.dim() == 1 and z.dtype == torch.long
        batch = torch.zeros_like(z) if batch is None else batch

        h = self.embedding(z)

        edge_index = radius_graph(pos, r=self.cutoff, batch=batch, max_num_neighbors=1000)
        row, col = edge_index
        edge_vec = pos[row] - pos[col]
        sh = o3.spherical_harmonics(self.Rs_sh, edge_vec, 'component') / self.num_neighbors**0.5

        for conv, act, shortcut in self.layers[:-1]:
            with torch.autograd.profiler.record_function("Layer"):
                if shortcut:
                    s = shortcut(h)

                h = conv(h, edge_index, edge_vec, sh)  # convolution
                h = act(h)  # gate non linearity

                if shortcut:
                    m = shortcut.output_mask
                    h = 0.5**0.5 * s + (1 + (0.5**0.5 - 1) * m) * h

        with torch.autograd.profiler.record_function("Layer"):
            h = self.layers[-1](h, edge_index, edge_vec, sh)

        s = 0
        for i, (mul, l, p) in enumerate(self.Rs_out):
            assert mul == 1 and l == 0
            if p == 1:
                s += h[:, i]
            if p == -1:
                s += h[:, i].pow(2).mul(0.5)  # odd^2 = even
        h = s.view(-1, 1)

        if self.mean is not None and self.std is not None:
            h = h * self.std + self.mean

        if self.atomref is not None:
            h = h + self.atomref(z)

        out = scatter(h, batch, dim=0, reduce=self.readout)

        if self.scale is not None:
            out = self.scale * out

        return out


def make_gated_block(Rs_in, muls, ps, Rs_sh):
    """
    Make a `GatedBlockParity` assuming many things
    """
    Rs_available = [
        (l, p_in * p_sh)
        for _, l_in, p_in in o3.simplify(Rs_in)
        for _, l_sh, p_sh in Rs_sh
        for l in range(abs(l_in - l_sh), l_in + l_sh + 1)
    ]

    scalars = [(muls[0], 0, p) for p in ps if (0, p) in Rs_available]
    act_scalars = [(mul, swish if p == 1 else torch.abs) for mul, l, p in scalars]

    nonscalars = [(muls[l], l, p*(-1)**l) for l in range(1, len(muls)) for p in ps if (l, p*(-1)**l) in Rs_available]
    if (0, +1) in Rs_available:
        gates = [(o3.mul_dim(nonscalars), 0, +1)]
        act_gates = [(-1, torch.sigmoid)]
    else:
        gates = [(o3.mul_dim(nonscalars), 0, -1)]
        act_gates = [(-1, torch.tanh)]

    return GatedBlockParity(scalars, act_scalars, gates, act_gates, nonscalars)


class Conv(MessagePassing):
    def __init__(self, Rs_in, Rs_out, Rs_sh, RadialModel, groups=1, normalization='component'):
        super().__init__(aggr='add')
        self.Rs_in = o3.simplify(Rs_in)
        self.Rs_out = o3.simplify(Rs_out)
        self.Rs_sh = o3.simplify(Rs_sh)

        self.si = Linear(self.Rs_in, self.Rs_out)
        self.tp = GroupedWeightedTensorProduct(self.Rs_in, self.Rs_sh, self.Rs_out, groups, normalization=normalization, own_weight=False, weight_batch=True)
        self.rm = RadialModel(self.tp.nweight)

        self.normalization = normalization

    def forward(self, x, edge_index, edge_vec, sh, size=None):
        with torch.autograd.profiler.record_function("Conv"):
            # x = [num_atoms, dim(Rs_in)]
            s = self.si(x)

            w = self.rm(edge_vec.norm(dim=1))  # [num_messages, nweight]
            x = self.propagate(edge_index, size=size, x=x, sh=sh, w=w)

            m = self.si.output_mask
            return 0.5**0.5 * s + (1 + (0.5**0.5 - 1) * m) * x

    def message(self, x_j, sh, w):
        return self.tp(x_j, sh, w)
