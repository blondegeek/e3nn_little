# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring, bare-except, abstract-method, arguments-differ
from math import pi

import torch
from torch.nn import Embedding
from torch_geometric.nn import MessagePassing, radius_graph
from torch_scatter import scatter

from e3nn_little import o3
from e3nn_little.nn import (FC, CustomWeightedTensorProduct, GatedBlockParity,
                            GaussianRadialModel, Identity, Linear)
from e3nn_little.math import swish


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
    def __init__(self, muls=(128, 8, 0), ps=(1, -1), lmax=1,
                 num_layers=6, cutoff=10.0, rad_gaussians=50,
                 rad_hs=(128,), num_neighbors=20,
                 readout='add', dipole=False, mean=None, std=None, scale=None,
                 atomref=None, options="res"):
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

        self.radial = GaussianRadialModel(rad_gaussians, cutoff)
        self.Rs_sh = [(1, l, (-1)**l) for l in range(lmax + 1)]  # spherical harmonics representation

        Rs = self.Rs_in
        modules = []
        for _ in range(num_layers):
            act = make_gated_block(Rs, muls, ps, self.Rs_sh)
            conv = Conv(Rs, act.Rs_in, self.Rs_sh, (rad_gaussians,) + rad_hs)
            if 'res' in self.options:
                if Rs == act.Rs_out:
                    shortcut = Identity(Rs, act.Rs_out)
                else:
                    shortcut = Linear(Rs, act.Rs_out)
            else:
                shortcut = None

            Rs = o3.simplify(act.Rs_out)

            modules += [torch.nn.ModuleList([conv, act, shortcut])]

        self.layers = torch.nn.ModuleList(modules)

        self.Rs_out = [(1, 0, p) for p in ps]
        self.layers.append(Conv(Rs, self.Rs_out, self.Rs_sh, (rad_gaussians,) + rad_hs))

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
        edge_sh = o3.spherical_harmonics(self.Rs_sh, edge_vec, 'component') / self.num_neighbors**0.5
        edge_len = edge_vec.norm(dim=1)
        edge_weight = self.radial(edge_len)
        edge_c = (pi * edge_len / self.cutoff).cos().add(1).div(2)

        for conv, act, shortcut in self.layers[:-1]:
            with torch.autograd.profiler.record_function("Layer"):
                if shortcut:
                    s = shortcut(h)

                h = conv(h, edge_index, edge_weight, edge_c, edge_sh)  # convolution
                h = act(h)  # gate non linearity

                if shortcut:
                    m = shortcut.output_mask
                    h = 0.5**0.5 * s + (1 + (0.5**0.5 - 1) * m) * h

        with torch.autograd.profiler.record_function("Layer"):
            h = self.layers[-1](h, edge_index, edge_weight, edge_c, edge_sh)

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
    act_scalars = [(mul, swish if p == 1 else torch.tanh) for mul, l, p in scalars]

    nonscalars = [(muls[l], l, p*(-1)**l) for l in range(1, len(muls)) for p in ps if (l, p*(-1)**l) in Rs_available]
    if (0, +1) in Rs_available:
        gates = [(o3.mul_dim(nonscalars), 0, +1)]
        act_gates = [(-1, torch.sigmoid)]
    else:
        gates = [(o3.mul_dim(nonscalars), 0, -1)]
        act_gates = [(-1, torch.tanh)]

    return GatedBlockParity(scalars, act_scalars, gates, act_gates, nonscalars)


class Conv(MessagePassing):
    def __init__(self, Rs_in, Rs_out, Rs_sh, rad_hs, normalization='component'):
        super().__init__(aggr='add')
        self.Rs_in = o3.simplify(Rs_in)
        self.Rs_out = o3.simplify(Rs_out)
        self.Rs_sh = o3.simplify(Rs_sh)

        self.si = Linear(self.Rs_in, self.Rs_out)
        self.lin1 = Linear(self.Rs_in, self.Rs_in)

        instr = []
        Rs = []
        for i_1, (mul_1, l_1, p_1) in enumerate(self.Rs_in):
            for i_2, (_, l_2, p_2) in enumerate(self.Rs_sh):
                for l_out in range(abs(l_1 - l_2), l_1 + l_2 + 1):
                    p_out = p_1 * p_2
                    if (l_out, p_out) in [(l, p) for _, l, p in self.Rs_out]:
                        r = (mul_1, l_out, p_out)
                        if r in Rs:
                            i_out = Rs.index(r)
                        else:
                            i_out = len(Rs)
                            Rs.append(r)
                        instr += [(i_1, i_2, i_out, 'uvu', True)]
        self.tp = CustomWeightedTensorProduct(self.Rs_in, self.Rs_sh, Rs, instr, own_weight=False, weight_batch=True)
        self.nn = FC(rad_hs + (self.tp.nweight,), swish)
        self.lin2 = Linear(Rs, self.Rs_out)

        self.normalization = normalization

    def forward(self, x, edge_index, edge_weight, edge_c, edge_sh, size=None):
        with torch.autograd.profiler.record_function("Conv"):
            # x = [num_atoms, dim(Rs_in)]
            s = self.si(x)

            w = self.nn(edge_weight)  # [num_messages, nweight]
            w = w * edge_c[:, None]

            x = self.lin1(x)
            x = self.propagate(edge_index, size=size, x=x, sh=edge_sh, w=w)
            x = self.lin2(x)

            m = self.si.output_mask
            return 0.5**0.5 * s + (1 + (0.5**0.5 - 1) * m) * x

    def message(self, x_j, sh, w):
        return self.tp(x_j, sh, w)
