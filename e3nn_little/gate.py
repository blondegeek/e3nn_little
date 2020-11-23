# pylint: disable=invalid-name, arguments-differ, missing-docstring, line-too-long, no-member, unbalanced-tuple-unpacking, abstract-method
import copy

import torch

from e3nn_little import o3
from e3nn_little.tensor_product import ElementwiseTensorProduct


class Activation(torch.nn.Module):
    def __init__(self, Rs, acts):
        '''
        Can be used only with scalar fields

        :param acts: list of tuple (multiplicity, activation)
        '''
        super().__init__()

        Rs = o3.simplify(Rs)
        acts = copy.deepcopy(acts)

        n1 = sum(mul for mul, _, _ in Rs)
        n2 = sum(mul for mul, _ in acts if mul > 0)

        for i, (mul, act) in enumerate(acts):
            if mul == -1:
                acts[i] = (n1 - n2, act)
                assert n1 - n2 >= 0

        assert n1 == sum(mul for mul, _ in acts)

        i = 0
        while i < len(Rs):
            mul_r, l, p_r = Rs[i]
            mul_a, act = acts[i]

            if mul_r < mul_a:
                acts[i] = (mul_r, act)
                acts.insert(i + 1, (mul_a - mul_r, act))

            if mul_a < mul_r:
                Rs[i] = (mul_a, l, p_r)
                Rs.insert(i + 1, (mul_r - mul_a, l, p_r))
            i += 1

        x = torch.linspace(0, 10, 256)

        Rs_out = []
        for (mul, l, p_in), (mul_a, act) in zip(Rs, acts):
            assert mul == mul_a

            a1, a2 = act(x), act(-x)
            if (a1 - a2).abs().max() < a1.abs().max() * 1e-10:
                p_act = 1
            elif (a1 + a2).abs().max() < a1.abs().max() * 1e-10:
                p_act = -1
            else:
                p_act = 0

            p = p_act if p_in == -1 else p_in
            Rs_out.append((mul, 0, p))

            if p_in != 0 and p == 0:
                raise ValueError("warning! the parity is violated")

        self.Rs_out = Rs_out
        self.acts = acts

    def forward(self, features, dim=-1):
        '''
        :param features: [..., channels, ...]
        '''
        output = []
        index = 0
        for mul, act in self.acts:
            output.append(act(features.narrow(dim, index, mul)))
            index += mul

        if output:
            return torch.cat(output, dim=dim)
        else:
            size = list(features.size())
            size[dim] = 0
            return features.new_zeros(*size)


class GatedBlockParity(torch.nn.Module):
    def __init__(self, Rs_scalars, act_scalars, Rs_gates, act_gates, Rs_nonscalars):
        super().__init__()

        self.Rs_in = Rs_scalars + Rs_gates + Rs_nonscalars
        self.Rs_scalars, self.Rs_gates, self.Rs_nonscalars = Rs_scalars, Rs_gates, Rs_nonscalars

        self.act_scalars = Activation(Rs_scalars, act_scalars)
        Rs_scalars = self.act_scalars.Rs_out

        self.act_gates = Activation(Rs_gates, act_gates)
        Rs_gates = self.act_gates.Rs_out

        self.mul = ElementwiseTensorProduct(Rs_nonscalars, Rs_gates)
        Rs_nonscalars = self.mul.Rs_out

        self.Rs_out = Rs_scalars + Rs_nonscalars

    def __repr__(self):
        return "{name} ({Rs_scalars} + {Rs_gates} + {Rs_nonscalars} -> {Rs_out})".format(
            name=self.__class__.__name__,
            Rs_scalars=o3.format_Rs(self.Rs_scalars),
            Rs_gates=o3.format_Rs(self.Rs_gates),
            Rs_nonscalars=o3.format_Rs(self.Rs_nonscalars),
            Rs_out=o3.format_Rs(self.Rs_out),
        )

    def forward(self, features, dim=-1):
        scalars, gates, nonscalars = o3.cut(features, self.Rs_scalars, self.Rs_gates, self.Rs_nonscalars, dim_=dim)
        scalars = self.act_scalars(scalars)
        if gates.shape[dim]:
            gates = self.act_gates(gates)
            nonscalars = self.mul(nonscalars, gates)
            features = torch.cat([scalars, nonscalars], dim=dim)
        else:
            features = scalars
        return features
