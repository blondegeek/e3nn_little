# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring, bare-except
import math
from typing import List, Tuple

import torch

from e3nn_little import o3
from e3nn_little.eval_code import eval_code


def WeightedTensorProduct(Rs_in1, Rs_in2, Rs_out, normalization='component', own_weight=True):
    Rs_in1 = o3.convention(Rs_in1)
    Rs_in2 = o3.convention(Rs_in2)
    Rs_out = o3.convention(Rs_out)

    instr = [
        (i_1, i_2, i_out, 'uvw')
        for i_1, (_, l_1, p_1) in enumerate(Rs_in1)
        for i_2, (_, l_2, p_2) in enumerate(Rs_in2)
        for i_out, (_, l_out, p_out) in enumerate(Rs_out)
        if abs(l_1 - l_2) <= l_out <= l_1 + l_2 and p_1 * p_2 == p_out
    ]
    return CustomWeightedTensorProduct(Rs_in1, Rs_in2, Rs_out, instr, normalization, own_weight)


def GroupedWeightedTensorProduct(Rs_in1, Rs_in2, Rs_out, groups=math.inf, normalization='component', own_weight=True):
    Rs_in1 = o3.convention(Rs_in1)
    Rs_in2 = o3.convention(Rs_in2)
    Rs_out = o3.convention(Rs_out)

    groups = min(groups, min(mul for mul, _, _ in Rs_in1), min(mul for mul, _, _ in Rs_out))

    Rs_in1 = [(mul // groups + (g < mul % groups), l, p) for mul, l, p in Rs_in1 for g in range(groups)]
    Rs_out = [(mul // groups + (g < mul % groups), l, p) for mul, l, p in Rs_out for g in range(groups)]

    instr = [
        (i_1, i_2, i_out, 'uvw')
        for i_1, (_, l_1, p_1) in enumerate(Rs_in1)
        for i_2, (_, l_2, p_2) in enumerate(Rs_in2)
        for i_out, (_, l_out, p_out) in enumerate(Rs_out)
        if abs(l_1 - l_2) <= l_out <= l_1 + l_2 and p_1 * p_2 == p_out
        if i_1 % groups == i_out % groups
    ]
    return CustomWeightedTensorProduct(Rs_in1, Rs_in2, Rs_out, instr, normalization, own_weight)


def ElementwiseTensorProduct(Rs_in1, Rs_in2, normalization='component'):
    Rs_in1 = o3.simplify(Rs_in1)
    Rs_in2 = o3.simplify(Rs_in2)

    assert sum(mul for mul, _, _ in Rs_in1) == sum(mul for mul, _, _ in Rs_in2)

    i = 0
    while i < len(Rs_in1):
        mul_1, l_1, p_1 = Rs_in1[i]
        mul_2, l_2, p_2 = Rs_in2[i]

        if mul_1 < mul_2:
            Rs_in2[i] = (mul_1, l_2, p_2)
            Rs_in2.insert(i + 1, (mul_2 - mul_1, l_2, p_2))

        if mul_2 < mul_1:
            Rs_in1[i] = (mul_2, l_1, p_1)
            Rs_in1.insert(i + 1, (mul_1 - mul_2, l_1, p_1))
        i += 1

    Rs_out = []
    instr = []
    for i, ((mul, l_1, p_1), (mul_2, l_2, p_2)) in enumerate(zip(Rs_in1, Rs_in2)):
        assert mul == mul_2
        for l in list(range(abs(l_1 - l_2), l_1 + l_2 + 1)):
            i_out = len(Rs_out)
            Rs_out.append((mul, l, p_1 * p_2))
            instr += [
                (i, i, i_out, 'uuu')
            ]

    return CustomWeightedTensorProduct(Rs_in1, Rs_in2, Rs_out, instr, normalization, own_weight=False)


class CustomWeightedTensorProduct(torch.nn.Module):
    def __init__(
            self,
            Rs_in1,
            Rs_in2,
            Rs_out,
            instr: List[Tuple[int, int, int, str]],
            normalization: str = 'component',
            own_weight: bool = True
        ):
        """
        Create a Tensor Product operation that has each of his path weighted by a parameter.
        `instr` is a list of instructions.
        An instruction if of the form (i_1, i_2, i_out, mode)
        it means "Put `Rs_in1[i_1] otimes Rs_in2[i_2] into Rs_out[i_out]"
        `mode` determines the way the multiplicities are treated.
        The default mode should be 'uvw', meaning that all paths are created.
        """

        super().__init__()

        self.Rs_in1 = o3.convention(Rs_in1)
        self.Rs_in2 = o3.convention(Rs_in2)
        self.Rs_out = o3.convention(Rs_out)

        code = ""

        index_w = 0
        wigners = set()
        count = [0 for _ in range(o3.dim(self.Rs_out))]

        instr = sorted(instr)  # for optimization

        last_s1, last_s2, last_ss = None, None, None
        for i_1, i_2, i_out, mode in instr:
            mul_1, l_1, p_1 = self.Rs_in1[i_1]
            mul_2, l_2, p_2 = self.Rs_in2[i_2]
            mul_out, l_out, p_out = self.Rs_out[i_out]
            dim_1 = mul_1 * (2 * l_1 + 1)
            dim_2 = mul_2 * (2 * l_2 + 1)
            dim_out = mul_out * (2 * l_out + 1)
            index_1 = o3.dim(self.Rs_in1[:i_1])
            index_2 = o3.dim(self.Rs_in2[:i_2])
            index_out = o3.dim(self.Rs_out[:i_out])

            assert p_1 * p_2 == p_out
            assert abs(l_1 - l_2) <= l_out <= l_1 + l_2

            if dim_1 == 0 or dim_2 == 0 or dim_out == 0:
                continue

            if last_s1 != i_1:
                code += f"    s1 = x1[:, {index_1}:{index_1+dim_1}].reshape(batch, {mul_1}, {2 * l_1 + 1})\n"
                last_s1 = i_1

            if last_s2 != i_2:
                code += f"    s2 = x2[:, {index_2}:{index_2+dim_2}].reshape(batch, {mul_2}, {2 * l_2 + 1})\n"
                last_s2 = i_2

            assert mode in ['uvw', 'uvu', 'uvv', 'uuw', 'uuu', 'uvuv']

            if last_ss != (i_1, i_2, mode[:2]):
                if mode[:2] == 'uv':
                    code += f"    ss = ein('zui,zvj->zuvij', s1, s2)\n"
                if mode[:2] == 'uu':
                    code += f"    ss = ein('zui,zuj->zuij', s1, s2)\n"
                last_ss = (i_1, i_2, mode[:2])

            wigners.add((l_1, l_2, l_out))

            if mode == 'uvw':
                dim_w = mul_1 * mul_2 * mul_out
                code += f"    sw = w[:, {index_w}:{index_w+dim_w}].reshape(batch, {mul_1}, {mul_2}, {mul_out})\n"
                code += f"    out[:, {index_out}:{index_out+dim_out}] += ein('zuvw,ijk,zuvij->zwk', sw, C{l_1}_{l_2}_{l_out}, ss).reshape(batch, {dim_out})\n"

                for pos in range(index_out, index_out + dim_out):
                    count[pos] += mul_1 * mul_2

            if mode == 'uvu':
                assert mul_1 == mul_out
                dim_w = mul_1 * mul_2
                code += f"    sw = w[:, {index_w}:{index_w+dim_w}].reshape(batch, {mul_1}, {mul_2})\n"
                code += f"    out[:, {index_out}:{index_out+dim_out}] += ein('zuv,ijk,zuvij->zuk', sw, C{l_1}_{l_2}_{l_out}, ss).reshape(batch, {dim_out})\n"

                for pos in range(index_out, index_out + dim_out):
                    count[pos] += mul_2

            if mode == 'uvv':
                assert mul_2 == mul_out
                dim_w = mul_1 * mul_2
                code += f"    sw = w[:, {index_w}:{index_w+dim_w}].reshape(batch, {mul_1}, {mul_2})\n"
                code += f"    out[:, {index_out}:{index_out+dim_out}] += ein('zuv,ijk,zuvij->zvk', sw, C{l_1}_{l_2}_{l_out}, ss).reshape(batch, {dim_out})\n"

                for pos in range(index_out, index_out + dim_out):
                    count[pos] += mul_1

            if mode == 'uuw':
                assert mul_1 == mul_2
                dim_w = mul_1 * mul_out
                code += f"    sw = w[:, {index_w}:{index_w+dim_w}].reshape(batch, {mul_1}, {mul_out})\n"
                code += f"    out[:, {index_out}:{index_out+dim_out}] += ein('zuw,ijk,zuij->zwk', sw, C{l_1}_{l_2}_{l_out}, ss).reshape(batch, {dim_out})\n"

                for pos in range(index_out, index_out + dim_out):
                    count[pos] += mul_1

            if mode == 'uuu':
                assert mul_1 == mul_2 == mul_out
                dim_w = 0
                code += f"    out[:, {index_out}:{index_out+dim_out}] += ein('ijk,zuij->zuk', C{l_1}_{l_2}_{l_out}, ss).reshape(batch, {dim_out})\n"

                for pos in range(index_out, index_out + dim_out):
                    count[pos] += 1

            if mode == 'uvuv':
                assert mul_1 * mul_2 == mul_out
                dim_w = 0
                code += f"    out[:, {index_out}:{index_out+dim_out}] += ein('ijk,zuvij->zuvk', C{l_1}_{l_2}_{l_out}, ss).reshape(batch, {dim_out})\n"

                for pos in range(index_out, index_out + dim_out):
                    count[pos] += 1

            index_w += dim_w
            code += "\n"

        ilast = 0
        clast = count[0]
        for i, c in enumerate(count):
            if clast != c:
                if clast > 1:
                    code += f"    out[:, {ilast}:{i}].div_({clast ** 0.5})\n"
                clast = c
                ilast = i
        if clast > 1:
            code += f"    out[:, {ilast}:].div_({clast ** 0.5})\n"

        wigners = sorted(wigners)
        self.wigners_names = [f"C{l_1}_{l_2}_{l_3}" for l_1, l_2, l_3 in wigners]
        args = ", ".join(f"{arg}: torch.Tensor" for arg in self.wigners_names)

        for arg, (l_1, l_2, l_out) in zip(self.wigners_names, wigners):
            wig = o3.wigner_3j(l_1, l_2, l_out)

            if normalization == 'component':
                wig *= (2 * l_out + 1) ** 0.5
            if normalization == 'norm':
                wig *= (2 * l_1 + 1) ** 0.5 * (2 * l_2 + 1) ** 0.5

            self.register_buffer(arg, wig)

        x = """
import torch

@torch.jit.script
def main(ARGS, x1: torch.Tensor, x2: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    batch = x1.shape[0]
    out = x1.new_zeros((batch, DIM))
    ein = torch.einsum

CODE
    return out
"""
        x = x.replace("DIM", f"{o3.dim(self.Rs_out)}")
        x = x.replace("ARGS", args)
        x = x.replace("CODE", code)

        self.code = x
        self.main = eval_code(x).main
        self.nweight = index_w
        if own_weight:
            self.weight = torch.nn.Parameter(torch.randn(self.nweight))

    def __repr__(self):
        return "{name} ({Rs_in1} x {Rs_in2} -> {Rs_out} using {nw} paths)".format(
            name=self.__class__.__name__,
            Rs_in1=o3.format_Rs(self.Rs_in1),
            Rs_in2=o3.format_Rs(self.Rs_in2),
            Rs_out=o3.format_Rs(self.Rs_out),
            nw=self.nweight,
        )

    def forward(self, features_1, features_2, weight=None):
        """
        :return:         tensor [..., channel]
        """
        *size, n = features_1.size()
        features_1 = features_1.reshape(-1, n)
        assert n == o3.dim(self.Rs_in1), f"{n} is not {o3.dim(self.Rs_in1)}"
        *size2, n = features_2.size()
        features_2 = features_2.reshape(-1, n)
        assert n == o3.dim(self.Rs_in2), f"{n} is not {o3.dim(self.Rs_in2)}"
        assert size == size2

        if weight is None:
            weight = self.weight
        weight = weight.reshape(-1, self.nweight)
        if weight.shape[0] == 1:
            weight = weight.repeat(features_1.shape[0], 1)

        wigners = [getattr(self, arg) for arg in self.wigners_names]

        if features_1.shape[0] == 0:
            return torch.zeros(*size, o3.dim(self.Rs_out))

        features = self.main(*wigners, features_1, features_2, weight)
        return features.reshape(*size, -1)
