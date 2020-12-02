# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring, bare-except
import torch
from torch_sparse import SparseTensor


def convention(Rs):
    """
    :param Rs: list of triplet (multiplicity, representation order, [parity])
    :return: conventional version of the same list which always includes parity
    """
    if isinstance(Rs, int):
        return [(1, Rs, 0)]

    out = []
    for r in Rs:
        if isinstance(r, int):
            mul, l, p = 1, r, 0
        elif len(r) == 2:
            (mul, l), p = r, 0
        elif len(r) == 3:
            mul, l, p = r

        assert isinstance(mul, int) and mul >= 0
        assert isinstance(l, int) and l >= 0
        assert p in [0, 1, -1]

        out.append((mul, l, p))
    return out


def simplify(Rs):
    """
    :param Rs: list of triplet (multiplicity, representation order, [parity])
    :return: An equivalent list with parity = {-1, 0, 1} and neighboring orders consolidated into higher multiplicity.

    Note that simplify does not sort the Rs.
    >>> simplify([(1, 1), (1, 1), (1, 0)])
    [(2, 1, 0), (1, 0, 0)]

    Same order Rs which are seperated from each other are not combined
    >>> simplify([(1, 1), (1, 1), (1, 0), (1, 1)])
    [(2, 1, 0), (1, 0, 0), (1, 1, 0)]

    Parity is normalized to {-1, 0, 1}
    >>> simplify([(1, 1, -1), (1, 1, 50), (1, 0, 0)])
    [(1, 1, -1), (1, 1, 1), (1, 0, 0)]
    """
    out = []
    Rs = convention(Rs)
    for mul, l, p in Rs:
        if out and out[-1][1:] == (l, p):
            out[-1] = (out[-1][0] + mul, l, p)
        elif mul > 0:
            out.append((mul, l, p))
    return out


def dim(Rs):
    """
    :param Rs: list of triplet (multiplicity, representation order, [parity])
    :return: dimention of the representation
    """
    Rs = convention(Rs)
    return sum(mul * (2 * l + 1) for mul, l, _ in Rs)


def mul_dim(Rs):
    """
    :param Rs: list of triplet (multiplicity, representation order, [parity])
    :return: number of multiplicities of the representation
    """
    Rs = convention(Rs)
    return sum(mul for mul, _, _ in Rs)


def lmax(Rs):
    """
    :param Rs: list of triplet (multiplicity, representation order, [parity])
    :return: maximum l present in the signal
    """
    return max(l for mul, l, _ in convention(Rs) if mul > 0)


def format_Rs(Rs):
    """
    :param Rs: list of triplet (multiplicity, representation order, [parity])
    :return: simplified version of the same list with the parity
    """
    Rs = convention(Rs)
    d = {
        0: "",
        1: "e",
        -1: "o",
    }
    return ",".join("{}{}{}".format("{}x".format(mul) if mul > 1 else "", l, d[p]) for mul, l, p in Rs if mul > 0)


def sort(Rs):
    """
    :return: (Rs_out, permutation_matrix)
    stable sorting of the representation by (l, p)

    sorted = perm @ unsorted

    >>> sort([(1, 1), (1, 0)])
    ([(1, 0, 0), (1, 1, 0)],
    tensor([[0., 0., 0., 1.],
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.]]))

    Example usage:
    sortedRs, permutation_matrix = sort(Rs)
    permuted_input = einsum('ij,j->i', permutation_matrix, input)
    """
    Rs_in = simplify(Rs)
    xs = []

    j = 0  # input offset
    for mul, l, p in Rs_in:
        d = mul * (2 * l + 1)
        xs.append((l, p, mul, j, d))
        j += d

    index = []

    Rs_out = []
    i = 0  # output offset
    for l, p, mul, j, d in sorted(xs):
        Rs_out.append((mul, l, p))
        for _ in range(d):
            index.append([i, j])
            i += 1
            j += 1

    index = torch.tensor(index).T
    permutation_matrix = SparseTensor(row=index[0], col=index[1], value=torch.ones(index.shape[1]))

    return Rs_out, permutation_matrix
