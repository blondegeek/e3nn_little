# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring, bare-except
import torch
from e3nn_little.group import Group
from e3nn_little.math import complete_basis, direct_sum, kron


def intertwiners(group: Group, D1, D2, eps=1e-9):
    """
    Compute a basis of the vector space of matrices A such that
    D1(g) A = A D2(g) for all g in O(3)
    """
    e = group.identity()
    I1 = D1(e)
    I2 = D2(e)

    # picking 20 random rotations seems good enough
    rr = [group.random() for i in range(20)]
    xs = [kron(D1(g), I2) - kron(I1, D2(g).T) for g in rr]
    xtx = sum(x.T @ x for x in xs)

    res = xtx.symeig(eigenvectors=True)
    null_space = res.eigenvectors.T[res.eigenvalues.abs() < eps]
    null_space = null_space.reshape(null_space.shape[0], I1.shape[0], I2.shape[0])

    # check that it works
    solutions = []
    for A in null_space:
        d = 0
        for _ in range(4):
            g = group.random()
            d += A @ D2(g) - D1(g) @ A
        d /= 4
        if d.abs().max() < eps:
            solutions.append((d.norm(), A))
    solutions = [A for _, A in sorted(solutions, key=lambda x: x[0])]

    return torch.stack(solutions) if len(solutions) > 0 else torch.zeros(0, I1.shape[0], I2.shape[0])


def reduce(group: Group, D, D_small, eps=1e-9):
    """
    Given a "big" representation and a "small" representation
    computes how many times the small appears in the big one and return:
    - how many times the "small" appears in the "big"
    - a matrix that block diagonalize the "big" rep.
    - the remaining of the "big" representation
    """
    def change_and_remove(A, oldD, d):
        def newD(g):
            return (A @ oldD(g) @ A.T)[d:][:, d:]
        return newD

    e = group.identity()
    dim = D(e).shape[0]
    dim_small = D_small(e).shape[0]

    D_rest = D
    bigA = torch.eye(dim)
    n = 0

    while True:
        A = intertwiners(group, D_small, D_rest, eps) * dim_small**0.5

        # stops if "small" does not appear in "big" anymore
        if A.shape[0] == 0:
            break
        A = A[0]

        expand = complete_basis(A, eps)
        A = torch.cat([A, expand])

        bigA = direct_sum(torch.eye(n * dim_small), A) @ bigA
        n += 1
        D_rest = change_and_remove(bigA, D, n * dim_small)

    g = group.random()
    assert (bigA @ D(g) @ bigA.T - direct_sum(*[D_small(g)] * n + [D_rest(g)])).abs().max() < eps
    return n, bigA, D_rest
