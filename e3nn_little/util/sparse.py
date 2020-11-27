# pylint: disable=invalid-name, arguments-differ, missing-docstring, line-too-long, no-member, redefined-builtin, not-callable
import math

import torch
from torch_sparse import SparseTensor


def register_sparse_buffer(module, name, sp):
    row, col, val = sp.coo()
    module.register_buffer("{}_row".format(name), row)
    module.register_buffer("{}_col".format(name), col)
    module.register_buffer("{}_val".format(name), val)
    module.register_buffer("{}_size".format(name), torch.tensor(sp.sparse_sizes()))


def get_sparse_buffer(module, name):
    row = getattr(module, "{}_row".format(name))
    col = getattr(module, "{}_col".format(name))
    val = getattr(module, "{}_val".format(name))
    siz = getattr(module, "{}_size".format(name))
    return SparseTensor(
        row=row,
        col=col,
        value=val,
        sparse_sizes=siz.tolist(),
    )


def sparse_multiply(sparse: SparseTensor, tensor: torch.Tensor, dim: int) -> torch.Tensor:
    """
    einsum('ij,abc...j...xyz->abc...i...xyz', sparse, tensor)
    """
    dim = dim % tensor.ndim
    n = tensor.shape[dim]
    s1 = tensor.shape[:dim]
    s2 = tensor.shape[dim+1:]

    # [abcjxyz]
    tensor = tensor.reshape(math.prod(s1), n, math.prod(s2))  # [(abc)j(xyz)]
    tensor = tensor.transpose(0, 1)  # [j(abc)(xyz)]
    tensor = tensor.reshape(n, math.prod(s1) * math.prod(s2))  # [j(abcxyz)]

    tensor = sparse @ tensor  # [i(abcxyz)]

    n = tensor.shape[0]
    tensor = tensor.reshape(n, math.prod(s1), math.prod(s2))  # [i(abc)(xyz)]
    tensor = tensor.transpose(1, 0)  # [(abc)i(xyz)]
    tensor = tensor.reshape(s1 + (n,) + s2)  # [abcixyz]

    return tensor
