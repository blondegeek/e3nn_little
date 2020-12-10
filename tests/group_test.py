# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring
import itertools

import torch
from e3nn_little.group import O3, SO3, is_representation


def test_representation():
    torch.set_default_dtype(torch.float64)
    for group in [SO3(), O3()]:
        for r in itertools.islice(group.irrep_indices(), 10):
            print(r)
            assert is_representation(group, group.irrep(r), 1e-9)
