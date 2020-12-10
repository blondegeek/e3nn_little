# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring
import torch
from e3nn_little.nn.models import Network


def test_dtypes():
    torch.set_default_dtype(torch.float32)
    z = torch.tensor([0, 1, 2, 3])
    pos = torch.randn(len(z), 3)
    Network()(z, pos)

    torch.set_default_dtype(torch.float64)
    z = torch.tensor([0, 1, 2, 3])
    pos = torch.randn(len(z), 3)
    Network()


def test_normalized():
    torch.set_default_dtype(torch.float32)
    i = torch.arange(1000)
    z = i % 100
    pos = torch.randn(len(z), 3)
    batch = i % 50
    model = Network()
    out = model(z, pos, batch)
    assert out.shape == (50, 1)
