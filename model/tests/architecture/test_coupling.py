from functools import partial

import pytest
import torch

from model.architecture.coupling import AffineCoupling
from model.architecture.initialization import init_hinet_
from model.architecture.subnet import DenseSubnet
from model.tests.helpers import randomize_


def make_block(split: int = 12, clamp: float = 2.0) -> AffineCoupling:
    return AffineCoupling(split, split, partial(DenseSubnet, growth_channels=8), clamp=clamp)


def test_subnet_shapes():
    subnet = DenseSubnet(12, 5)
    assert subnet(torch.randn(2, 12, 6, 6)).shape == (2, 5, 6, 6)


def test_subnet_parameter_count_matches_official_layout():
    # in=12, out=12, growth=32: conv1..conv5 = 3488 + 12704 + 21920 + 31136 + 15132
    subnet = DenseSubnet(12, 12)
    assert sum(p.numel() for p in subnet.parameters()) == 84_380


def test_scale_range():
    block = make_block(clamp=2.0)
    s = torch.tensor([-1e6, 0.0, 1e6])
    scale = block.scale(s)
    torch.testing.assert_close(scale[1], torch.tensor(1.0))
    # exponent = clamp * 2 * (sigmoid(s) - 0.5) lies in (-clamp, clamp)
    torch.testing.assert_close(scale[0], torch.exp(torch.tensor(-2.0)))
    torch.testing.assert_close(scale[2], torch.exp(torch.tensor(2.0)))


def test_forward_reverse_round_trip():
    block = randomize_(make_block()).double()
    x = torch.randn(2, 24, 8, 8, dtype=torch.float64)
    with torch.no_grad():
        y = block(x)
        assert not torch.allclose(y, x)
        torch.testing.assert_close(block(y, reverse=True), x)
        torch.testing.assert_close(block(block(x, reverse=True)), x)


def test_identity_at_official_init():
    block = init_hinet_(make_block())
    x = torch.randn(2, 24, 8, 8)
    with torch.no_grad():
        torch.testing.assert_close(block(x), x, rtol=0, atol=0)
        torch.testing.assert_close(block(x, reverse=True), x, rtol=0, atol=0)


def test_rejects_wrong_channel_count():
    with pytest.raises(ValueError):
        make_block()(torch.randn(1, 20, 4, 4))
