import pytest
import torch

from model.architecture.config import ArchitectureConfig
from model.architecture.hinet import HiNet
from model.architecture.subnet import DenseSubnet
from model.tests.helpers import randomize_


def test_default_architecture_size():
    net = HiNet()
    assert len(net.blocks) == 16
    assert net.channels == 24
    assert sum(p.numel() for p in net.parameters()) == 4_050_240


def test_identity_at_init():
    net = HiNet()
    x = torch.randn(1, 24, 8, 8)
    with torch.no_grad():
        torch.testing.assert_close(net(x), x, rtol=0, atol=0)


def test_init_statistics():
    net = HiNet(generator=torch.Generator().manual_seed(0))
    hidden = torch.cat([
        p.flatten()
        for m in net.modules() if isinstance(m, DenseSubnet)
        for layer in m.hidden_layers for p in layer.parameters()
    ])
    output = [
        p for m in net.modules() if isinstance(m, DenseSubnet)
        for p in m.output_layer.parameters()
    ]
    assert abs(hidden.std().item() - 0.01) < 0.0005
    assert abs(hidden.mean().item()) < 0.0005
    assert all(torch.count_nonzero(p) == 0 for p in output)


def test_init_is_reproducible_with_generator():
    a = HiNet(generator=torch.Generator().manual_seed(123))
    b = HiNet(generator=torch.Generator().manual_seed(123))
    for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
        assert torch.equal(pa, pb)


def test_initialize_false_keeps_pytorch_default_init():
    net = HiNet(initialize=False)
    output = net.blocks[0].f.output_layer.weight
    assert torch.count_nonzero(output) > 0


def test_forward_reverse_round_trip():
    net = randomize_(HiNet(ArchitectureConfig(num_blocks=4))).double()
    x = torch.randn(2, 24, 8, 8, dtype=torch.float64)
    with torch.no_grad():
        y = net(x)
        assert not torch.allclose(y, x)
        torch.testing.assert_close(net(y, reverse=True), x)
        torch.testing.assert_close(net(net(x, reverse=True)), x)


def test_full_depth_round_trip():
    net = randomize_(HiNet(), std=0.02).double()
    x = torch.randn(1, 24, 8, 8, dtype=torch.float64)
    with torch.no_grad():
        torch.testing.assert_close(net(net(x), reverse=True), x)


def test_reverse_applies_blocks_in_reverse_order():
    net = randomize_(HiNet(ArchitectureConfig(num_blocks=3))).double()
    x = torch.randn(1, 24, 4, 4, dtype=torch.float64)
    with torch.no_grad():
        expected = x
        for block in reversed(list(net.blocks)):
            expected = block(expected, reverse=True)
        torch.testing.assert_close(net(x, reverse=True), expected)


@pytest.mark.parametrize("field, value", [
    ("in_channels", 0), ("num_blocks", 0), ("clamp", 0.0),
    ("growth_channels", 0), ("init_scale", -1.0),
])
def test_config_validation(field, value):
    with pytest.raises(ValueError):
        ArchitectureConfig(**{field: value})


def test_custom_channels():
    net = HiNet(ArchitectureConfig(in_channels=1, num_blocks=2))
    assert net.channels == 8
    assert net(torch.randn(1, 8, 4, 4)).shape == (1, 8, 4, 4)
