import pytest
import torch

from model.architecture.config import ArchitectureConfig
from model.architecture.steganography import HiNetSteganography
from model.tests.helpers import randomize_
from model.training.losses import (
    HiNetLoss,
    LossConfig,
    guide_loss,
    hinet_losses,
    low_frequency_loss,
    reconstruction_loss,
)

SMALL = ArchitectureConfig(num_blocks=2)


def test_zero_for_identical_inputs():
    x = torch.rand(2, 3, 8, 8)
    assert guide_loss(x, x) == 0
    assert reconstruction_loss(x, x) == 0
    w = torch.rand(2, 12, 4, 4)
    assert low_frequency_loss(w, w) == 0


def test_sum_reduction_is_default_and_matches_official():
    a = torch.zeros(1, 3, 4, 4)
    b = torch.full((1, 3, 4, 4), 0.1)
    torch.testing.assert_close(guide_loss(b, a), torch.tensor(48 * 0.01))
    torch.testing.assert_close(guide_loss(b, a, reduction="mean"), torch.tensor(0.01))
    official = torch.nn.MSELoss(reduction="sum")(b, a)
    torch.testing.assert_close(reconstruction_loss(b, a), official)


def test_low_frequency_uses_only_ll_channels():
    cover = torch.rand(1, 12, 4, 4)
    stego = cover.clone()
    stego[:, 3:] += 1.0  # perturb HL, LH, HH only
    assert low_frequency_loss(stego, cover) == 0
    stego[:, 1] += 0.5  # perturb one LL channel
    torch.testing.assert_close(low_frequency_loss(stego, cover), torch.tensor(16 * 0.25))


def test_low_frequency_respects_in_channels():
    cover = torch.rand(1, 4, 4, 4)
    stego = cover.clone()
    stego[:, 1:] += 1.0
    assert low_frequency_loss(stego, cover, in_channels=1) == 0


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        guide_loss(torch.rand(1, 3, 8, 8), torch.rand(1, 3, 4, 4))


@pytest.mark.parametrize("kwargs", [
    {"reconstruction_weight": -1.0}, {"guide_weight": -0.1},
    {"low_frequency_weight": -2.0}, {"reduction": "max"},
])
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        LossConfig(**kwargs)


def test_default_weights_are_official():
    config = LossConfig()
    assert (config.reconstruction_weight, config.guide_weight, config.low_frequency_weight) == (5, 1, 1)
    assert config.reduction == "sum"


def _run(model, cover, secret, config=None):
    hidden, revealed = model(cover, secret, generator=torch.Generator().manual_seed(0))
    return hidden, revealed, hinet_losses(hidden, revealed, cover, secret, config=config)


def test_total_is_weighted_sum():
    model = randomize_(HiNetSteganography(SMALL))
    cover, secret = torch.rand(2, 3, 8, 8), torch.rand(2, 3, 8, 8)
    config = LossConfig(reconstruction_weight=2.0, guide_weight=3.0, low_frequency_weight=0.5)
    _, _, out = _run(model, cover, secret, config)
    expected = 2.0 * out.reconstruction + 3.0 * out.guide + 0.5 * out.low_frequency
    torch.testing.assert_close(out.total, expected)
    assert out.guide > 0 and out.reconstruction > 0 and out.low_frequency > 0


def test_terms_match_standalone_functions():
    model = randomize_(HiNetSteganography(SMALL))
    cover, secret = torch.rand(2, 3, 8, 8), torch.rand(2, 3, 8, 8)
    hidden, revealed, out = _run(model, cover, secret)
    torch.testing.assert_close(out.guide, guide_loss(hidden.stego, cover))
    torch.testing.assert_close(out.reconstruction, reconstruction_loss(revealed.recovered_secret, secret))
    torch.testing.assert_close(
        out.low_frequency, low_frequency_loss(hidden.stego_wavelet, hidden.cover_wavelet)
    )


def test_hiding_losses_vanish_at_identity_init():
    model = HiNetSteganography(SMALL)
    cover, secret = torch.rand(2, 3, 8, 8), torch.rand(2, 3, 8, 8)
    _, _, out = _run(model, cover, secret)
    assert out.low_frequency == 0
    assert out.guide < 1e-8
    assert out.reconstruction > 0


def test_module_wrapper_and_backward():
    model = randomize_(HiNetSteganography(SMALL))
    criterion = HiNetLoss()
    cover, secret = torch.rand(2, 3, 8, 8), torch.rand(2, 3, 8, 8)
    hidden, revealed = model(cover, secret)
    out = criterion(hidden, revealed, cover, secret)
    out.total.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    logged = out.as_dict()
    assert set(logged) == {"total", "guide", "reconstruction", "low_frequency"}
    assert all(isinstance(v, float) for v in logged.values())
