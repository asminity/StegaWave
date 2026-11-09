import pytest
import torch

from model.architecture.config import ArchitectureConfig
from model.architecture.steganography import HiNetSteganography
from model.architecture.wavelet import haar_dwt
from model.tests.helpers import randomize_

SMALL = ArchitectureConfig(num_blocks=4)


def test_output_shapes():
    model = HiNetSteganography(SMALL)
    cover, secret = torch.rand(2, 3, 16, 16), torch.rand(2, 3, 16, 16)
    hidden, revealed = model(cover, secret)

    assert hidden.stego.shape == cover.shape
    for t in (hidden.cover_wavelet, hidden.secret_wavelet, hidden.stego_wavelet,
              hidden.lost_wavelet, revealed.z, revealed.recovered_cover_wavelet,
              revealed.recovered_secret_wavelet):
        assert t.shape == (2, 12, 8, 8)
    assert revealed.recovered_secret.shape == secret.shape


def test_stego_equals_cover_at_init():
    model = HiNetSteganography()
    cover, secret = torch.rand(1, 3, 16, 16), torch.rand(1, 3, 16, 16)
    with torch.no_grad():
        hidden = model.hide(cover, secret)
    torch.testing.assert_close(hidden.stego, cover)
    torch.testing.assert_close(hidden.lost_wavelet, haar_dwt(secret))


def test_exact_recovery_when_z_is_the_lost_wavelet():
    model = randomize_(HiNetSteganography(SMALL)).double()
    cover = torch.rand(2, 3, 16, 16, dtype=torch.float64)
    secret = torch.rand(2, 3, 16, 16, dtype=torch.float64)
    with torch.no_grad():
        hidden = model.hide(cover, secret)
        revealed = model.reveal(hidden.stego_wavelet, z=hidden.lost_wavelet)
    torch.testing.assert_close(revealed.recovered_secret, secret)
    torch.testing.assert_close(revealed.recovered_cover_wavelet, hidden.cover_wavelet)


def test_reveal_image_matches_wavelet_reveal():
    model = randomize_(HiNetSteganography(SMALL)).double()
    cover = torch.rand(1, 3, 16, 16, dtype=torch.float64)
    secret = torch.rand(1, 3, 16, 16, dtype=torch.float64)
    with torch.no_grad():
        hidden = model.hide(cover, secret)
        z = model.sample_z(hidden.stego_wavelet)
        a = model.reveal(hidden.stego_wavelet, z=z).recovered_secret
        b = model.reveal_image(hidden.stego, z=z).recovered_secret
    torch.testing.assert_close(a, b)


def test_seeded_noise_is_reproducible():
    model = HiNetSteganography(SMALL)
    like = torch.zeros(2, 12, 4, 4)
    a = model.sample_z(like, torch.Generator().manual_seed(7))
    b = model.sample_z(like, torch.Generator().manual_seed(7))
    assert torch.equal(a, b)
    assert a.std() > 0.5


def test_low_frequency_helpers():
    model = HiNetSteganography(SMALL)
    hidden = model.hide(torch.rand(1, 3, 8, 8), torch.rand(1, 3, 8, 8))
    torch.testing.assert_close(model.stego_low_frequency(hidden), hidden.stego_wavelet[:, :3])
    torch.testing.assert_close(model.cover_low_frequency(hidden), hidden.cover_wavelet[:, :3])


def test_gradients_reach_every_parameter():
    model = randomize_(HiNetSteganography(SMALL))
    cover, secret = torch.rand(2, 3, 8, 8), torch.rand(2, 3, 8, 8)
    hidden, revealed = model(cover, secret, generator=torch.Generator().manual_seed(0))
    loss = (hidden.stego - cover).pow(2).sum() + (revealed.recovered_secret - secret).pow(2).sum()
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
        assert param.grad.abs().sum() > 0, name


def test_reverse_output_layers_receive_gradient_at_init():
    # Training must be able to leave the identity initialisation. At init the
    # stego equals the cover, so only the reconstruction term has gradient, and
    # it reaches the r and y output convs through the reverse pass.
    model = HiNetSteganography(SMALL)
    cover, secret = torch.rand(2, 3, 8, 8), torch.rand(2, 3, 8, 8)
    hidden, revealed = model(cover, secret)
    loss = (hidden.stego - cover).pow(2).sum() + (revealed.recovered_secret - secret).pow(2).sum()
    loss.backward()
    for block in model.hinet.blocks:
        assert block.r.output_layer.weight.grad.abs().sum() > 0
        assert block.y.output_layer.weight.grad.abs().sum() > 0


def test_input_validation():
    model = HiNetSteganography(SMALL)
    with pytest.raises(ValueError):
        model.hide(torch.rand(1, 3, 8, 8), torch.rand(1, 3, 16, 16))
    with pytest.raises(ValueError):
        model.hide(torch.rand(1, 4, 8, 8), torch.rand(1, 4, 8, 8))
    with pytest.raises(ValueError):
        model.reveal(torch.rand(1, 11, 4, 4))
    with pytest.raises(ValueError):
        model.reveal(torch.rand(1, 12, 4, 4), z=torch.rand(1, 12, 2, 2))
