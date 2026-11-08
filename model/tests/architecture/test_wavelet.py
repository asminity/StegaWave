import pytest
import torch

from model.architecture.wavelet import haar_dwt, haar_iwt, low_frequency


def test_dwt_known_values():
    # [[1, 2], [3, 4]]: a=x00/2=0.5, b=x10/2=1.5, c=x01/2=1.0, d=x11/2=2.0
    x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    coefficients = haar_dwt(x)
    expected = torch.tensor([5.0, 1.0, 2.0, 0.0]).view(1, 4, 1, 1)  # LL, HL, LH, HH
    torch.testing.assert_close(coefficients, expected)


def test_dwt_channel_layout_is_subband_major():
    # Channel k of every subband block must come from input channel k.
    x = torch.zeros(1, 3, 2, 2)
    x[:, 1] = 1.0  # only channel 1 is non-zero and constant
    coefficients = haar_dwt(x)
    ll = coefficients[:, 0:3, 0, 0]
    torch.testing.assert_close(ll, torch.tensor([[0.0, 2.0, 0.0]]))
    torch.testing.assert_close(coefficients[:, 3:], torch.zeros(1, 9, 1, 1))


@pytest.mark.parametrize("shape", [(1, 3, 8, 8), (2, 3, 16, 12), (3, 1, 4, 10)])
def test_round_trip(shape):
    x = torch.randn(shape, dtype=torch.float64)
    coefficients = haar_dwt(x)
    assert coefficients.shape == (shape[0], shape[1] * 4, shape[2] // 2, shape[3] // 2)
    torch.testing.assert_close(haar_iwt(coefficients), x)


def test_inverse_round_trip():
    w = torch.randn(2, 12, 5, 7, dtype=torch.float64)
    torch.testing.assert_close(haar_dwt(haar_iwt(w)), w)


def test_orthonormal_energy_preserved():
    x = torch.randn(2, 3, 16, 16, dtype=torch.float64)
    torch.testing.assert_close(haar_dwt(x).pow(2).sum(), x.pow(2).sum())


def test_low_frequency_is_scaled_average():
    x = torch.rand(1, 3, 8, 8, dtype=torch.float64)
    expected = torch.nn.functional.avg_pool2d(x, 2) * 2
    torch.testing.assert_close(low_frequency(haar_dwt(x), 3), expected)


def test_gradients_flow_through_transforms():
    x = torch.randn(1, 3, 4, 4, requires_grad=True)
    haar_iwt(haar_dwt(x)).sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))


def test_rejects_odd_dimensions():
    with pytest.raises(ValueError):
        haar_dwt(torch.rand(1, 3, 7, 8))


def test_rejects_wrong_rank():
    with pytest.raises(ValueError):
        haar_dwt(torch.rand(3, 8, 8))
    with pytest.raises(ValueError):
        haar_iwt(torch.rand(12, 4, 4))


def test_iwt_rejects_channels_not_divisible_by_four():
    with pytest.raises(ValueError):
        haar_iwt(torch.rand(1, 10, 4, 4))
