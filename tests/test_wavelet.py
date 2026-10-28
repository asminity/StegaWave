import pytest
import torch

from stegawave.transforms.wavelet import DWT, IWT


@pytest.mark.parametrize("shape", [
    (1, 3, 64, 64),
    (2, 3, 128, 96),
    (1, 1, 32, 48),
])
def test_dwt_iwt_round_trip(shape):
    image = torch.rand(shape, dtype=torch.float32)

    coefficients = DWT()(image)
    reconstructed = IWT()(coefficients)

    assert coefficients.shape == (
        shape[0], shape[1] * 4, shape[2] // 2, shape[3] // 2
    )
    assert reconstructed.shape == image.shape
    torch.testing.assert_close(reconstructed, image, atol=1e-6, rtol=1e-6)


def test_dwt_rejects_odd_dimensions():
    with pytest.raises(ValueError):
        DWT()(torch.rand(1, 3, 63, 64))


def test_iwt_runs_on_cpu():
    coefficients = torch.rand(1, 12, 32, 32, device="cpu")
    output = IWT()(coefficients)

    assert output.device.type == "cpu"
    assert output.shape == (1, 3, 64, 64)
