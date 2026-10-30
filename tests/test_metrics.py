import numpy as np
import pytest

from stegawave.metrics.evaluation import calculate_psnr, calculate_ssim


def test_calculate_psnr_identical():
    img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    psnr = calculate_psnr(img, img)
    assert psnr == float("inf") or psnr > 99.0


def test_calculate_psnr_noisy():
    img1 = np.full((128, 128, 3), 100, dtype=np.uint8)
    img2 = np.full((128, 128, 3), 105, dtype=np.uint8)
    psnr = calculate_psnr(img1, img2)
    assert 30.0 < psnr < 40.0


def test_calculate_ssim_identical():
    img = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
    ssim = calculate_ssim(img, img)
    assert pytest.approx(ssim, rel=1e-3) == 1.0


def test_shape_mismatch():
    img1 = np.zeros((100, 100, 3), dtype=np.uint8)
    img2 = np.zeros((200, 200, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="Shape mismatch"):
        calculate_psnr(img1, img2)
    with pytest.raises(ValueError, match="Shape mismatch"):
        calculate_ssim(img1, img2)

