import math

import numpy as np
import pytest
import torch

from model.evaluation.metrics import official_psnr, to_uint8, uint8_metrics


def test_official_psnr_identical_is_100():
    x = torch.rand(2, 3, 8, 8)
    assert official_psnr(x, x) == 100.0


def test_official_psnr_known_value():
    a = torch.zeros(1, 3, 4, 4)
    b = torch.full((1, 3, 4, 4), 5 / 255)
    assert official_psnr(b, a) == pytest.approx(10 * math.log10(255**2 / 25))


def test_official_psnr_is_batch_level_not_per_image_mean():
    target = torch.zeros(2, 1, 2, 2)
    pred = torch.stack([torch.full((1, 2, 2), 1 / 255), torch.full((1, 2, 2), 10 / 255)])
    batch = official_psnr(pred, target)
    per_image = np.mean([10 * math.log10(255**2 / 1), 10 * math.log10(255**2 / 100)])
    assert batch == pytest.approx(10 * math.log10(255**2 / 50.5))
    assert batch != pytest.approx(per_image)


def test_official_psnr_does_not_clip():
    target = torch.ones(1, 1, 2, 2)
    assert official_psnr(target + 0.1, target) < 100.0


def test_official_psnr_shape_mismatch():
    with pytest.raises(ValueError):
        official_psnr(torch.rand(1, 3, 4, 4), torch.rand(1, 3, 2, 2))


def test_uint8_metrics_identical_and_known_value():
    x = torch.full((1, 3, 16, 16), 100 / 255)
    same = uint8_metrics(x, x)
    assert same["psnr"] == 100.0 and same["mae"] == 0 and same["ssim"] == pytest.approx(1.0)
    off = uint8_metrics(x + 5 / 255, x)
    assert off["psnr"] == pytest.approx(10 * math.log10(255**2 / 25))
    assert off["mae"] == pytest.approx(5) and off["rmse"] == pytest.approx(5)


def test_to_uint8_rejects_batches():
    with pytest.raises(ValueError):
        to_uint8(torch.rand(2, 3, 4, 4))
