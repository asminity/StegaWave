"""Image-quality metrics for training and evaluation.

official_psnr reproduces the official train.py computePSNR: images are scaled
to [0, 255] but NOT rounded or clipped, the MSE is taken over the whole batch at
once (not per image), and 100 dB is returned when the MSE is below 1e-10.
Numbers from this function are comparable with the official training logs, not
with per-image uint8 PSNR.

uint8_metrics measures what a viewer of a saved 8-bit image would get: both
images are clamped to [0, 1], rounded to 8 bits, and compared per image (PSNR,
SSIM, MAE, RMSE on the 0-255 scale). PSNR is capped at 100 dB for identical
images, matching the official convention, so averages stay finite.
"""

import math

import numpy as np
from skimage.metrics import structural_similarity
from torch import Tensor

PSNR_CAP = 100.0


def _check_shapes(prediction: Tensor, target: Tensor) -> None:
    if prediction.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: {tuple(prediction.shape)} vs {tuple(target.shape)}."
        )


def _psnr_from_mse(mse: float) -> float:
    if mse < 1.0e-10:
        return PSNR_CAP
    return 10 * math.log10(255.0**2 / mse)


def official_psnr(prediction: Tensor, target: Tensor) -> float:
    _check_shapes(prediction, target)
    diff = prediction.detach().double() * 255.0 - target.detach().double() * 255.0
    return _psnr_from_mse(diff.pow(2).mean().item())


def to_uint8(image: Tensor) -> np.ndarray:
    """(3, H, W) or (1, 3, H, W) float tensor in [0, 1] -> (H, W, 3) uint8 array."""
    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError("to_uint8 expects a single image.")
        image = image[0]
    array = image.detach().cpu().clamp(0, 1).mul(255).round()
    return array.permute(1, 2, 0).numpy().astype(np.uint8)


def uint8_metrics(prediction: Tensor, target: Tensor) -> dict[str, float]:
    """Per-image PSNR / SSIM / MAE / RMSE after 8-bit quantisation of both images."""
    _check_shapes(prediction, target)
    pred = to_uint8(prediction)
    ref = to_uint8(target)
    diff = pred.astype(np.float64) - ref.astype(np.float64)
    mse = float(np.mean(diff**2))
    return {
        "psnr": _psnr_from_mse(mse),
        "ssim": float(structural_similarity(ref, pred, channel_axis=2, data_range=255)),
        "mae": float(np.mean(np.abs(diff))),
        "rmse": math.sqrt(mse),
    }
