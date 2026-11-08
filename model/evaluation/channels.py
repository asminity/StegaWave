"""Image loading and what happens to the stego image between hiding and revealing.

float      untouched float output (official test.py)
quantized  clamped to [0, 1] and rounded to 8 bits: exactly what a PNG holds
jpeg<q>    quantized, then JPEG-compressed at quality q (robustness check)
"""

import io
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor
from torchvision.transforms import functional as TF

from model.data import SmallImagePolicy, val_transform
from model.evaluation.metrics import to_uint8


def load_image(path: Path, crop_size: int | None, policy: SmallImagePolicy) -> Tensor:
    """(1, 3, H, W) in [0, 1]: centre crop, or the full image trimmed to even size."""
    with Image.open(path) as image:
        image = image.convert("RGB")
        if crop_size is not None:
            try:
                tensor = val_transform(crop_size, policy)(image)
            except ValueError as exc:
                raise ValueError(f"{path}: {exc}") from exc
        else:
            tensor = TF.to_tensor(image)
            height, width = tensor.shape[-2:]
            tensor = tensor[:, : height - height % 2, : width - width % 2]
    return tensor.unsqueeze(0)


def quantize_8bit(image: Tensor) -> Tensor:
    """What a lossless 8-bit file of `image` contains."""
    return image.clamp(0, 1).mul(255).round().div(255)


def jpeg_round_trip(image: Tensor, quality: int) -> Tensor:
    """Encode each image in the batch as JPEG in memory and decode it again."""
    decoded = []
    for single in image:
        buffer = io.BytesIO()
        Image.fromarray(to_uint8(single)).save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        with Image.open(buffer) as jpeg:
            decoded.append(TF.to_tensor(jpeg.convert("RGB")))
    return torch.stack(decoded).to(image.device, image.dtype)


def apply_channel(stego: Tensor, setting: str) -> Tensor:
    """The stego image as the receiver gets it under `setting` (not 'float')."""
    if setting == "quantized":
        return quantize_8bit(stego)
    if setting.startswith("jpeg"):
        return jpeg_round_trip(quantize_8bit(stego), int(setting[4:]))
    raise ValueError(f"Unknown setting {setting!r}")
