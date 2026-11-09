"""Single-level orthonormal 2D Haar wavelet transform.

Output channel layout is [LL, HL, LH, HH], each block holding C channels, so for
an RGB image channels 0-2 are the LL (low-frequency) band. Coefficients and
layout match the official HiNet dwt_init / iwt_init (modules/Unet_common.py).
"""

import torch
from torch import Tensor, nn

from model.architecture.config import SUBBANDS


def haar_dwt(x: Tensor) -> Tensor:
    """(B, C, H, W) -> (B, 4C, H/2, W/2). H and W must be even."""
    if x.ndim != 4:
        raise ValueError(f"Expected (B, C, H, W) input; got shape {tuple(x.shape)}.")
    height, width = x.shape[-2:]
    if height % 2 or width % 2:
        raise ValueError(f"Height and width must be even; got {height}x{width}.")

    a = x[..., 0::2, 0::2] / 2  # even row, even column
    b = x[..., 1::2, 0::2] / 2  # odd row, even column
    c = x[..., 0::2, 1::2] / 2  # even row, odd column
    d = x[..., 1::2, 1::2] / 2  # odd row, odd column

    ll = a + b + c + d
    hl = -a - b + c + d
    lh = -a + b - c + d
    hh = a - b - c + d
    return torch.cat((ll, hl, lh, hh), dim=1)


def haar_iwt(x: Tensor) -> Tensor:
    """(B, 4C, H, W) -> (B, C, 2H, 2W). Exact inverse of haar_dwt."""
    if x.ndim != 4:
        raise ValueError(f"Expected (B, 4C, H, W) input; got shape {tuple(x.shape)}.")
    if x.shape[1] % SUBBANDS:
        raise ValueError(f"Channel count must be divisible by 4; got {x.shape[1]}.")

    ll, hl, lh, hh = x.chunk(SUBBANDS, dim=1)
    batch, channels, height, width = ll.shape

    out = x.new_empty(batch, channels, height * 2, width * 2)
    out[..., 0::2, 0::2] = (ll - hl - lh + hh) / 2
    out[..., 1::2, 0::2] = (ll - hl + lh - hh) / 2
    out[..., 0::2, 1::2] = (ll + hl - lh - hh) / 2
    out[..., 1::2, 1::2] = (ll + hl + lh + hh) / 2
    return out


def low_frequency(wavelet: Tensor, in_channels: int) -> Tensor:
    """LL band of a haar_dwt output: its first `in_channels` channels."""
    return wavelet[:, :in_channels]


class HaarDWT(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return haar_dwt(x)


class HaarIWT(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return haar_iwt(x)
