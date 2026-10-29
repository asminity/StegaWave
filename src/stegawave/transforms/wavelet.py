import torch
from torch import Tensor, nn


class DWT(nn.Module):
    """Convert BCHW images into four wavelet subbands per channel."""

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError("Expected input with shape (B, C, H, W).")

        _, _, height, width = x.shape
        if height % 2 or width % 2:
            raise ValueError("Image height and width must be even.")

        x00 = x[:, :, 0::2, 0::2] / 2
        x10 = x[:, :, 1::2, 0::2] / 2
        x01 = x[:, :, 0::2, 1::2] / 2
        x11 = x[:, :, 1::2, 1::2] / 2

        ll = x00 + x10 + x01 + x11
        hl = -x00 - x10 + x01 + x11
        lh = -x00 + x10 - x01 + x11
        hh = x00 - x10 - x01 + x11

        return torch.cat((ll, hl, lh, hh), dim=1)


class IWT(nn.Module):
    """Reconstruct BCHW images from four concatenated wavelet subbands."""

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError("Expected input with shape (B, C, H, W).")

        channels = x.shape[1]
        if channels % 4:
            raise ValueError("Input channel count must be divisible by 4.")

        ll, hl, lh, hh = x.chunk(4, dim=1)

        x00 = (ll - hl - lh + hh) / 2
        x10 = (ll - hl + lh - hh) / 2
        x01 = (ll + hl - lh - hh) / 2
        x11 = (ll + hl + lh + hh) / 2

        batch, channels_per_band, height, width = ll.shape

        output = x.new_empty(
            batch, channels_per_band, height * 2, width * 2
        )
        output[:, :, 0::2, 0::2] = x00
        output[:, :, 1::2, 0::2] = x10
        output[:, :, 0::2, 1::2] = x01
        output[:, :, 1::2, 1::2] = x11

        return output
