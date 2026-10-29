
"""Invertible coupling block used by HiNet."""

import torch
from torch import Tensor, nn

from .dense_block import ResidualDenseBlockOut


class INVBlock(nn.Module):
    def __init__(self, clamp: float = 2.0):
        super().__init__()

        self.split_len1 = 12
        self.split_len2 = 12
        self.clamp = clamp

        self.r = ResidualDenseBlockOut(self.split_len1, self.split_len2)
        self.y = ResidualDenseBlockOut(self.split_len1, self.split_len2)
        self.f = ResidualDenseBlockOut(self.split_len2, self.split_len1)

    def e(self, s: Tensor) -> Tensor:
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))

    def forward(self, x: Tensor, rev: bool = False) -> Tensor:
        x1 = x.narrow(1, 0, self.split_len1)
        x2 = x.narrow(1, self.split_len1, self.split_len2)

        if not rev:
            t2 = self.f(x2)
            y1 = x1 + t2

            s1 = self.r(y1)
            t1 = self.y(y1)

            y2 = self.e(s1) * x2 + t1
        else:
            s1 = self.r(x1)
            t1 = self.y(x1)

            y2 = (x2 - t1) / self.e(s1)
            t2 = self.f(y2)
            y1 = x1 - t2

        return torch.cat((y1, y2), dim=1)
