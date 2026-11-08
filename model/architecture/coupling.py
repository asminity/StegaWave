"""Affine coupling block (the official INV_block, invblock.py).

The input is split into x1 (cover / stego branch) and x2 (secret / lost branch).

Forward:
    y1 = x1 + f(x2)
    y2 = e(r(y1)) * x2 + y(y1)
Reverse:
    x2 = (y2 - y(y1)) / e(r(y1))
    x1 = y1 - f(x2)
with e(s) = exp(clamp * 2 * (sigmoid(s) - 0.5)), so e(s) lies in (e^-c, e^c).
"""

from collections.abc import Callable

import torch
from torch import Tensor, nn

SubnetFactory = Callable[[int, int], nn.Module]


class AffineCoupling(nn.Module):
    def __init__(
        self,
        split1: int,
        split2: int,
        subnet_factory: SubnetFactory,
        clamp: float = 2.0,
    ):
        super().__init__()
        self.split1 = split1
        self.split2 = split2
        self.clamp = clamp

        self.f = subnet_factory(split2, split1)  # shift applied to x1, from x2
        self.r = subnet_factory(split1, split2)  # log-scale applied to x2, from y1
        self.y = subnet_factory(split1, split2)  # shift applied to x2, from y1

    def scale(self, s: Tensor) -> Tensor:
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))

    def forward(self, x: Tensor, reverse: bool = False) -> Tensor:
        expected = self.split1 + self.split2
        if x.ndim != 4 or x.shape[1] != expected:
            raise ValueError(
                f"Expected (B, {expected}, H, W) input; got shape {tuple(x.shape)}."
            )

        a, b = x.split((self.split1, self.split2), dim=1)

        if not reverse:
            y1 = a + self.f(b)
            y2 = self.scale(self.r(y1)) * b + self.y(y1)
            return torch.cat((y1, y2), dim=1)

        x2 = (b - self.y(a)) / self.scale(self.r(a))
        x1 = a - self.f(x2)
        return torch.cat((x1, x2), dim=1)
