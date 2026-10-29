
"""The 16-block HiNet invertible network."""

from torch import Tensor, nn

from .invblock import INVBlock


class Hinet(nn.Module):
    def __init__(self):
        super().__init__()

        for index in range(1, 17):
            setattr(self, f"inv{index}", INVBlock(clamp=2.0))

    def forward(self, x: Tensor, rev: bool = False) -> Tensor:
        indices = range(16, 0, -1) if rev else range(1, 17)

        for index in indices:
            block = getattr(self, f"inv{index}")
            x = block(x, rev=rev)

        return x
