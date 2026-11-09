"""HiNet backbone: a stack of affine coupling blocks in the wavelet domain."""

from functools import partial

import torch
from torch import Tensor, nn

from model.architecture.config import ArchitectureConfig
from model.architecture.coupling import AffineCoupling
from model.architecture.initialization import init_hinet_
from model.architecture.subnet import DenseSubnet


class HiNet(nn.Module):
    """Invertible map on (B, 2 * branch_channels, H, W) wavelet tensors.

    Forward applies blocks 0..N-1; reverse applies N-1..0 with `reverse=True`.
    """

    def __init__(
        self,
        config: ArchitectureConfig | None = None,
        initialize: bool = True,
        generator: torch.Generator | None = None,
    ):
        super().__init__()
        self.config = config or ArchitectureConfig()
        branch = self.config.branch_channels

        subnet_factory = partial(
            DenseSubnet,
            growth_channels=self.config.growth_channels,
            negative_slope=self.config.negative_slope,
        )
        self.blocks = nn.ModuleList(
            AffineCoupling(branch, branch, subnet_factory, clamp=self.config.clamp)
            for _ in range(self.config.num_blocks)
        )

        if initialize:
            init_hinet_(self, self.config.init_scale, generator=generator)

    @property
    def channels(self) -> int:
        return 2 * self.config.branch_channels

    def forward(self, x: Tensor, reverse: bool = False) -> Tensor:
        blocks = list(self.blocks)
        if reverse:
            blocks.reverse()
        for block in blocks:
            x = block(x, reverse=reverse)
        return x
