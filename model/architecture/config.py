"""Architecture hyperparameters for the from-scratch HiNet rebuild.

Defaults follow the official HiNet code (github.com/TomTomTommi/HiNet, config.py,
invblock.py, rrdb_denselayer.py, model.py). Values were taken from that source,
not from the paper.
"""

from dataclasses import dataclass

# Number of Haar subbands per input channel (LL, HL, LH, HH).
SUBBANDS = 4


@dataclass(frozen=True)
class ArchitectureConfig:
    in_channels: int = 3  # official: channels_in = 3 (RGB)
    num_blocks: int = 16  # official: inv1 ... inv16
    clamp: float = 2.0  # official: clamp = 2.0
    growth_channels: int = 32  # official dense block growth
    negative_slope: float = 0.01  # official: nn.LeakyReLU() default slope
    init_scale: float = 0.01  # official: init_scale = 0.01

    def __post_init__(self) -> None:
        if self.in_channels < 1:
            raise ValueError("in_channels must be >= 1")
        if self.num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")
        if self.clamp <= 0:
            raise ValueError("clamp must be > 0")
        if self.growth_channels < 1:
            raise ValueError("growth_channels must be >= 1")
        if self.init_scale < 0:
            raise ValueError("init_scale must be >= 0")

    @property
    def branch_channels(self) -> int:
        """Channels per coupling branch: one image in the wavelet domain."""
        return self.in_channels * SUBBANDS
