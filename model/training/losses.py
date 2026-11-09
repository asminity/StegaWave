"""HiNet training losses (official train.py).

    guide          = MSE(stego, cover)                      spatial domain
    reconstruction = MSE(recovered_secret, secret)          spatial domain
    low_frequency  = MSE(LL(stego_wavelet), LL(cover_wavelet))
    total          = 5 * reconstruction + 1 * guide + 1 * low_frequency

The official code builds each term with nn.MSELoss(reduce=True, size_average=False),
i.e. sum reduction, which is the default here. "mean" is available for experiments.
"""

from dataclasses import dataclass
from typing import Literal

import torch.nn.functional as F
from torch import Tensor, nn

from model.architecture.steganography import HidingOutput, RevealOutput
from model.architecture.wavelet import low_frequency

Reduction = Literal["sum", "mean"]
_REDUCTIONS = ("sum", "mean")


@dataclass(frozen=True)
class LossConfig:
    reconstruction_weight: float = 5.0  # official: lamda_reconstruction = 5
    guide_weight: float = 1.0  # official: lamda_guide = 1
    low_frequency_weight: float = 1.0  # official: lamda_low_frequency = 1
    reduction: Reduction = "sum"  # official: MSELoss(size_average=False)

    def __post_init__(self) -> None:
        for name in ("reconstruction_weight", "guide_weight", "low_frequency_weight"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.reduction not in _REDUCTIONS:
            raise ValueError(f"reduction must be one of {_REDUCTIONS}; got {self.reduction!r}")


@dataclass
class LossOutput:
    total: Tensor
    guide: Tensor
    reconstruction: Tensor
    low_frequency: Tensor

    def as_dict(self) -> dict[str, float]:
        """Detached scalar values, for logging."""
        return {
            "total": self.total.detach().item(),
            "guide": self.guide.detach().item(),
            "reconstruction": self.reconstruction.detach().item(),
            "low_frequency": self.low_frequency.detach().item(),
        }


def _mse(prediction: Tensor, target: Tensor, reduction: Reduction, name: str) -> Tensor:
    if prediction.shape != target.shape:
        raise ValueError(
            f"{name}: shape mismatch {tuple(prediction.shape)} vs {tuple(target.shape)}."
        )
    if reduction not in _REDUCTIONS:
        raise ValueError(f"reduction must be one of {_REDUCTIONS}; got {reduction!r}")
    return F.mse_loss(prediction, target, reduction=reduction)


def guide_loss(stego: Tensor, cover: Tensor, reduction: Reduction = "sum") -> Tensor:
    """Imperceptibility: the stego image should look like the cover."""
    return _mse(stego, cover, reduction, "guide_loss")


def reconstruction_loss(
    recovered_secret: Tensor, secret: Tensor, reduction: Reduction = "sum"
) -> Tensor:
    """Recovery: the revealed secret should match the original secret."""
    return _mse(recovered_secret, secret, reduction, "reconstruction_loss")


def low_frequency_loss(
    stego_wavelet: Tensor,
    cover_wavelet: Tensor,
    in_channels: int = 3,
    reduction: Reduction = "sum",
) -> Tensor:
    """Keeps the secret out of the stego's LL band (the first in_channels channels)."""
    return _mse(
        low_frequency(stego_wavelet, in_channels),
        low_frequency(cover_wavelet, in_channels),
        reduction,
        "low_frequency_loss",
    )


def hinet_losses(
    hidden: HidingOutput,
    revealed: RevealOutput,
    cover: Tensor,
    secret: Tensor,
    in_channels: int = 3,
    config: LossConfig | None = None,
) -> LossOutput:
    config = config or LossConfig()
    guide = guide_loss(hidden.stego, cover, config.reduction)
    reconstruction = reconstruction_loss(revealed.recovered_secret, secret, config.reduction)
    low = low_frequency_loss(
        hidden.stego_wavelet, hidden.cover_wavelet, in_channels, config.reduction
    )
    total = (
        config.reconstruction_weight * reconstruction
        + config.guide_weight * guide
        + config.low_frequency_weight * low
    )
    return LossOutput(total=total, guide=guide, reconstruction=reconstruction, low_frequency=low)


class HiNetLoss(nn.Module):
    def __init__(self, config: LossConfig | None = None, in_channels: int = 3):
        super().__init__()
        self.config = config or LossConfig()
        self.in_channels = in_channels

    def forward(
        self,
        hidden: HidingOutput,
        revealed: RevealOutput,
        cover: Tensor,
        secret: Tensor,
    ) -> LossOutput:
        return hinet_losses(hidden, revealed, cover, secret, self.in_channels, self.config)
