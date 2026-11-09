"""Image-level HiNet steganography: hide a secret image in a cover, then reveal it.

Hiding (forward):
    DWT(cover) ++ DWT(secret) -> HiNet -> [stego_wavelet | lost_wavelet]
    stego = IWT(stego_wavelet)

Revealing (reverse):
    stego_wavelet ++ z -> HiNet^-1 -> [recovered_cover_wavelet | recovered_secret_wavelet]
    recovered_secret = IWT(recovered_secret_wavelet)

z stands in for the discarded lost_wavelet and is sampled from N(0, 1) at reveal
time. As in the official train.py / test.py, `forward` reveals from the stego
wavelet coefficients directly rather than from DWT(IWT(stego_wavelet)).
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from model.architecture.config import ArchitectureConfig
from model.architecture.hinet import HiNet
from model.architecture.wavelet import HaarDWT, HaarIWT, low_frequency


@dataclass
class HidingOutput:
    cover_wavelet: Tensor  # (B, 4C, H/2, W/2)
    secret_wavelet: Tensor  # (B, 4C, H/2, W/2)
    stego_wavelet: Tensor  # (B, 4C, H/2, W/2)
    lost_wavelet: Tensor  # (B, 4C, H/2, W/2), discarded after hiding
    stego: Tensor  # (B, C, H, W)


@dataclass
class RevealOutput:
    recovered_cover_wavelet: Tensor  # (B, 4C, H/2, W/2)
    recovered_secret_wavelet: Tensor  # (B, 4C, H/2, W/2)
    recovered_secret: Tensor  # (B, C, H, W)
    z: Tensor  # the auxiliary variable used for the reverse pass


class HiNetSteganography(nn.Module):
    def __init__(
        self,
        config: ArchitectureConfig | None = None,
        initialize: bool = True,
        generator: torch.Generator | None = None,
    ):
        super().__init__()
        self.config = config or ArchitectureConfig()
        self.dwt = HaarDWT()
        self.iwt = HaarIWT()
        self.hinet = HiNet(self.config, initialize=initialize, generator=generator)

    @property
    def branch_channels(self) -> int:
        return self.config.branch_channels

    def _check_image(self, image: Tensor, name: str) -> None:
        if image.ndim != 4 or image.shape[1] != self.config.in_channels:
            raise ValueError(
                f"{name} must have shape (B, {self.config.in_channels}, H, W); "
                f"got {tuple(image.shape)}."
            )

    def sample_z(self, like: Tensor, generator: torch.Generator | None = None) -> Tensor:
        """Standard normal z with the shape, dtype and device of `like`."""
        if generator is None:
            return torch.randn_like(like)
        noise = torch.randn(
            like.shape, generator=generator, dtype=like.dtype, device=generator.device
        )
        return noise.to(like.device)

    def hide(self, cover: Tensor, secret: Tensor) -> HidingOutput:
        self._check_image(cover, "cover")
        self._check_image(secret, "secret")
        if cover.shape != secret.shape:
            raise ValueError(
                f"cover and secret shapes differ: {tuple(cover.shape)} vs {tuple(secret.shape)}."
            )

        cover_wavelet = self.dwt(cover)
        secret_wavelet = self.dwt(secret)
        out = self.hinet(torch.cat((cover_wavelet, secret_wavelet), dim=1))
        stego_wavelet, lost_wavelet = out.split(
            (self.branch_channels, self.branch_channels), dim=1
        )
        return HidingOutput(
            cover_wavelet=cover_wavelet,
            secret_wavelet=secret_wavelet,
            stego_wavelet=stego_wavelet,
            lost_wavelet=lost_wavelet,
            stego=self.iwt(stego_wavelet),
        )

    def reveal(
        self,
        stego_wavelet: Tensor,
        z: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> RevealOutput:
        """Reverse pass from stego wavelet coefficients."""
        if stego_wavelet.ndim != 4 or stego_wavelet.shape[1] != self.branch_channels:
            raise ValueError(
                f"stego_wavelet must have shape (B, {self.branch_channels}, H, W); "
                f"got {tuple(stego_wavelet.shape)}."
            )
        if z is None:
            z = self.sample_z(stego_wavelet, generator)
        elif z.shape != stego_wavelet.shape:
            raise ValueError(
                f"z shape {tuple(z.shape)} must match stego_wavelet {tuple(stego_wavelet.shape)}."
            )

        out = self.hinet(torch.cat((stego_wavelet, z), dim=1), reverse=True)
        recovered_cover_wavelet, recovered_secret_wavelet = out.split(
            (self.branch_channels, self.branch_channels), dim=1
        )
        return RevealOutput(
            recovered_cover_wavelet=recovered_cover_wavelet,
            recovered_secret_wavelet=recovered_secret_wavelet,
            recovered_secret=self.iwt(recovered_secret_wavelet),
            z=z,
        )

    def reveal_image(
        self,
        stego: Tensor,
        z: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> RevealOutput:
        """Reverse pass from a spatial stego image (e.g. one loaded from disk)."""
        self._check_image(stego, "stego")
        return self.reveal(self.dwt(stego), z=z, generator=generator)

    def stego_low_frequency(self, hidden: HidingOutput) -> Tensor:
        """LL band of the stego wavelet (target of HiNet's low-frequency loss)."""
        return low_frequency(hidden.stego_wavelet, self.config.in_channels)

    def cover_low_frequency(self, hidden: HidingOutput) -> Tensor:
        return low_frequency(hidden.cover_wavelet, self.config.in_channels)

    def forward(
        self,
        cover: Tensor,
        secret: Tensor,
        z: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[HidingOutput, RevealOutput]:
        """Full hide-then-reveal cycle, as used during training."""
        hidden = self.hide(cover, secret)
        revealed = self.reveal(hidden.stego_wavelet, z=z, generator=generator)
        return hidden, revealed
