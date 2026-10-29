from typing import Any

import torch
from torch import nn

from stegawave.models.hinet import Hinet
from stegawave.transforms.wavelet import DWT, IWT


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Hinet()
        self.dwt = DWT()
        self.iwt = IWT()

    @property
    def net(self) -> Hinet:
        return self.model

    def forward(self,
        cover: torch.Tensor,
        secret: torch.Tensor | None = None,
        rev: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """
        Runs the hiding (forward) and extraction (reverse) passes when both cover
        and secret images are provided.

        If secret is None, delegates directly to the invertible network
        (e.g., when evaluating intermediate 24-channel wavelet representations).
        """
        if secret is None:
            return self.model(cover, rev=rev)

        """ 1. Transform cover and secret images into the wavelet domain (B, 12, H/2, W/2) """
        dwt_cover = self.dwt(cover)
        dwt_secret = self.dwt(secret)

        """ Combine cover and secret wavelet subbands along channels: shape (B, 24, H/2, W/2) """
        x = torch.cat([dwt_cover, dwt_secret], dim=1)

        """ 2. Forward pass: Conceal secret in cover """
        output_forward = self.model(x, rev=False)

        """ Stego wavelet coefficients are the first 12 channels; lost secret/residual is the rest """
        stego_dwt = output_forward[:, :12, :, :]

        """ Reconstruct spatial stego image via IWT: shape (B, 3, H, W)"""
        stego = self.iwt(stego_dwt)

        """ 3. Reverse pass: Extract secret from stego image """

        """ Transform stego into wavelet domain """
        stego_rev_dwt = self.dwt(stego)

        """ In inference/recovery, the missing secret channels are replaced with Gaussian noise """
        noise = torch.randn_like(stego_rev_dwt)
        x_rev = torch.cat([stego_rev_dwt, noise], dim=1)

        output_reverse = self.model(x_rev, rev=True)

        """ Secret is extracted from the remaining 12 channels and reconstructed via IWT """
        recovered_secret_dwt = output_reverse[:, 12:, :, :]
        recovered_secret = self.iwt(recovered_secret_dwt)

        return stego, recovered_secret

    def conceal(
        self, 
        cover: torch.Tensor, 
        secret: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Conceals a secret image inside a cover image.

        Args:
            cover: Spatial cover image of shape (B, 3, H, W).
            secret: Spatial secret image of shape (B, 3, H, W).

        Returns:
            stego: Spatial stego image of shape (B, 3, H, W).
            lost_secret: Residual / lost secret wavelet coefficients of shape (B, 12, H/2, W/2).
        """
        dwt_cover = self.dwt(cover)
        dwt_secret = self.dwt(secret)
        x = torch.cat([dwt_cover, dwt_secret], dim=1)
        output_forward = self.model(x, rev=False)
        stego = self.iwt(output_forward[:, :12, :, :])
        lost_secret = output_forward[:, 12:, :, :]
        return stego, lost_secret

    def reveal(
        self,
        stego: torch.Tensor,
        noise: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Extracts the secret image from a stego image.

        Args:
            stego: Spatial stego image of shape (B, 3, H, W).
            noise: Optional wavelet noise tensor of shape (B, 12, H/2, W/2).
                   If None, standard Gaussian noise is sampled.

        Returns:
            recovered_secret: Spatial recovered secret image of shape (B, 3, H, W).
        """
        stego_dwt = self.dwt(stego)
        if noise is None:
            noise = torch.randn_like(stego_dwt)
        x_rev = torch.cat([stego_dwt, noise], dim=1)
        output_reverse = self.model(x_rev, rev=True)
        return self.iwt(output_reverse[:, 12:, :, :])

    def load_state_dict(
        self,
        state_dict: dict[str, Any],
        strict: bool = True,
        assign: bool = False
    ):
        """
        Flexible state dict loader supporting checkpoints saved with:
        - 'module.model.' prefix (e.g. from DataParallel)
        - 'model.' prefix
        - bare parameter names without prefix ('inv1...')
        Filters out checkpoint optimizer buffers like 'tmp_var'.
        """
        cleaned = {}
        for k, v in state_dict.items():
            if "tmp_var" in k:
                continue
            key = k
            while key.startswith("module."):
                key = key[len("module."):]
            if not key.startswith("model."):
                key = f"model.{key}"
            cleaned[key] = v

        kwargs: dict[str, Any] = {"strict": strict}
        if assign:
            kwargs["assign"] = assign
        return super().load_state_dict(cleaned, **kwargs)
