import torch
from torch import nn


@torch.no_grad()
def randomize_(module: nn.Module, std: float = 0.05, seed: int = 0) -> nn.Module:
    """Set every parameter to N(0, std^2) so the network is not the identity."""
    generator = torch.Generator().manual_seed(seed)
    for param in module.parameters():
        param.copy_(torch.randn(param.shape, generator=generator, dtype=param.dtype) * std)
    return module
