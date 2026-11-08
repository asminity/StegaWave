"""Weight initialization matching the official HiNet init_model (model.py).

Every trainable tensor (weights and biases) is drawn from N(0, init_scale^2),
then every tensor of each subnet's output conv (conv5) is set to zero. With the
output convs at zero, every subnet outputs 0, so each coupling block is the
identity map at initialization.
"""

import torch
from torch import nn

from model.architecture.subnet import DenseSubnet


@torch.no_grad()
def init_hinet_(
    module: nn.Module,
    init_scale: float = 0.01,
    generator: torch.Generator | None = None,
) -> nn.Module:
    """Initialise every DenseSubnet inside `module` in place and return `module`.

    Pass a CPU `generator` for reproducible initialisation on any device.
    """
    subnets = [m for m in module.modules() if isinstance(m, DenseSubnet)]
    if not subnets:
        raise ValueError("No DenseSubnet found to initialise.")

    for subnet in subnets:
        for layer in subnet.hidden_layers:
            for param in layer.parameters():
                noise = torch.randn(param.shape, generator=generator, dtype=param.dtype)
                param.copy_(noise.to(param.device) * init_scale)
        for param in subnet.output_layer.parameters():
            param.zero_()

    return module
