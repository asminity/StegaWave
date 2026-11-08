"""Dense convolutional subnetwork used inside each coupling block.

Matches the official ResidualDenseBlock_out (rrdb_denselayer.py): five 3x3 convs
with dense concatenation, LeakyReLU after the first four, and no residual add
(the output is the last conv's output, despite the official name).
"""

import torch
from torch import Tensor, nn


def _conv3x3(in_channels: int, out_channels: int) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)


class DenseSubnet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        growth_channels: int = 32,
        negative_slope: float = 0.01,
    ):
        super().__init__()
        g = growth_channels
        self.conv1 = _conv3x3(in_channels, g)
        self.conv2 = _conv3x3(in_channels + g, g)
        self.conv3 = _conv3x3(in_channels + 2 * g, g)
        self.conv4 = _conv3x3(in_channels + 3 * g, g)
        self.conv5 = _conv3x3(in_channels + 4 * g, out_channels)
        self.act = nn.LeakyReLU(negative_slope, inplace=True)

    @property
    def hidden_layers(self) -> tuple[nn.Conv2d, ...]:
        return (self.conv1, self.conv2, self.conv3, self.conv4)

    @property
    def output_layer(self) -> nn.Conv2d:
        return self.conv5

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.conv2(torch.cat((x, x1), dim=1)))
        x3 = self.act(self.conv3(torch.cat((x, x1, x2), dim=1)))
        x4 = self.act(self.conv4(torch.cat((x, x1, x2, x3), dim=1)))
        return self.conv5(torch.cat((x, x1, x2, x3, x4), dim=1))
