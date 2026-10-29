
"""Residual dense subnet used by the original HiNet architecture."""

import torch
from torch import Tensor, nn


class ResidualDenseBlockOut(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()

        self.conv1 = nn.Conv2d(input_channels, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(input_channels + 32, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(input_channels + 64, 32, 3, 1, 1)
        self.conv4 = nn.Conv2d(input_channels + 96, 32, 3, 1, 1)
        self.conv5 = nn.Conv2d(input_channels + 128, output_channels, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), dim=1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), dim=1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), dim=1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), dim=1))

        return x5
