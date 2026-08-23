"""MobileNet: depthwise-separable convolutions, an efficiency axis
orthogonal to depth.

Reference: Howard, Zhu, Chen, Kalenichenko, Wang, Weyand, Andreetto, Adam,
"MobileNets: Efficient Convolutional Neural Networks for Mobile Vision
Applications", 2017. arXiv:1704.04861. See papers/README.md (bibtex key
`howard2017mobilenet`).

A standard Conv3x3(in_channels -> out_channels) mixes spatial information
(the 3x3 neighborhood) AND cross-channel information (in_channels ->
out_channels) in one operation, costing roughly
`in_channels * out_channels * 9` multiply-adds per output pixel. MobileNet
factors that into two much cheaper operations in sequence:

  1. a DEPTHWISE conv: a 3x3 conv with `groups=in_channels` -- one filter
     per input channel, mixing spatial information only, no cross-channel
     mixing at all (`in_channels * 9` multiply-adds per output pixel).
  2. a POINTWISE conv: a 1x1 conv, mixing channels only, no spatial mixing
     (`in_channels * out_channels` multiply-adds per output pixel).

Total cost ~`in_channels * (9 + out_channels)` instead of
`in_channels * out_channels * 9` -- for typical channel counts this is
close to an order of magnitude fewer multiply-adds for a similar
receptive field. `example.py` builds the identical-depth *non*-separable
network alongside this one and prints both parameter counts side by side,
so the efficiency claim is demonstrated, not just asserted.

This is a CIFAR-10-sized MobileNet (7 depthwise-separable stages after one
regular stem conv) -- the paper's ImageNet version has more (13) layers at
much higher input resolution; the core building block is unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.depthwise(x)))
        x = self.relu(self.bn2(self.pointwise(x)))
        return x


_CHANNEL_SCHEDULE = [
    # (out_channels, stride)
    (32, 1),
    (64, 2),  # 32x32 -> 16x16
    (64, 1),
    (128, 2),  # 16x16 -> 8x8
    (128, 1),
    (256, 2),  # 8x8 -> 4x4
    (256, 1),
]


class MobileNetModel(nn.Module):
    """(B, 3, 32, 32) -> (B, num_classes). Regular Conv3x3 stem, then 7
    depthwise-separable stages, global average pool, one FC classifier."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        layers = []
        in_channels = 32
        for out_channels, stride in _CHANNEL_SCHEDULE:
            layers.append(DepthwiseSeparableConv(in_channels, out_channels, stride=stride))
            in_channels = out_channels
        self.stages = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        x = self.gap(x).flatten(1)
        return self.fc(x)


class PlainConvComparisonModel(nn.Module):
    """Identical depth/channel schedule to MobileNetModel, but every stage
    is one regular Conv3x3 (not depthwise-separable) -- built purely so
    example.py can print an honest side-by-side parameter-count comparison,
    not to be trained as a serious baseline itself."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        layers = []
        in_channels = 32
        for out_channels, stride in _CHANNEL_SCHEDULE:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )
            in_channels = out_channels
        self.stages = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        x = self.gap(x).flatten(1)
        return self.fc(x)
