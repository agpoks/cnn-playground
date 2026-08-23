"""GoogLeNet / Inception: multi-branch "Inception modules" instead of one
kernel size per layer.

Reference: Szegedy, Liu, Jia, Sermanet, Reed, Anguelov, Erhan, Vanhoucke,
Rabinovich, "Going Deeper with Convolutions", CVPR 2015. arXiv:1409.4842.
See papers/README.md (bibtex key `szegedy2015googlenet`).

Every layer elsewhere in this repo commits to one kernel size. GoogLeNet's
core idea is the Inception module: run several branches with DIFFERENT
receptive fields over the SAME input in parallel, then concatenate their
outputs along the channel axis, so the network can blend fine (1x1),
medium (3x3), and coarse (5x5) detail at every stage instead of having to
pick one. This module implements the paper's "dimension-reduced" version
(Fig. 2b), not the naive one (Fig. 2a): a 1x1 conv "bottleneck" is inserted
before the 3x3 and 5x5 branches purely to cut their input channel count
first -- without it, the 5x5 branch's compute cost would grow with the
(large, unreduced) number of input channels, which is what made the naive
version impractical.

Adaptation for CIFAR-10 (32x32) vs. the paper's 224x224 ImageNet input:
the paper's stem alone (7x7 stride-2 conv + pool, then a 3x3 stride-1
conv + pool) discards 16x of resolution before a single Inception module
runs -- fine at 224x224, but it would leave almost nothing at 32x32. This
version uses a much gentler stem (two 3x3 stride-1 convs + one 2x2 pool)
and only 4 Inception modules total (the paper stacks 9), with global
average pooling and one FC layer at the end. The paper's auxiliary
classifiers (two extra loss heads part-way through the network, added to
help gradients reach the earlier layers of its full 22-layer depth) are
skipped entirely -- unnecessary at this much shallower depth.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class InceptionModule(nn.Module):
    """Four parallel branches over the same input, concatenated on the
    channel axis: 1x1 conv; 1x1 reduce -> 3x3 conv; 1x1 reduce -> 5x5 conv;
    3x3 maxpool -> 1x1 conv. Output channels = sum of the four branches'
    output channels."""

    def __init__(
        self,
        in_channels: int,
        out_1x1: int,
        reduce_3x3: int,
        out_3x3: int,
        reduce_5x5: int,
        out_5x5: int,
        out_pool: int,
    ):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_1x1, kernel_size=1), nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_3x3, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_3x3, out_3x3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_5x5, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_5x5, out_5x5, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_pool, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.out_channels = out_1x1 + out_3x3 + out_5x5 + out_pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)


class GoogLeNetModel(nn.Module):
    """(B, 3, 32, 32) -> (B, num_classes). Reduced stem + 4 Inception
    modules, sized for CIFAR-10, not the paper's 224x224/9-module depth."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 -> 16
        )
        self.inc1 = InceptionModule(96, 32, 48, 64, 8, 16, 16)  # out: 128
        self.inc2 = InceptionModule(128, 64, 64, 96, 16, 32, 32)  # out: 224
        self.pool = nn.MaxPool2d(2)  # 16 -> 8
        self.inc3 = InceptionModule(224, 96, 64, 104, 16, 32, 32)  # out: 264
        self.inc4 = InceptionModule(264, 112, 72, 144, 16, 32, 32)  # out: 320
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(320, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.inc1(x)
        x = self.inc2(x)
        x = self.pool(x)
        x = self.inc3(x)
        x = self.inc4(x)
        x = self.gap(x).flatten(1)
        return self.fc(x)
