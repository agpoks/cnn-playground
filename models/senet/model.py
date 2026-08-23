"""SE-Net: a learned per-channel reweighting ("squeeze-and-excite") block.

Reference: Hu, Shen, Sun, "Squeeze-and-Excitation Networks", CVPR 2018.
arXiv:1709.01507. See papers/README.md (bibtex key `hu2018senet`).

A regular conv layer treats every output channel as equally important,
regardless of what's actually in the image. The Squeeze-and-Excitation
(SE) block adds a cheap, learned per-channel gate, computed from the
*whole* feature map (not just one channel's own receptive field):

  1. SQUEEZE: global-average-pool each channel down to one scalar, giving
     a (B, C) descriptor of "how active is each channel, on average, over
     the whole image."
  2. EXCITATION: pass that through a small bottleneck MLP
     (Linear(C -> C/r) -> ReLU -> Linear(C/r -> C) -> Sigmoid, r=16) to get
     a (B, C) set of per-channel weights in (0, 1).
  3. RESCALE: multiply the original feature map by those weights
     (broadcasting over the spatial dimensions).

The paper's whole point is that this is a **drop-in addition to an
existing architecture** ("SE-ResNet", Fig. 3), not a new backbone by
itself. This module builds a small plain CIFAR-style residual backbone
(the same BasicBlock idea as models/resnet, written independently here)
and inserts one SEBlock into each residual block's main path, right
before the skip-connection addition. `example.py` trains BOTH the plain
backbone and the SE-augmented version and reports both accuracies, so
SE's actual effect is demonstrated as an ablation, not just claimed.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        weights = self.fc(self.pool(x).view(b, c)).view(b, c, 1, 1)
        return x * weights


class ResidualBlock(nn.Module):
    """[Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN] (-> optional SEBlock) +
    shortcut(x), then ReLU. Same shape-changing shortcut logic as
    models/resnet's BasicBlock."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, use_se: bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(out_channels) if use_se else nn.Identity()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)  # channel reweighting BEFORE the residual add (paper Fig. 3)
        return self.relu(out + self.shortcut(x))


def _make_stage(in_channels, out_channels, n_blocks, stride, use_se):
    layers = [ResidualBlock(in_channels, out_channels, stride=stride, use_se=use_se)]
    for _ in range(n_blocks - 1):
        layers.append(ResidualBlock(out_channels, out_channels, stride=1, use_se=use_se))
    return nn.Sequential(*layers)


class SEResNetModel(nn.Module):
    """(B, 3, 32, 32) -> (B, num_classes). A small CIFAR-style residual
    backbone (3 stages x n_blocks, 16/32/64 channels), with SEBlocks
    inserted in every residual block when `use_se=True` -- the exact same
    backbone with `use_se=False` is the plain-ResNet ablation baseline."""

    def __init__(self, num_classes: int = 10, n_blocks_per_stage: int = 2, use_se: bool = True):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.stage1 = _make_stage(16, 16, n_blocks_per_stage, stride=1, use_se=use_se)
        self.stage2 = _make_stage(16, 32, n_blocks_per_stage, stride=2, use_se=use_se)
        self.stage3 = _make_stage(32, 64, n_blocks_per_stage, stride=2, use_se=use_se)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.gap(x).flatten(1)
        return self.fc(x)
