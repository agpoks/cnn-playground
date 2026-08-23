"""ResNet: identity ("residual") skip connections around each block.

Reference: He, Zhang, Ren, Sun, "Deep Residual Learning for Image
Recognition", CVPR 2016. arXiv:1512.03385. See papers/README.md (bibtex
key `he2016resnet`).

Before ResNet, stacking more conv layers eventually made both training AND
test error *worse* -- not from overfitting, but because very deep networks
became hard to optimize at all (paper Fig. 1). The fix: instead of asking
each block to learn a full mapping H(x), let it learn only the RESIDUAL
F(x) = H(x) - x, and add the input back explicitly:

    y = ReLU(F(x) + shortcut(x))

If the best thing a block can do is nothing, F(x) only has to learn to
output zero -- trivial for a network to do, and gradients can flow straight
through the `+ x` term to every earlier layer regardless of how deep the
stack is. `shortcut(x)` is the identity when a block's input/output shapes
already match; when a block changes channel count or downsamples spatially
(the first block of each new stage below), `shortcut` is instead a 1x1
conv + BatchNorm "projection shortcut" (paper Sec. 3.2) so the two branches
can still be added elementwise.

This is the paper's own CIFAR-10 architecture (Sec. 4.2), not its
ImageNet one: a plain 3x3-conv stem (no aggressive 7x7-stride-2 + maxpool,
which would throw away too much of a 32x32 image immediately), then 3
stages of BasicBlocks with 16/32/64 channels, downsampling only via the
stride of each stage's first block. `n=3` blocks per stage below (a
"ResNet-20" in the paper's naming, 6n+2=20 weight layers) trades the
paper's deeper CIFAR variants (up to ResNet-110) for CPU training speed.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """[Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN] + shortcut(x), then ReLU."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            # projection shortcut: shapes changed, identity alone won't add
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


def _make_stage(in_channels: int, out_channels: int, n_blocks: int, stride: int) -> nn.Sequential:
    layers = [BasicBlock(in_channels, out_channels, stride=stride)]
    for _ in range(n_blocks - 1):
        layers.append(BasicBlock(out_channels, out_channels, stride=1))
    return nn.Sequential(*layers)


class ResNetModel(nn.Module):
    """(B, 3, 32, 32) -> (B, num_classes). The paper's CIFAR-10 ResNet-20:
    3x3 stem, 3 stages x 3 BasicBlocks each (16/32/64 channels)."""

    def __init__(self, num_classes: int = 10, n_blocks_per_stage: int = 3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.stage1 = _make_stage(16, 16, n_blocks_per_stage, stride=1)  # 32x32
        self.stage2 = _make_stage(16, 32, n_blocks_per_stage, stride=2)  # 16x16
        self.stage3 = _make_stage(32, 64, n_blocks_per_stage, stride=2)  # 8x8
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.gap(x).flatten(1)
        return self.fc(x)
