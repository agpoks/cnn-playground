"""VGG (config "A" / VGG11), sized for CIFAR-10's 32x32 input.

Reference: Simonyan, Zisserman, "Very Deep Convolutional Networks for
Large-Scale Image Recognition", 2014. arXiv:1409.1556. See papers/README.md
(bibtex key `simonyan2014vgg`).

VGG's core idea: **only 3x3 convolutions, stacked, with 2x2 max-pool
between stages, and channel count doubling each stage** -- depth is the
only architectural knob, no varying kernel sizes the way AlexNet mixes
receptive-field sizes across layers. This module is the paper's simplest
listed configuration, "config A" / VGG11 (8 conv layers + 3 FC layers),
with its exact per-stage conv counts and channel progression (1, 1, 2, 2, 2
conv layers per stage; 64, 128, 256, 512, 512 channels):

    32x32 -> [Conv(3->64)]                    -> MaxPool : 32->16
           -> [Conv(64->128)]                  -> MaxPool : 16->8
           -> [Conv(128->256), Conv(256->256)] -> MaxPool : 8->4
           -> [Conv(256->512), Conv(512->512)] -> MaxPool : 4->2
           -> [Conv(512->512), Conv(512->512)] -> MaxPool : 2->1
    Flatten (512) -> FC(512) -> Dropout -> FC(512) -> Dropout -> FC(10)

Five stride-2 pooling stages take CIFAR-10's 32x32 down to exactly 1x1,
which is why the FC classifier here is much smaller (512 -> 512 -> 10) than
the paper's own 4096-wide classifier (built for a 7x7x512 feature map from
224x224 ImageNet input) -- a deliberate, CPU-speed-motivated reduction,
not a change to the conv stack itself, which keeps VGG11's exact recipe.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1), nn.ReLU(inplace=True))


class VGG(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(3, 64),
            nn.MaxPool2d(2, 2),  # 32 -> 16
            _conv_block(64, 128),
            nn.MaxPool2d(2, 2),  # 16 -> 8
            _conv_block(128, 256),
            _conv_block(256, 256),
            nn.MaxPool2d(2, 2),  # 8 -> 4
            _conv_block(256, 512),
            _conv_block(512, 512),
            nn.MaxPool2d(2, 2),  # 4 -> 2
            _conv_block(512, 512),
            _conv_block(512, 512),
            nn.MaxPool2d(2, 2),  # 2 -> 1
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, 32, 32) -> logits: (B, num_classes)."""
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
