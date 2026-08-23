"""AlexNet, adapted for CIFAR-10's 32x32 input.

Reference: Krizhevsky, Sutskever, Hinton, "ImageNet Classification with
Deep Convolutional Neural Networks", NeurIPS 2012. See papers/README.md
(bibtex key `krizhevsky2012alexnet`).

The original AlexNet was designed for 224x224 ImageNet input, with a giant
11x11 stride-4 first conv to aggressively downsample before anything else
happens. Applied directly to a 32x32 CIFAR-10 image, that stride-4 kernel
would collapse the spatial size to almost nothing in one step. This module
is AlexNet's actual *recipe* -- much deeper than LeNet, ReLU instead of
tanh, dropout in the FC layers, max-pooling between conv stages -- with
kernel sizes/strides re-tuned for 32x32 input instead of 224x224: five 3x3
convs (stride 1, `padding=1`, so spatial size only changes at the three
max-pool layers) with the *same* per-layer channel progression as the
original paper (64 -> 192 -> 384 -> 256 -> 256). This is standard practice
when adapting ImageNet-era architectures to CIFAR-sized input (the same
adaptation the ResNet paper itself makes for its own CIFAR-10 experiments)
-- stated explicitly here rather than silently reproducing the 224x224
numbers on data they don't fit.

One further, CPU-speed-motivated simplification beyond the conv/stride
adaptation: the paper's FC stage is 4096-wide (256*4*4 -> 4096 -> 4096,
~33M parameters in the classifier alone), which is fine on the GPUs the
paper trains on but is the single biggest cost in this repo's CPU-only
budget -- two 4096x4096 matrix multiplies per batch dominate training time
far more than any of the conv layers above them. This module uses a
512-wide FC stage instead (256*4*4 -> 512 -> 512), keeping AlexNet's
"deeper + ReLU + dropout" recipe intact while cutting the classifier's
parameter count by roughly 64x -- reproducing the paper's exact
1000-way-ImageNet-sized FC width buys nothing on a 10-class CIFAR-10 task.

    32x32 -> Conv(3->64)   -> ReLU -> MaxPool  : 32x32 -> 16x16
           -> Conv(64->192) -> ReLU -> MaxPool  : 16x16 -> 8x8
           -> Conv(192->384) -> ReLU
           -> Conv(384->256) -> ReLU
           -> Conv(256->256) -> ReLU -> MaxPool  : 8x8 -> 4x4
    Flatten (256*4*4=4096) -> FC(512) -> Dropout -> FC(512) -> Dropout -> FC(10)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32 -> 16
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16 -> 8
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8 -> 4
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, 32, 32) -> logits: (B, num_classes)."""
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
