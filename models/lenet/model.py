"""LeNet-5.

Reference: LeCun, Bottou, Bengio, Haffner, "Gradient-Based Learning
Applied to Document Recognition", Proceedings of the IEEE, 1998. See
papers/README.md (bibtex key `lecun1998lenet`).

The original small CNN: two conv+pool stages, then three fully-connected
layers. Deliberately NOT modernized -- it uses the paper's actual choices,
`Tanh` activations and `AvgPool` (not ReLU/MaxPool), so it reads as the
historical starting point the rest of this repo's models (AlexNet, VGG,
...) then improve on.

Applied here to real 28x28 MNIST. The original paper used 32x32 padded
input (so a 5x5 conv without padding still leaves room for the second
conv+pool stage); using 28x28 directly with `padding=2` on the first conv
reproduces the same feature-map sizes throughout the network -- a standard,
well-documented adaptation, not a departure from the architecture itself:

    Conv(1->6, 5x5, pad=2) -> Tanh -> AvgPool(2x2)   :  28x28 -> 14x14
    Conv(6->16, 5x5)       -> Tanh -> AvgPool(2x2)    :  14x14 -> 5x5
    Flatten -> FC(16*5*5->120) -> Tanh -> FC(120->84) -> Tanh -> FC(84->10)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LeNet5(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, 28, 28) -> logits: (B, num_classes)."""
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
