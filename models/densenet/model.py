"""DenseNet: every layer's output feeds every LATER layer in its block, not
just the next one.

Reference: Huang, Liu, Van Der Maaten, Weinberger, "Densely Connected
Convolutional Networks", CVPR 2017. arXiv:1608.06993. See papers/README.md
(bibtex key `huang2017densenet`).

ResNet (models/resnet) adds a block's input back to its output. DenseNet
goes further: inside a DenseBlock, layer i's input is the channel-wise
CONCATENATION of the block's original input plus every earlier layer's
output within that block -- not a sum, a concatenation, so no information
computed anywhere in the block is ever discarded before the block ends.
Layer i therefore has `in_channels + i * growth_rate` input channels,
where `growth_rate` (k) is how many new channels each layer contributes.
This maximizes feature reuse: a later layer can directly use a feature an
early layer computed, instead of the network having to re-derive it.

Each DenseBlock layer here is "DenseNet-B" (paper Sec. 3, bottleneck
variant): `[BN -> ReLU -> 1x1 Conv (bottleneck, 4*growth_rate channels) ->
BN -> ReLU -> 3x3 Conv (-> growth_rate channels)]` -- the 1x1 bottleneck
keeps the 3x3 conv's input channel count bounded even as the block grows,
which matters because that input count grows every layer. Between
DenseBlocks, a "transition layer" (`[BN -> ReLU -> 1x1 Conv (channel
compression) -> 2x2 AvgPool]`, "DenseNet-C") halves both channel count and
spatial size, keeping the whole network's channel growth under control.

Sized for CIFAR-10 and CPU training speed: 3 DenseBlocks of 4 layers each,
growth_rate=12, compression=0.5 -- much shallower than the paper's deeper
CIFAR configs (which stack up to 100+ layers total).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _DenseLayer(nn.Module):
    def __init__(self, in_channels: int, growth_rate: int):
        super().__init__()
        bottleneck_channels = 4 * growth_rate
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channels, growth_rate, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)  # (B, growth_rate, H, W) -- caller concatenates onto x


class DenseBlock(nn.Module):
    """`n_layers` _DenseLayers; layer i sees the concatenation of the
    block's input and every previous layer's output (in_channels +
    i*growth_rate input channels)."""

    def __init__(self, in_channels: int, growth_rate: int, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [_DenseLayer(in_channels + i * growth_rate, growth_rate) for i in range(n_layers)]
        )
        self.out_channels = in_channels + n_layers * growth_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for layer in self.layers:
            new_features = layer(torch.cat(features, dim=1))
            features.append(new_features)
        return torch.cat(features, dim=1)


class TransitionLayer(nn.Module):
    """[BN -> ReLU -> 1x1 Conv (compress channels) -> 2x2 AvgPool (halve spatial size)]."""

    def __init__(self, in_channels: int, compression: float = 0.5):
        super().__init__()
        out_channels = int(in_channels * compression)
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(2),
        )
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DenseNetModel(nn.Module):
    """(B, 3, 32, 32) -> (B, num_classes). 3 DenseBlocks (4 layers each,
    growth_rate=12) with a compression-0.5 transition between each pair."""

    def __init__(self, num_classes: int = 10, growth_rate: int = 12, n_layers_per_block: int = 4):
        super().__init__()
        stem_channels = 2 * growth_rate
        self.stem = nn.Conv2d(3, stem_channels, kernel_size=3, padding=1, bias=False)

        channels = stem_channels
        self.block1 = DenseBlock(channels, growth_rate, n_layers_per_block)
        channels = self.block1.out_channels
        self.trans1 = TransitionLayer(channels)
        channels = self.trans1.out_channels

        self.block2 = DenseBlock(channels, growth_rate, n_layers_per_block)
        channels = self.block2.out_channels
        self.trans2 = TransitionLayer(channels)
        channels = self.trans2.out_channels

        self.block3 = DenseBlock(channels, growth_rate, n_layers_per_block)
        channels = self.block3.out_channels

        self.bn_final = nn.BatchNorm2d(channels)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.trans1(self.block1(x))
        x = self.trans2(self.block2(x))
        x = self.block3(x)
        x = nn.functional.relu(self.bn_final(x), inplace=True)
        x = self.gap(x).flatten(1)
        return self.fc(x)
