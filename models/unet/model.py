"""U-Net: convolutional encoder-decoder with skip connections, for
per-pixel prediction instead of one label per image.

Reference: Ronneberger, Fischer, Brox, "U-Net: Convolutional Networks for
Biomedical Image Segmentation", MICCAI 2015. arXiv:1505.04597. See
papers/README.md (bibtex key `ronneberger2015unet`).

Every model elsewhere in this repo (LeNet..ViT) reduces an image down to
one label. U-Net keeps the same conv/pool vocabulary but reuses it for a
*dense* prediction task: one label per pixel. The trick is the decoder
(expanding path): at each upsampling stage, the decoder's feature map is
concatenated with the *same-resolution* feature map from the encoder
(contracting path) before convolving further -- a "skip connection" that
gives the decoder access to the fine spatial detail the encoder's pooling
already threw away, while still benefiting from the encoder's deep,
downsampled semantic features.

Simplification vs. the paper: the original uses unpadded ("valid") 3x3
convolutions, so the output is smaller than the input and the skip
connections need cropping. This module uses `padding=1` ("same")
convolutions throughout instead, so encoder and decoder feature maps at
matching stages are already the same spatial size -- no cropping needed,
simpler to get exactly right, output is the same H x W as the input. Also
uses 4 downsampling stages with a smaller channel count (32->64->128->256,
bottleneck 512) than the paper's (64->128->256->512, bottleneck 1024), to
keep CPU training time reasonable; the encoder/decoder *structure* (two
3x3 convs + ReLU per stage, skip connections, symmetric expand/contract)
is unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """[Conv3x3 -> ReLU -> Conv3x3 -> ReLU], same spatial size in and out."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 32):
        super().__init__()
        c1, c2, c3, c4, c5 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8, base_channels * 16

        # contracting (encoder) path
        self.enc1 = DoubleConv(in_channels, c1)
        self.enc2 = DoubleConv(c1, c2)
        self.enc3 = DoubleConv(c2, c3)
        self.enc4 = DoubleConv(c3, c4)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(c4, c5)

        # expanding (decoder) path: upsample, concat with the matching
        # encoder stage's feature map (skip connection), then DoubleConv
        self.up4 = nn.ConvTranspose2d(c5, c4, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(c5, c4)  # c4 (upsampled) + c4 (skip) = c5 in channels
        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(c4, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(c3, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(c2, c1)

        self.head = nn.Conv2d(c1, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_channels, H, W) -> (B, out_channels, H, W) logits.
        H, W must be divisible by 16 (four 2x poolings)."""
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)
