"""Legendre-KAN-Conv: a Kolmogorov-Arnold convolutional network using a
Legendre-polynomial edge-function basis.

References: Liu et al., "KAN: Kolmogorov-Arnold Networks", 2024,
arXiv:2404.19756 (bibtex key `liu2024kan`); Bodner et al., "Convolutional
Kolmogorov-Arnold Networks", 2024, arXiv:2406.13155 (bibtex key
`bodner2024convkan`). See papers/README.md.

Ordinary KAN replaces a linear weight + fixed nonlinearity with one
*learnable univariate function* per input-output edge (originally a
B-spline). ConvKAN puts that idea inside a convolution: instead of a
single learned scalar weight per kernel tap, each tap gets a learned
univariate function of the input value at that tap. This file uses a
truncated Legendre-polynomial expansion as that univariate function
(cheaper than a B-spline: no piecewise/De Boor machinery, a smooth global
basis over [-1, 1] via a three-term recurrence).

Honesty note: there is no single canonical paper for "Legendre-basis
ConvKAN" specifically -- Bodner et al. establish the general ConvKAN
pattern (they use B-splines); Legendre (and Chebyshev, Fourier, wavelet)
variants exist mainly as community implementations (e.g. the
`torch-conv-kan` GitHub project's `ResKANet`, which reports 84.17% on
CIFAR-10 with Legendre convolutions) rather than one dedicated paper. The
mechanism implemented here -- degree-K Legendre expansion of a squashed
input, folded into channels, then one ordinary conv -- follows that
general community pattern.

Implementation trick, also used by the real ConvKAN implementations for
efficiency: rather than literally allocating a separate weight per
(kernel tap, polynomial degree) pair, the K+1 Legendre-polynomial
transforms of the (tanh-squashed) input are concatenated along the
channel dimension, and a single ordinary `nn.Conv2d` is applied to the
expanded tensor. That conv's learned weights ARE the per-edge Legendre
coefficients -- mathematically identical to a "one polynomial-weight
vector per tap" formulation, just expressed as one matmul instead of K+1
small ones.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def legendre_basis(x: torch.Tensor, degree: int) -> torch.Tensor:
    """x: (B, C, H, W) already squashed to [-1, 1]. Returns (B, C*(degree+1), H, W),
    the channel-concatenated P_0(x) .. P_degree(x), via the standard
    three-term recurrence P_0=1, P_1=x, (n+1) P_{n+1} = (2n+1) x P_n - n P_{n-1}.
    """
    polys = [torch.ones_like(x), x]
    for n in range(1, degree):
        p_next = ((2 * n + 1) * x * polys[n] - n * polys[n - 1]) / (n + 1)
        polys.append(p_next)
    polys = polys[: degree + 1]
    return torch.cat(polys, dim=1)


class LegendreKANConv2d(nn.Module):
    """One KAN-style conv layer: per-tap learned function = degree-K
    Legendre expansion of the (tanh-squashed) input, folded into channels
    and consumed by one ordinary Conv2d. A parallel SiLU(x)-then-Conv2d
    "base" path is added on top, mirroring the original KAN paper's
    base-function + learned-spline residual design (there, base=SiLU;
    here, base=SiLU too, spline replaced by the Legendre term)."""

    def __init__(self, in_channels, out_channels, kernel_size=3, degree=2, stride=1, padding=1):
        super().__init__()
        self.degree = degree
        self.poly_conv = nn.Conv2d(in_channels * (degree + 1), out_channels, kernel_size, stride=stride, padding=padding)
        self.base_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_squashed = torch.tanh(x)
        basis = legendre_basis(x_squashed, self.degree)
        return self.poly_conv(basis) + self.base_conv(F.silu(x))


class LegendreKANModel(nn.Module):
    """(B, 3, 32, 32) -> (B, num_classes). Three LegendreKANConv2d blocks
    (each: KAN conv -> BatchNorm -> ReLU), with an ordinary strided conv
    for downsampling between the first two (downsampling itself carries
    no "edge function" content worth expanding -- only the KAN layers do
    the Legendre-basis feature extraction), then global average pool and
    a linear classifier."""

    def __init__(self, num_classes: int = 10, degree: int = 2, c1: int = 32, c2: int = 64, c3: int = 64):
        super().__init__()
        self.kan1 = LegendreKANConv2d(3, c1, kernel_size=3, degree=degree, padding=1)
        self.bn1 = nn.BatchNorm2d(c1)
        self.down1 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32 -> 16x16
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )

        self.kan2 = LegendreKANConv2d(c1, c2, kernel_size=3, degree=degree, padding=1)
        self.bn2 = nn.BatchNorm2d(c2)
        self.down2 = nn.Sequential(
            nn.Conv2d(c2, c2, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 8x8
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )

        self.kan3 = LegendreKANConv2d(c2, c3, kernel_size=3, degree=degree, padding=1)
        self.bn3 = nn.BatchNorm2d(c3)

        self.relu = nn.ReLU(inplace=True)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c3, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.relu(self.bn1(self.kan1(x)))
        h = self.down1(h)
        h = self.relu(self.bn2(self.kan2(h)))
        h = self.down2(h)
        h = self.relu(self.bn3(self.kan3(h)))
        h = self.gap(h).flatten(1)
        return self.fc(h)


if __name__ == "__main__":
    m = LegendreKANModel()
    x = torch.randn(4, 3, 32, 32)
    y = m(x)
    assert y.shape == (4, 10), y.shape
    y.sum().backward()
    n_missing = sum(1 for p in m.parameters() if p.grad is None)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"output shape {tuple(y.shape)}, params {n_params}, missing grads {n_missing}")
