"""OBF-Conv: a convolution whose spatial kernel is constrained to the span
of a small number of fixed Kautz/Laguerre orthonormal basis functions
(OBFs), instead of having every tap learned freely.

**Honesty note, stated up front: no CNN-kernel paper does this.** Kautz
and Laguerre orthonormal basis functions are real, well-established tools
{cite}`wahlberg1991laguerre,oliveira2011obf` -- but exclusively in linear
*system identification* / Volterra-kernel-expansion literature: given a
prior belief about a system's dominant time constant (Laguerre, one real
pole) or resonant frequency (Kautz, a complex-conjugate pole pair), an
FIR/Volterra impulse response can be represented compactly as a short
linear combination of B fixed OBF sequences instead of many free taps.
As far as could be found by searching the literature while building this
repo, nobody has applied this idea as a *spatial* CNN kernel
parameterization for 2D (or 3D) vision. This file is this repo's own
from-scratch transplant of that idea, not a reproduction of any paper's
architecture -- read every claim below as "this repo's design."

Conceptual contrast with {doc}`legendrekan` (built just before this
file): Legendre-KAN-Conv puts its polynomial basis over the *pixel
intensity value* at a tap (a KAN-style learned nonlinear edge function).
OBF-Conv puts its basis over the *tap/spatial index* of the kernel itself
-- i.e. it constrains the *shape* of the kernel's receptive field, not
the function applied to what's under it. That is the faithful transplant
of what Kautz/Laguerre OBFs actually do in system identification: they
are a basis for representing an impulse response's *shape* compactly,
given a decay/resonance prior, exactly analogous to constraining a conv
kernel's shape here.

Construction, all real (not fabricated) DSP recursions, implemented by
directly simulating the difference equations rather than importing any
OBF library:

  Laguerre (`generate_laguerre_basis`, one real pole `xi` in (0,1)):
  l_0 is the impulse response of a first-order lowpass gain stage,
  `s[k] = xi*s[k-1] + u[k]`, `l_0[k] = sqrt(1-xi^2) * s[k]`, driven by a
  unit impulse -- giving the closed form `l_0[k] = sqrt(1-xi^2) * xi^k`.
  Each subsequent `l_b` is obtained by passing `l_{b-1}` through the
  first-order all-pass section `y[k] = xi*y[k-1] - xi*u[k] + u[k-1]`
  (transfer function `(z^-1 - xi)/(1 - xi*z^-1)`), cascaded b times --
  this is the standard Laguerre OBF cascade realization
  {cite}`wahlberg1991laguerre`.

  Kautz (`generate_kautz_basis`, resonant pole pair `r*exp(+-j*theta)`):
  built from a second-order resonant section
  `y[k] = 2*r*cos(theta)*y[k-1] - r^2*y[k-2] + u[k]`. The first pair of
  basis sequences are that section's responses to a unit impulse at
  k=0 and at k=1 respectively (an impulse and its one-sample-delayed
  twin excite genuinely different phases of the same resonance, unlike
  driving the same k=0 impulse with two different amplitudes, which
  turns out to give nearly-parallel, not phase-quadrature, sequences --
  verified numerically while building this). Further basis sequences
  cascade the same resonant section, mirroring the Laguerre construction.

  Both raw sequences are only *approximately* orthonormal at finite
  `kernel_size` (exact orthonormality of the underlying OBF construction
  is an infinite-impulse-response property; truncating to a finite kernel
  breaks it slightly -- verified numerically: the raw Laguerre Gram matrix
  is already close to the identity for reasonable `kernel_size`, while the
  raw Kautz Gram matrix is not). Both are therefore passed through
  Gram-Schmidt at the end to guarantee an exactly orthonormal finite-length
  basis regardless -- stated explicitly, not hidden.

`OBFConv2d` builds a 2D basis via the outer product of the (orthonormal)
1D basis with itself (a *separable* simplification -- the true 2D-optimal
OBF basis is not generally a separable product, but this keeps the
construction simple and still spans a meaningfully constrained subspace),
giving `n_basis**2` fixed 2D filters. Only a `(out_channels, in_channels,
n_basis**2)` tensor of combination coefficients is learned; the actual
`(out_channels, in_channels, kernel_size, kernel_size)` conv weight is
assembled from these coefficients times the fixed basis at every forward
pass via `torch.einsum`, then consumed by `F.conv2d` -- i.e. the kernel
*shape* is constrained to a low-dimensional, DSP-motivated subspace,
while an ordinary conv's kernel shape is unconstrained.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def generate_laguerre_basis(n_basis: int, kernel_size: int, pole: float) -> torch.Tensor:
    """Returns (n_basis, kernel_size): l_0 (lowpass impulse response) then
    n_basis-1 further sequences from cascading the first-order all-pass
    section. Real recursion, see module docstring."""
    xi = pole
    seqs = []

    s = 0.0
    l0 = []
    for k in range(kernel_size):
        u = 1.0 if k == 0 else 0.0
        s = xi * s + u
        l0.append(math.sqrt(1 - xi**2) * s)
    seqs.append(l0)

    prev = l0
    for _ in range(1, n_basis):
        y_prev, u_prev = 0.0, 0.0
        cur = []
        for k in range(kernel_size):
            u = prev[k]
            y = xi * y_prev - xi * u + u_prev
            cur.append(y)
            y_prev, u_prev = y, u
        seqs.append(cur)
        prev = cur

    return torch.tensor(seqs, dtype=torch.float32)


def generate_kautz_basis(n_basis: int, kernel_size: int, r: float, theta: float) -> torch.Tensor:
    """Returns (n_basis, kernel_size): a resonant-pole-pair impulse-response
    pair (b0, b1, from an impulse at k=0 and at k=1) then further sequences
    from cascading the same second-order section. Real recursion, see
    module docstring."""

    def resonant_response(impulse_k: int):
        y1, y2 = 0.0, 0.0
        seq = []
        for k in range(kernel_size):
            u = 1.0 if k == impulse_k else 0.0
            y = 2 * r * math.cos(theta) * y1 - r * r * y2 + u
            seq.append(y)
            y2, y1 = y1, y
        return seq

    b0 = resonant_response(0)
    seqs = [b0]
    if n_basis > 1:
        seqs.append(resonant_response(1))

    prev = b0
    while len(seqs) < n_basis:
        y1, y2 = 0.0, 0.0
        cur = []
        for k in range(kernel_size):
            u = prev[k]
            y = 2 * r * math.cos(theta) * y1 - r * r * y2 + u
            cur.append(y)
            y2, y1 = y1, y
        seqs.append(cur)
        prev = cur

    return torch.tensor(seqs[:n_basis], dtype=torch.float32)


def _gram_schmidt(mat: torch.Tensor) -> torch.Tensor:
    """mat: (n_basis, kernel_size) -> exactly orthonormal (n_basis, kernel_size).
    Corrects the finite-kernel-size truncation error the raw OBF
    recursions leave behind (see module docstring)."""
    out = []
    for row in mat:
        v = row.clone()
        for u in out:
            v = v - (v @ u) * u
        out.append(v / v.norm())
    return torch.stack(out)


class OBFConv2d(nn.Module):
    """Conv layer whose (kernel_size, kernel_size) spatial kernel is
    constrained to the span of `n_basis**2` fixed, orthonormal Kautz/
    Laguerre-generated 2D filters (separable outer product of the 1D
    basis), with only the combination coefficients learned."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        n_basis: int = 4,
        basis: str = "laguerre",
        stride: int = 1,
        padding: int | None = None,
        pole: float = 0.5,
        kautz_r: float = 0.75,
        kautz_theta: float = 1.2,
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.stride = stride
        self.padding = padding
        self.n_basis = n_basis

        if basis == "laguerre":
            basis_1d = generate_laguerre_basis(n_basis, kernel_size, pole)
        elif basis == "kautz":
            basis_1d = generate_kautz_basis(n_basis, kernel_size, kautz_r, kautz_theta)
        else:
            raise ValueError(f"unknown basis {basis!r}, expected 'laguerre' or 'kautz'")
        basis_1d = _gram_schmidt(basis_1d)

        basis_2d = torch.einsum("ik,jl->ijkl", basis_1d, basis_1d).reshape(
            n_basis * n_basis, kernel_size, kernel_size
        )
        self.register_buffer("basis_2d", basis_2d)

        fan_in = in_channels * n_basis * n_basis
        self.coeffs = nn.Parameter(torch.randn(out_channels, in_channels, n_basis * n_basis) / math.sqrt(fan_in))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.einsum("oib,bkl->oikl", self.coeffs, self.basis_2d)
        return F.conv2d(x, weight, stride=self.stride, padding=self.padding)


class OBFConvModel(nn.Module):
    """(B, 3, 32, 32) -> (B, num_classes). Three OBFConv2d blocks (each:
    OBF conv -> BatchNorm -> ReLU) with ordinary strided convs for
    downsampling between them (same shape as {doc}`legendrekan`'s
    LegendreKANModel, for a direct benchmark comparison), then global
    average pool and a linear classifier."""

    def __init__(
        self,
        num_classes: int = 10,
        basis: str = "laguerre",
        n_basis: int = 4,
        kernel_size: int = 7,
        c1: int = 32,
        c2: int = 64,
        c3: int = 64,
    ):
        super().__init__()
        self.conv1 = OBFConv2d(3, c1, kernel_size=kernel_size, n_basis=n_basis, basis=basis)
        self.bn1 = nn.BatchNorm2d(c1)
        self.down1 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32 -> 16x16
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )

        self.conv2 = OBFConv2d(c1, c2, kernel_size=kernel_size, n_basis=n_basis, basis=basis)
        self.bn2 = nn.BatchNorm2d(c2)
        self.down2 = nn.Sequential(
            nn.Conv2d(c2, c2, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 8x8
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )

        self.conv3 = OBFConv2d(c2, c3, kernel_size=kernel_size, n_basis=n_basis, basis=basis)
        self.bn3 = nn.BatchNorm2d(c3)

        self.relu = nn.ReLU(inplace=True)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c3, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.relu(self.bn1(self.conv1(x)))
        h = self.down1(h)
        h = self.relu(self.bn2(self.conv2(h)))
        h = self.down2(h)
        h = self.relu(self.bn3(self.conv3(h)))
        h = self.gap(h).flatten(1)
        return self.fc(h)


if __name__ == "__main__":
    for basis in ("laguerre", "kautz"):
        raw = generate_laguerre_basis(4, 11, 0.5) if basis == "laguerre" else generate_kautz_basis(4, 11, 0.75, 1.2)
        ortho = _gram_schmidt(raw)
        gram = ortho @ ortho.T
        off_diag_err = (gram - torch.eye(4)).abs().max().item()
        print(f"[{basis}] orthonormality check: max |Gram - I| = {off_diag_err:.2e}")

        m = OBFConvModel(basis=basis)
        x = torch.randn(4, 3, 32, 32)
        y = m(x)
        assert y.shape == (4, 10), y.shape
        y.sum().backward()
        n_missing = sum(1 for p in m.parameters() if p.grad is None)
        n_params = sum(p.numel() for p in m.parameters())
        print(f"[{basis}] output shape {tuple(y.shape)}, params {n_params}, missing grads {n_missing}")
