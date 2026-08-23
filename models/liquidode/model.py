"""Liquid-ODE: a continuous-depth conv net whose dynamics are gated like a
Liquid Time-Constant network, instead of a plain conv net.

**Honesty note, stated up front: this is NOT a single published paper.**
It is this repo's own assembly of two separately-published, real ideas that
-- as far as could be found by searching the literature while building this
repo -- have not been combined this way before (LTC/CfC papers focus on
sequential/temporal/robotics data with a small state vector; this applies
the LTC *governing equation* to a conv feature map as the ODE state):

  1. {doc}`odenet`'s continuous-depth formulation (Chen, Rubanova,
     Bettencourt, Duvenaud, "Neural Ordinary Differential Equations",
     NeurIPS 2018, arXiv:1806.07366, bibtex key `chen2018odenet`):
     depth as continuous integration of dh/dt, instead of a stack of
     discrete blocks.
  2. Liquid Time-Constant Networks' governing equation (Hasani, Lechner,
     Amini, Rus, Grosu, "Liquid Time-constant Networks", AAAI 2021,
     arXiv:2006.04439, bibtex key `hasani2021ltc`):

         dh/dt = -h / tau(h, x, theta) + S(h, x, theta) * (A - h)

     where tau (a per-unit, input/state-dependent *time constant*) and S
     (an input/state-dependent *gate*) are themselves learned functions,
     not just a plain f(h, t) -- this is what makes the dynamics "liquid":
     how fast each unit relaxes, and toward what, both depend on the
     current input and state.

odenet's `ODEFunc` is a plain conv net defining dh/dt directly. Here,
`LiquidODEFunc` instead computes tau(h,x) and S(h,x) via small conv nets
and assembles dh/dt from the LTC equation above, applied per-pixel to a
conv feature map. Everything else (stem, RK4 integrator, classifier head)
mirrors `models/odenet/model.py`'s structure for a fair, direct benchmark
comparison on the same real CIFAR-10 task/cluster -- deliberately not
importing odenet's classes across files, matching this repo's convention
of small, self-contained per-model files (see e.g. fno/pino in the sibling
sciml-playground repo for the same pattern).

Simplifications, stated explicitly:
  1. Same RK4/no-adjoint simplification as {doc}`odenet` (see that file's
     docstring) -- fixed-step RK4 (`n_steps`, default 6), plain autograd
     through the unrolled solver, no torchdiffeq or any ODE-solver library.
  2. `x` in the LTC equation (the external input driving the dynamics) is
     taken to be the block's own input h(0) at every step, re-injected at
     each RK4 stage -- the paper's LTC neuron takes a genuine external
     input signal (e.g. sensor readings) at each timestep, but this conv
     block has no such per-step external signal (unlike an RNN processing
     a sequence), so h(0) plays that role instead, i.e. the dynamics are
     "self-input-driven" from the stem's output.
  3. `A` (the equilibrium/rest state the paper's gate S pulls h toward) is
     a learned per-channel parameter, not itself a function of anything --
     the simplest faithful reading of the paper's constant bias term.
  4. `tau` is kept positive via softplus (+ a small floor) since it's a
     literal denominator in the ODE; `S` is kept in [0, 1] via sigmoid,
     matching the paper's gate semantics.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LiquidODEFunc(nn.Module):
    """dh/dt = -h/tau(h,x) + S(h,x) * (A - h), the LTC governing equation,
    with tau and S each a small conv net of [h, x] (x = the block's own
    input h(0), re-injected at every RK4 stage -- see module docstring)."""

    def __init__(self, channels: int):
        super().__init__()
        hidden = channels
        self.tau_net = nn.Sequential(
            nn.Conv2d(2 * channels, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )
        self.gate_net = nn.Sequential(
            nn.Conv2d(2 * channels, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )
        self.A = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        hx = torch.cat([h, x], dim=1)
        tau = F.softplus(self.tau_net(hx)) + 0.1
        gate = torch.sigmoid(self.gate_net(hx))
        return -h / tau + gate * (self.A - h)


class LiquidODEBlock(nn.Module):
    """Integrates the LTC equation from t=0 to t=1 via fixed-step RK4,
    holding x = h0 fixed across all steps (see module docstring, point 2).
    Same integration scheme as {doc}`odenet`'s `ODEBlock`."""

    def __init__(self, ode_func: nn.Module, n_steps: int = 6):
        super().__init__()
        self.ode_func = ode_func
        self.n_steps = n_steps

    def forward(self, h0: torch.Tensor) -> torch.Tensor:
        dt = 1.0 / self.n_steps
        h = h0
        for _ in range(self.n_steps):
            k1 = self.ode_func(h, h0)
            k2 = self.ode_func(h + 0.5 * dt * k1, h0)
            k3 = self.ode_func(h + 0.5 * dt * k2, h0)
            k4 = self.ode_func(h + dt * k3, h0)
            h = h + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return h


class LiquidODENetModel(nn.Module):
    """(B, 3, 32, 32) -> (B, num_classes). Identical stem/head shape to
    {doc}`odenet`'s `ODENetModel`, with `ODEBlock` replaced by
    `LiquidODEBlock` -- the only difference between the two models is
    what defines dh/dt."""

    def __init__(self, num_classes: int = 10, channels: int = 64, n_steps: int = 6):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32 -> 16x16
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 8x8
        )
        self.ode_block = LiquidODEBlock(LiquidODEFunc(channels), n_steps=n_steps)
        self.norm = nn.GroupNorm(8, channels)
        self.relu = nn.ReLU(inplace=True)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.ode_block(h)
        h = self.relu(self.norm(h))
        h = self.gap(h).flatten(1)
        return self.fc(h)


if __name__ == "__main__":
    m = LiquidODENetModel()
    x = torch.randn(4, 3, 32, 32)
    y = m(x)
    assert y.shape == (4, 10), y.shape
    y.sum().backward()
    n_missing = sum(1 for p in m.parameters() if p.grad is None)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"output shape {tuple(y.shape)}, params {n_params}, missing grads {n_missing}")
