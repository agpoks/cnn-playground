"""ODE-Net: continuous-depth conv nets, via Neural ODEs.

Reference: Chen, Rubanova, Bettencourt, Duvenaud, "Neural Ordinary
Differential Equations", NeurIPS 2018. arXiv:1806.07366. See
papers/README.md (bibtex key `chen2018odenet`).

Compare {doc}`resnet`'s residual block: `y = x + F(x)`, one discrete Euler
step of size 1 applied a fixed number of times (one per block). The paper's
observation is that this *is* forward Euler integration of an ODE

    dh/dt = f(h(t), t, theta)

so instead of stacking N discrete blocks, parameterize f once with a small
conv net and let an ODE solver choose how finely to integrate from t=0 to
t=1. "Depth" becomes a continuous integration variable instead of a layer
count -- the same residual idea, taken to its continuous limit.

Two simplifications vs. the paper, stated explicitly:
  1. The paper uses a black-box *adaptive-step* solver (e.g. dopri5) and
     the adjoint sensitivity method to backpropagate in O(1) memory
     regardless of the number of solver steps. This repo hand-rolls a
     fixed-step RK4 integrator (`n_steps` steps, default 6) and
     backpropagates directly through the unrolled solver via ordinary
     autograd -- simpler to implement from primitives, at the cost of
     memory that scales with `n_steps` (irrelevant at this model's size).
  2. No torchdiffeq or other ODE-solver library is imported anywhere --
     RK4 is ~10 lines of tensor arithmetic, written out below.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ODEFunc(nn.Module):
    """f(h, t): the learned right-hand side dh/dt. Time t is concatenated
    to h as an extra constant-valued channel, so f is a genuine function
    of both state and time (the paper's Sec. 2 conditions its example
    ODE-nets on t the same way)."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels + 1, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels + 1, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def _cat_time(self, h: torch.Tensor, t: float) -> torch.Tensor:
        t_channel = torch.full((h.shape[0], 1, h.shape[2], h.shape[3]), float(t), device=h.device, dtype=h.dtype)
        return torch.cat([h, t_channel], dim=1)

    def forward(self, h: torch.Tensor, t: float) -> torch.Tensor:
        out = self.relu(self.norm1(h))
        out = self.conv1(self._cat_time(out, t))
        out = self.relu(self.norm2(out))
        out = self.conv2(self._cat_time(out, t))
        return out


class ODEBlock(nn.Module):
    """Integrates dh/dt = ode_func(h, t) from t=0 to t=1 via fixed-step
    RK4, and returns h(1). This replaces an entire stack of ResNet
    BasicBlocks with ONE ode_func evaluated `n_steps` times."""

    def __init__(self, ode_func: nn.Module, n_steps: int = 6):
        super().__init__()
        self.ode_func = ode_func
        self.n_steps = n_steps

    def forward(self, h0: torch.Tensor) -> torch.Tensor:
        dt = 1.0 / self.n_steps
        h = h0
        t = 0.0
        for _ in range(self.n_steps):
            k1 = self.ode_func(h, t)
            k2 = self.ode_func(h + 0.5 * dt * k1, t + 0.5 * dt)
            k3 = self.ode_func(h + 0.5 * dt * k2, t + 0.5 * dt)
            k4 = self.ode_func(h + dt * k3, t + dt)
            h = h + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            t += dt
        return h


class ODENetModel(nn.Module):
    """(B, 3, 32, 32) -> (B, num_classes). Downsampling stem (same role as
    ResNet's stem + early stages: get to a manageable spatial size before
    the expensive part), then ONE ODEBlock replacing ResNet's entire stack
    of stage-2/stage-3 BasicBlocks, then GAP + linear classifier."""

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
        self.ode_block = ODEBlock(ODEFunc(channels), n_steps=n_steps)
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
    m = ODENetModel()
    x = torch.randn(4, 3, 32, 32)
    y = m(x)
    assert y.shape == (4, 10), y.shape
    y.sum().backward()
    n_missing = sum(1 for p in m.parameters() if p.grad is None)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"output shape {tuple(y.shape)}, params {n_params}, missing grads {n_missing}")
