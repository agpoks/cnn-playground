"""NCA: Growing Neural Cellular Automata.

Reference: Mordvintsev, Randazzo, Niklasson, Levin, "Growing Neural
Cellular Automata: Differentiable Model of Morphogenesis", Distill, 2020.
https://distill.pub/2020/growing-ca/ (bibtex key `mordvintsev2020nca`).

Every other model in this repo does ONE forward pass: image in, label (or
mask, or boxes) out. NCA is a different kind of object: the learned thing
is a *local update rule*, applied identically and independently at every
cell of a grid, for many stochastic asynchronous steps -- closer in spirit
to a cellular automaton or a PDE time-stepper than to a classifier. A
single RGBA "seed" pixel, iterated under this learned rule, self-organizes
into a target pattern.

Simplifications vs. the paper, stated explicitly:
  1. Target image: the paper trains on Google Noto emoji (CC-BY-licensed
     raster glyphs). This repo procedurally generates a small synthetic
     RGBA target instead (`make_target`, a radial "flower" pattern drawn
     from closed-form trig/sigmoid expressions -- no external image asset,
     no licensing dependency). The mechanism being demonstrated (perceive
     -> tiny update net -> stochastic async update -> alive-masking,
     trained by unrolling through time) is unchanged by what the target
     actually looks like.
  2. No "sample pool" persistence trick: the paper maintains a pool of
     previously-grown states and trains on samples drawn from it (mixed
     with fresh seeds), so the rule also learns to *persist* a pattern
     indefinitely and *regenerate* it after damage. This repo trains a
     single freshly-seeded batch per iteration (grow-only) -- enough to
     demonstrate the growing objective, but not the persistence/damage-
     recovery property the pool specifically targets.
  3. Perception (fixed depthwise Sobel-x/Sobel-y/identity kernels) IS
     faithful to the paper -- only the update rule is learned, exactly as
     in the original.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

CHANNELS = 16          # RGBA (4) + hidden (12), same total channel count as the paper
LIVING_THRESHOLD = 0.1
FIRE_RATE = 0.5


def make_target(size: int = 40) -> torch.Tensor:
    """Procedurally generate a (4, size, size) RGBA target in [0, 1]
    (straight, not premultiplied, alpha) -- a simple radial "flower": a
    warm-colored disc with 5 petal lobes and a soft anti-aliased edge.
    Stand-in for the paper's emoji target, see the module docstring."""
    ys, xs = torch.meshgrid(
        torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing="ij"
    )
    r = torch.sqrt(xs**2 + ys**2)
    theta = torch.atan2(ys, xs)
    petal_radius = 0.55 + 0.12 * torch.cos(5 * theta)
    alpha = torch.sigmoid((petal_radius - r) * 18.0)  # soft edge

    red = (0.85 + 0.15 * torch.cos(theta)).clamp(0, 1)
    green = (0.35 + 0.15 * torch.sin(2 * theta)).clamp(0, 1)
    blue = (0.25 + 0.10 * r).clamp(0, 1)

    rgb = torch.stack([red, green, blue], dim=0)
    return torch.cat([rgb, alpha.unsqueeze(0)], dim=0)


def _make_perception_kernel(channels: int = CHANNELS) -> torch.Tensor:
    """Fixed (non-learned) depthwise 3x3 kernels: identity, Sobel-x,
    Sobel-y, applied per-channel -- the paper's "eyes." Shape
    (3*channels, 1, 3, 3), laid out so a `groups=channels` conv produces 3
    perception features (id, dx, dy) for every input channel."""
    identity = torch.zeros(3, 3)
    identity[1, 1] = 1.0
    sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0
    sobel_y = sobel_x.t()

    kernels = torch.stack([identity, sobel_x, sobel_y], dim=0).unsqueeze(1)  # (3, 1, 3, 3)
    return kernels.repeat(channels, 1, 1, 1)  # (3*channels, 1, 3, 3)


class NCAModel(nn.Module):
    """The full growing-CA rule: perceive (fixed) -> update_net (learned,
    two 1x1 convs) -> stochastic async update -> alive-masking. `forward`
    unrolls this for `n_steps` and returns the final state."""

    def __init__(self, channels: int = CHANNELS, hidden: int = 128, fire_rate: float = FIRE_RATE):
        super().__init__()
        self.channels = channels
        self.fire_rate = fire_rate
        self.register_buffer("perception_kernel", _make_perception_kernel(channels))
        self.update_net = nn.Sequential(
            nn.Conv2d(channels * 3, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        nn.init.zeros_(self.update_net[-1].weight)  # untrained rule = a no-op, as in the paper

    def perceive(self, state: torch.Tensor) -> torch.Tensor:
        return F.conv2d(state, self.perception_kernel, padding=1, groups=self.channels)

    def alive_mask(self, state: torch.Tensor) -> torch.Tensor:
        alpha = state[:, 3:4]
        return (F.max_pool2d(alpha, kernel_size=3, stride=1, padding=1) > LIVING_THRESHOLD).float()

    def step(self, state: torch.Tensor) -> torch.Tensor:
        pre_alive = self.alive_mask(state)
        perception = self.perceive(state)
        delta = self.update_net(perception)
        fire_mask = (
            torch.rand(state.shape[0], 1, state.shape[2], state.shape[3], device=state.device)
            <= self.fire_rate
        ).float()
        state = state + delta * fire_mask
        post_alive = self.alive_mask(state)
        return state * (pre_alive * post_alive)

    def forward(self, state: torch.Tensor, n_steps: int) -> torch.Tensor:
        for _ in range(n_steps):
            state = self.step(state)
        return state


def seed_state(batch_size: int, size: int, channels: int = CHANNELS, device="cpu") -> torch.Tensor:
    """A grid of zeros with a single alive seed cell at the center, all
    `channels` set to 1.0 there (the paper's convention)."""
    state = torch.zeros(batch_size, channels, size, size, device=device)
    mid = size // 2
    state[:, :, mid, mid] = 1.0
    return state


if __name__ == "__main__":
    m = NCAModel()
    x = seed_state(batch_size=2, size=16)
    y = m(x, n_steps=8)
    assert y.shape == (2, CHANNELS, 16, 16), y.shape
    loss = y[:, :4].mean()
    loss.backward()
    n_missing = sum(1 for p in m.parameters() if p.grad is None)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"output shape {tuple(y.shape)}, params {n_params}, missing grads {n_missing}")
