# Legendre-KAN-Conv

**Papers:** Liu, Wang, Vaidya, Ruehle, Halverson, Soljačić, Hou, Tegmark,
*"KAN: Kolmogorov-Arnold Networks"*, 2024 —
[arXiv:2404.19756](https://arxiv.org/abs/2404.19756); Bodner, Tepsich,
Spolski, Pourteau, *"Convolutional Kolmogorov-Arnold Networks"*, 2024 —
[arXiv:2406.13155](https://arxiv.org/abs/2406.13155). See
[`papers/README.md`](../../papers/README.md).

## Idea in one paragraph

KAN replaces a linear weight + fixed nonlinearity with a *learnable
univariate function* per input-output edge (originally a B-spline).
ConvKAN puts that idea inside a convolution: instead of one learned scalar
weight per kernel tap, each tap gets a learned function of the input value
there. This model uses a truncated Legendre-polynomial expansion as that
per-tap function -- cheaper than a B-spline (a smooth global basis via a
three-term recurrence, no piecewise/De Boor machinery). The input is
`tanh`-squashed to `[-1, 1]` (where Legendre polynomials are defined), the
degree-K expansion is concatenated along the channel dimension, and one
ordinary `Conv2d` consumes it -- that conv's weights *are* the per-edge
Legendre coefficients. A parallel `SiLU(x)`-then-`Conv2d` "base" path is
added, mirroring the original KAN paper's base+spline residual design.

**Honesty note:** there is no single canonical paper for "Legendre-basis
ConvKAN" specifically -- this follows the general ConvKAN pattern (Bodner
et al. use B-splines) as implemented by community projects like
`torch-conv-kan` (whose Legendre-based `ResKANet` reports 84.17% on
CIFAR-10), rather than reproducing one dedicated paper. Documented in
`model.py`.

## Files

- `model.py` — `legendre_basis` (three-term recurrence) +
  `LegendreKANConv2d` (basis expansion -> conv, + base path) +
  `LegendreKANModel` (three KAN-conv blocks with strided downsampling
  between them, then GAP + linear classifier).
- `example.py` — trains on real CIFAR-10 (`--device {auto,cpu,cuda,mps}`).
- `example.ipynb` — same walkthrough with loss/accuracy plots.

## Run it

```bash
pip install -e .
python models/legendrekan/example.py --device auto
# or open models/legendrekan/example.ipynb
```
