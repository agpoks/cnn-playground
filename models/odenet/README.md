# ODE-Net

**Paper:** Chen, Rubanova, Bettencourt, Duvenaud, *"Neural Ordinary
Differential Equations"*, NeurIPS 2018 —
[arXiv:1806.07366](https://arxiv.org/abs/1806.07366). See
[`papers/README.md`](../../papers/README.md).

## Idea in one paragraph

A [ResNet](../resnet) block computes `y = x + F(x)` -- one discrete Euler
step of a fixed size, repeated once per block. The paper's observation:
that *is* forward-Euler integration of an ODE `dh/dt = f(h(t), t)`. So
instead of stacking N discrete blocks, parameterize `f` once with a small
conv net and integrate it from `t=0` to `t=1` with an ODE solver. "Depth"
becomes a continuous integration variable instead of a layer count --
the same residual idea, taken to its continuous limit. This repo hand-rolls
a fixed-step RK4 integrator (`n_steps`, default 6) and backprops straight
through the unrolled solver, instead of the paper's adaptive-step solver +
adjoint method (which gives O(1) memory regardless of step count) -- a
deliberate simplification, documented in `model.py`.

## Files

- `model.py` — `ODEFunc` (the learned `dh/dt`, conditioned on time) +
  `ODEBlock` (fixed-step RK4 integrator) + `ODENetModel` (downsampling
  stem -> one `ODEBlock` -> GAP + linear classifier).
- `example.py` — trains on real CIFAR-10 (`--device {auto,cpu,cuda,mps}`).
- `example.ipynb` — same walkthrough with loss/accuracy plots.

## Run it

```bash
pip install -e .
python models/odenet/example.py --device auto
# or open models/odenet/example.ipynb
```
