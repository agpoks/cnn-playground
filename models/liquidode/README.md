# Liquid-ODE

**Not a single paper** -- this repo's own combination of two real,
separately-published ideas (see `model.py`'s docstring for the full
honesty note):

- Chen, Rubanova, Bettencourt, Duvenaud, *"Neural Ordinary Differential
  Equations"*, NeurIPS 2018 —
  [arXiv:1806.07366](https://arxiv.org/abs/1806.07366) (continuous-depth
  formulation, same as [ODE-Net](../odenet)).
- Hasani, Lechner, Amini, Rus, Grosu, *"Liquid Time-constant Networks"*,
  AAAI 2021 — [arXiv:2006.04439](https://arxiv.org/abs/2006.04439) (the
  `dh/dt = -h/tau(h,x) + S(h,x)*(A-h)` governing equation).

See [`papers/README.md`](../../papers/README.md).

## Idea in one paragraph

[ODE-Net](../odenet)'s `ODEFunc` is a plain conv net defining `dh/dt`
directly. Liquid-ODE instead defines `dh/dt` via the Liquid
Time-Constant equation above: `tau` (how fast each unit relaxes) and `S`
(a gate) are each small conv nets of the current state and input, so both
the relaxation rate and the target state are input/state-dependent
("liquid") instead of fixed. As far as could be found by searching the
literature, LTC/CfC papers apply this equation to a small state vector
over time (sequential/robotics data); applying it to a conv feature map as
the ODE state, in place of a plain Neural-ODE `f(h,t)`, is this repo's own
assembly of the two ideas, not a reproduction of a published architecture.
Same stem/RK4-integration/classifier-head shape as ODE-Net, same real
CIFAR-10 task, so the two are directly comparable.

## Files

- `model.py` — `LiquidODEFunc` (the LTC-gated `dh/dt`) + `LiquidODEBlock`
  (fixed-step RK4 integrator, same scheme as ODE-Net) + `LiquidODENetModel`
  (downsampling stem -> one `LiquidODEBlock` -> GAP + linear classifier).
- `example.py` — trains on real CIFAR-10 (`--device {auto,cpu,cuda,mps}`).
- `example.ipynb` — same walkthrough with loss/accuracy plots.

## Run it

```bash
pip install -e .
python models/liquidode/example.py --device auto
# or open models/liquidode/example.ipynb
```
