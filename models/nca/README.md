# NCA -- Growing Neural Cellular Automata

**Paper:** Mordvintsev, Randazzo, Niklasson, Levin, *"Growing Neural
Cellular Automata: Differentiable Model of Morphogenesis"*, Distill 2020 —
[distill.pub/2020/growing-ca](https://distill.pub/2020/growing-ca/). See
[`papers/README.md`](../../papers/README.md).

## Idea in one paragraph

Every other model in this repo does one forward pass: image in, label (or
mask, or boxes) out. NCA is a different kind of object -- the learned
thing is a *local update rule*, applied identically and independently at
every cell of a grid, for many stochastic asynchronous steps. Each cell
perceives its neighborhood through fixed Sobel-x/Sobel-y/identity kernels
(the "eyes," not learned), feeds that into a tiny learned update net, and
is stochastically updated (a random ~50% of cells fire each step, so the
rule can't rely on a global clock). A living-cell mask (based on a
neighborhood-alpha threshold) keeps the background at zero. Trained by
unrolling this rule for dozens of steps from a single seed cell and
comparing the final RGBA state to a target image, it self-organizes into
that target -- growth, not classification. This repo trains against a
procedurally generated target pattern (see `model.make_target`) instead of
the paper's emoji, to avoid an external asset/licensing dependency -- a
deliberate simplification, documented in `model.py`, alongside skipping
the paper's "sample pool" persistence/regeneration trick (this repo trains
grow-only, from a fresh seed every iteration).

## Files

- `model.py` — `make_target` (procedural RGBA target), the fixed Sobel/
  identity perception kernel, and `NCAModel` (learned update net +
  stochastic firing + alive-masking + step-unrolling `forward`).
- `example.py` — trains the update rule against the procedural target
  (`--device {auto,cpu,cuda,mps}`).
- `example.ipynb` — same walkthrough, plus a growth-trajectory
  visualization.

## Run it

```bash
pip install -e .
python models/nca/example.py --device auto
# or open models/nca/example.ipynb
```
