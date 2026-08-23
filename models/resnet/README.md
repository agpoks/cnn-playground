# ResNet

**Paper:** He, Zhang, Ren, Sun, *"Deep Residual Learning for Image
Recognition"*, CVPR 2016 —
[arXiv:1512.03385](https://arxiv.org/abs/1512.03385). See
[`papers/README.md`](../../papers/README.md).

## Idea in one paragraph

Before ResNet, stacking more conv layers eventually made networks *harder*
to train, not just prone to overfitting. The fix: instead of asking a
block to learn a full mapping `H(x)`, let it learn only the residual
`F(x) = H(x) - x` and add the input back explicitly: `y = ReLU(F(x) +
shortcut(x))`. If the best thing a block can do is nothing, `F(x)` only
has to learn to output zero — trivial — and gradients flow straight
through the `+ x` term to every earlier layer no matter how deep the
stack. `shortcut` is the identity when shapes already match, or a `1x1
Conv + BatchNorm` "projection shortcut" when a block changes channel
count or downsamples.

**This is the paper's own CIFAR-10 architecture** (Sec. 4.2), not its
ImageNet one: a plain 3x3 stem (no aggressive 7x7-stride-2+pool), 3 stages
of BasicBlocks at 16/32/64 channels. `model.py` uses `n=3` blocks per stage
("ResNet-20"), smaller than the paper's deeper CIFAR variants (up to
ResNet-110), for CPU training speed.

## Files

- `model.py` — `BasicBlock` (residual block with identity/projection
  shortcut) + `ResNetModel` (the paper's CIFAR-10 ResNet-20).
- `example.py` — trains on real CIFAR-10 (`--device {auto,cpu,cuda,mps}`).
- `example.ipynb` — same walkthrough with loss/accuracy plots.

## Run it

```bash
pip install -e .
python models/resnet/example.py --device auto
# or open models/resnet/example.ipynb
```
