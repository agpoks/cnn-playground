# DenseNet

**Paper:** Huang, Liu, Van Der Maaten, Weinberger, *"Densely Connected
Convolutional Networks"*, CVPR 2017 —
[arXiv:1608.06993](https://arxiv.org/abs/1608.06993). See
[`papers/README.md`](../../papers/README.md).

## Idea in one paragraph

[ResNet](../resnet) adds a block's input back to its output (a sum).
DenseNet goes further: inside a DenseBlock, layer `i`'s input is the
channel-wise *concatenation* of the block's original input plus every
earlier layer's output in that block — not a sum, so nothing computed
anywhere in the block is ever discarded before the block ends. Layer `i`
therefore sees `in_channels + i * growth_rate` input channels. This
maximizes feature reuse: a later layer can use an early layer's feature
directly instead of the network re-deriving it. A 1x1-bottleneck before
each layer's 3x3 conv ("DenseNet-B") keeps that growing input channel
count from making every layer expensive; a "transition layer" between
blocks (1x1 conv channel compression + 2x2 avg-pool, "DenseNet-C") keeps
the whole network's channel count under control.

**Sized down for CIFAR-10/CPU speed:** 3 DenseBlocks of 4 layers each,
growth_rate=12 — much shallower than the paper's 100+-layer CIFAR configs,
same block/transition structure.

## Files

- `model.py` — `DenseBlock`/`_DenseLayer` (bottleneck, concatenated
  connectivity), `TransitionLayer`, `DenseNetModel`.
- `example.py` — trains on real CIFAR-10 (`--device {auto,cpu,cuda,mps}`).
- `example.ipynb` — same walkthrough with loss/accuracy plots.

## Run it

```bash
pip install -e .
python models/densenet/example.py --device auto
# or open models/densenet/example.ipynb
```
