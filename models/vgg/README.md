# VGG

**Paper:** Simonyan, Zisserman, *"Very Deep Convolutional Networks for
Large-Scale Image Recognition"*, 2014 —
[arXiv:1409.1556](https://arxiv.org/abs/1409.1556). See
[`papers/README.md`](../../papers/README.md).

## Idea in one paragraph

Only 3x3 convolutions, stacked, with 2x2 max-pool between stages -- depth
is the only architectural knob, unlike [`models/alexnet`](../alexnet)'s
mixed kernel sizes. This module is the paper's simplest listed
configuration, "config A" / VGG11 (8 conv layers + 3 FC layers), with its
exact per-stage conv counts (1, 1, 2, 2, 2) and channel progression (64,
128, 256, 512, 512). Five stride-2 pooling stages take CIFAR-10's 32x32
input down to exactly 1x1, so the FC classifier here is much smaller
(512 -> 512 -> 10) than the paper's own 4096-wide classifier -- a
CPU-speed-motivated reduction to the classifier only, not to the conv
stack, which keeps VGG11's recipe intact.

## Files

- `model.py` -- `VGG`, the VGG11/config-A conv stack + a smaller classifier.
- `example.py` -- trains on real CIFAR-10 (`--device {auto,cpu,cuda,mps}`).
- `example.ipynb` -- same walkthrough with loss/accuracy plots and sample predictions.

## Run it

```bash
pip install -e .
python models/vgg/example.py --device auto
# or open models/vgg/example.ipynb
```
