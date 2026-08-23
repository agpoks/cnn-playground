# AlexNet

**Paper:** Krizhevsky, Sutskever, Hinton, *"ImageNet Classification with
Deep Convolutional Neural Networks"*, NeurIPS 2012. See
[`papers/README.md`](../../papers/README.md).

## Idea in one paragraph

Much deeper than [`models/lenet`](../lenet), ReLU instead of tanh, dropout
in the FC layers -- the deep-learning-era CNN recipe. The original network
was built for 224x224 ImageNet input with an aggressive stride-4 first
conv; applied directly to CIFAR-10's 32x32 images that would collapse the
spatial size almost immediately, so this module keeps the paper's exact
per-layer channel progression (64 -> 192 -> 384 -> 256 -> 256, the same
4096-wide FC stage) but re-tunes kernel sizes/strides for 32x32 input --
the same adaptation the ResNet paper itself makes for its own CIFAR-10
experiments, stated explicitly rather than silently swapped in.

## Files

- `model.py` -- `AlexNet`, CIFAR-adapted per the note above.
- `example.py` -- trains on real CIFAR-10 (`--device {auto,cpu,cuda,mps}`).
- `example.ipynb` -- same walkthrough with loss/accuracy plots and sample predictions.

## Run it

```bash
pip install -e .
python models/alexnet/example.py --device auto
# or open models/alexnet/example.ipynb
```
