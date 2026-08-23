# MobileNet

**Paper:** Howard, Zhu, Chen, Kalenichenko, Wang, Weyand, Andreetto, Adam,
*"MobileNets: Efficient Convolutional Neural Networks for Mobile Vision
Applications"*, 2017 — [arXiv:1704.04861](https://arxiv.org/abs/1704.04861).
See [`papers/README.md`](../../papers/README.md).

## Idea in one paragraph

A regular `Conv3x3` mixes spatial and cross-channel information in one
operation. MobileNet factors that into two cheaper steps: a **depthwise**
conv (one 3x3 filter per input channel, spatial mixing only) followed by a
**pointwise** `1x1` conv (channel mixing only). For similar receptive
field and depth, this costs roughly an order of magnitude fewer
multiply-adds than a regular conv stack. `example.py` builds the identical
architecture with regular convs instead and prints both parameter counts
side by side — in this repo's configuration, the depthwise-separable
version has about **8x fewer parameters**.

## Files

- `model.py` — `DepthwiseSeparableConv` + `MobileNetModel`, plus
  `PlainConvComparisonModel` (identical depth/channels, regular convs, for
  the parameter-count comparison only).
- `example.py` — trains on real CIFAR-10 (`--device {auto,cpu,cuda,mps}`),
  prints the parameter-count comparison before training.
- `example.ipynb` — same walkthrough with loss/accuracy plots.

## Run it

```bash
pip install -e .
python models/mobilenet/example.py --device auto
# or open models/mobilenet/example.ipynb
```
