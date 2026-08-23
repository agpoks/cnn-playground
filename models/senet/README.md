# SE-Net (Squeeze-and-Excitation)

**Paper:** Hu, Shen, Sun, *"Squeeze-and-Excitation Networks"*, CVPR 2018 —
[arXiv:1709.01507](https://arxiv.org/abs/1709.01507). See
[`papers/README.md`](../../papers/README.md).

## Idea in one paragraph

A regular conv layer treats every output channel as equally important. An
SE block adds a cheap, learned per-channel gate computed from the *whole*
feature map: global-average-pool each channel to one scalar (squeeze),
pass the resulting per-channel descriptor through a small bottleneck MLP
to get per-channel weights in `(0,1)` (excitation), then rescale the
original feature map by those weights. It's a drop-in addition to an
existing backbone, not a new architecture — this repo builds a small
CIFAR-style residual backbone and trains it **with and without** SE
blocks, reporting both accuracies as a direct ablation rather than just
asserting SE helps.

## Files

- `model.py` — `SEBlock` + `ResidualBlock` (with an SE toggle) +
  `SEResNetModel`.
- `example.py` — trains both the plain and SE-augmented backbone on real
  CIFAR-10 (`--device {auto,cpu,cuda,mps}`) and prints both accuracies.
- `example.ipynb` — same walkthrough with loss/accuracy plots for both.

## Run it

```bash
pip install -e .
python models/senet/example.py --device auto
# or open models/senet/example.ipynb
```
