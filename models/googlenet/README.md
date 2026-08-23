# GoogLeNet / Inception

**Paper:** Szegedy, Liu, Jia, Sermanet, Reed, Anguelov, Erhan, Vanhoucke,
Rabinovich, *"Going Deeper with Convolutions"*, CVPR 2015 —
[arXiv:1409.4842](https://arxiv.org/abs/1409.4842). See
[`papers/README.md`](../../papers/README.md).

## Idea in one paragraph

Every other model in this repo commits one kernel size per layer.
GoogLeNet's Inception module instead runs several branches with
*different* receptive fields (1x1, 3x3, 5x5, plus a pooling branch) over
the *same* input in parallel, then concatenates their outputs — the
network blends fine and coarse spatial detail at every stage instead of
picking one. A 1x1 "bottleneck" conv reduces channels before the 3x3/5x5
branches (the paper's dimension-reduced module, Fig. 2b), keeping their
cost from scaling with the (potentially large) input channel count.

**CIFAR-10 adaptation:** the paper's 224x224-input stem and 9-module,
22-layer depth would leave almost nothing of a 32x32 image. This version
uses a gentler 2-conv stem and only 4 Inception modules, with the paper's
auxiliary classifiers (a training aid for its much deeper original)
dropped entirely — see `model.py` for details.

## Files

- `model.py` — `InceptionModule` (4-branch, dimension-reduced) + `GoogLeNetModel`.
- `example.py` — trains on real CIFAR-10 (`--device {auto,cpu,cuda,mps}`).
- `example.ipynb` — same walkthrough with loss/accuracy plots.

## Run it

```bash
pip install -e .
python models/googlenet/example.py --device auto
# or open models/googlenet/example.ipynb
```
