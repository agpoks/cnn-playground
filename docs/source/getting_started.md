# Getting started

## What `nn.Linear`/`nn.Conv2d` actually compute

Every model in this repo is hand-written from scratch: the architectural
ideas themselves (residual connections, dense connections, depthwise-
separable convolutions, channel attention, patch-based attention) are all
literal Python in `models/*/model.py`, not calls into a pre-built
`torchvision.models` architecture. The building blocks every model *does*
reuse are `nn.Conv2d` and `nn.Linear`. `nn.Linear` computes
$y = xW^\top + b$ (see
[`liquid-nn-playground`'s getting-started page](https://github.com/agpoks/liquid-nn-playground/blob/main/docs/source/getting_started.md)
for a from-scratch reimplementation). `nn.Conv2d` slides a small learned
kernel spatially across the input and computes a weighted sum at every
position -- the same affine-map idea as `nn.Linear`, just applied locally
and shared across positions instead of once globally.

## Install

```bash
git clone https://github.com/agpoks/cnn-playground.git
cd cnn-playground
pip install -e ".[notebooks]"
```

## Run one model

```bash
python models/lenet/example.py --device auto
```

Every example script accepts `--device {auto,cpu,cuda,mps}`. Every model
also has a matching `example.ipynb` in the same folder.

## Datasets

Four real datasets auto-download on first use via `torchvision` or one
direct download -- see {doc}`datasets` for what's available and why each
was picked.
