# Papers

For each model in `models/`, the paper picked here is the one that
introduces the architectural idea in its original form. BibTeX for all
eighteen references (covering sixteen models -- Liquid-ODE,
Legendre-KAN-Conv, and OBF-Conv each cite two papers) is in
[`references.bib`](references.bib).

| Model | Paper | Year | Link |
|---|---|---|---|
| [LeNet-5](../models/lenet) | Gradient-Based Learning Applied to Document Recognition | 1998 | classic reference, no arXiv preprint |
| [AlexNet](../models/alexnet) | ImageNet Classification with Deep Convolutional Neural Networks | NeurIPS 2012 | classic reference, no arXiv preprint |
| [VGG](../models/vgg) | Very Deep Convolutional Networks for Large-Scale Image Recognition | 2014 | [arXiv:1409.1556](https://arxiv.org/abs/1409.1556) |
| [GoogLeNet/Inception](../models/googlenet) | Going Deeper with Convolutions | CVPR 2015 | [arXiv:1409.4842](https://arxiv.org/abs/1409.4842) |
| [ResNet](../models/resnet) | Deep Residual Learning for Image Recognition | CVPR 2016 | [arXiv:1512.03385](https://arxiv.org/abs/1512.03385) |
| [DenseNet](../models/densenet) | Densely Connected Convolutional Networks | CVPR 2017 | [arXiv:1608.06993](https://arxiv.org/abs/1608.06993) |
| [MobileNet](../models/mobilenet) | MobileNets: Efficient CNNs for Mobile Vision Applications | 2017 | [arXiv:1704.04861](https://arxiv.org/abs/1704.04861) |
| [SE-Net](../models/senet) | Squeeze-and-Excitation Networks | CVPR 2018 | [arXiv:1709.01507](https://arxiv.org/abs/1709.01507) |
| [ODE-Net](../models/odenet) | Neural Ordinary Differential Equations | NeurIPS 2018 | [arXiv:1806.07366](https://arxiv.org/abs/1806.07366) |
| [Liquid-ODE](../models/liquidode) | *(this repo's own combination, not one paper)* -- Neural ODEs + Liquid Time-constant Networks | 2018 / 2021 | [1806.07366](https://arxiv.org/abs/1806.07366), [2006.04439](https://arxiv.org/abs/2006.04439) |
| [Legendre-KAN-Conv](../models/legendrekan) | *(follows a community pattern, not one paper)* -- KAN + Convolutional KANs | 2024 | [2404.19756](https://arxiv.org/abs/2404.19756), [2406.13155](https://arxiv.org/abs/2406.13155) |
| [OBF-Conv](../models/obfconv) | *(this repo's own combination, not one paper)* -- Kautz/Laguerre orthonormal basis functions from system identification | 1991 / 2011 | classic reference, no arXiv preprint; IJMIC 2011 |
| [NCA](../models/nca) | Growing Neural Cellular Automata: Differentiable Model of Morphogenesis | Distill 2020 | [distill.pub/2020/growing-ca](https://distill.pub/2020/growing-ca/) |
| [U-Net](../models/unet) | U-Net: Convolutional Networks for Biomedical Image Segmentation | MICCAI 2015 | [arXiv:1505.04597](https://arxiv.org/abs/1505.04597) |
| [YOLO-style detector](../models/yolo) | You Only Look Once: Unified, Real-Time Object Detection | CVPR 2016 | [arXiv:1506.02640](https://arxiv.org/abs/1506.02640) |
| [ViT](../models/vit) | An Image is Worth 16x16 Words (Vision Transformer) | ICLR 2021 | [arXiv:2010.11929](https://arxiv.org/abs/2010.11929) |

## Why these sixteen, and the ladder they form

Fifteen of these are genuine CNNs (in the sense of being built from
`nn.Conv2d`), each isolating one specific architectural idea, roughly in
the order the field introduced them:

**LeNet** (the original, small CNN) -> **AlexNet** (much deeper + ReLU +
dropout, the deep-learning-era CNN) -> **VGG** (uniform stacks of small 3x3
convs, depth as the only variable) -> **GoogLeNet/Inception** (multi-branch
modules, several kernel sizes in parallel per layer) -> **ResNet** (skip/
identity connections, solving vanishing gradients at real depth) ->
**DenseNet** (every layer connects to every later layer, maximal feature
reuse) -> **MobileNet** (depthwise-separable convolutions, an efficiency
axis orthogonal to depth) -> **SE-Net** (channel attention as a drop-in
block, added on top of any of the above) -> **ODE-Net** (ResNet's residual
block taken to its continuous limit: depth becomes an ODE integration
variable instead of a discrete layer count, see below) -> **Liquid-ODE**
(ODE-Net's own dynamics made "liquid": the same continuous-depth
integration, but `dh/dt` is now a Liquid-Time-Constant-gated equation with
a learned, input-dependent time constant, instead of a plain conv net --
see below) -> **Legendre-KAN-Conv** (a different axis entirely: instead of
one learned scalar weight per kernel tap, each tap's contribution is a
learned Legendre-polynomial function of the input value there -- a
structured/basis-function kernel parameterization instead of a free
weight, see below) -> **OBF-Conv** (a different structured-kernel-basis
choice again: the kernel's *spatial shape* -- not the function applied to
the input value, as in Legendre-KAN-Conv -- is constrained to the span of
a few fixed Kautz/Laguerre basis filters borrowed from system
identification, see below) -> **U-Net** (the same convolution/pooling vocabulary,
repurposed as an encoder-decoder with skip connections for *dense*
per-pixel prediction instead of one label per image) -> **NCA** (the same
convolution primitive, repurposed again: not one forward pass at all, but
a local update rule iterated for many stochastic steps, see below).

ODE-Net is worth contrasting directly with ResNet: a ResNet block computes
`y = x + F(x)`, one discrete Euler step of a residual update, repeated once
per block. ODE-Net's insight is that this *is* forward-Euler integration of
`dh/dt = f(h(t), t)` -- so instead of stacking N discrete blocks, one small
conv net `f` is integrated continuously from `t=0` to `t=1` by an ODE
solver (a hand-rolled fixed-step RK4 here, see `models/odenet/model.py`).
Same residual idea, continuous instead of discrete depth.

Liquid-ODE takes that one step further, and is **not itself a published
paper** -- it's this repo's own assembly of two real ideas (ODE-Net's
continuous depth + Liquid Time-constant Networks' governing equation,
Hasani et al. 2021), stated explicitly as such in `model.py`. Where
ODE-Net's `f(h,t)` is a plain conv net, Liquid-ODE's dynamics are
`dh/dt = -h/tau(h,x) + S(h,x)*(A-h)`: `tau` (how fast the state relaxes)
and `S` (a gate toward a learned rest state `A`) are themselves small conv
nets of the current state and input, so the relaxation rate itself is
input-dependent -- see `models/liquidode/model.py` for the full honesty
note on what is and isn't reproduced from either source paper.

Legendre-KAN-Conv is a different kind of idea from all of the above: every
other CNN here learns one scalar weight per kernel tap, full stop.
Kolmogorov-Arnold Networks (Liu et al. 2024) replace that with a learned
*univariate function* per edge, originally a B-spline; ConvKAN (Bodner
et al. 2024) puts that inside a convolution. This repo's variant uses a
degree-K Legendre-polynomial expansion of the (`tanh`-squashed) input as
that per-tap function -- a smooth global polynomial basis via a
three-term recurrence, cheaper than a B-spline's piecewise machinery. It
belongs to the same "structured/basis-function kernel" family as
Structured Receptive Fields (Gaussian-derivative basis), Gabor CNNs, and
Spherical CNNs (spherical-harmonic/associated-Legendre-function basis) in
the wider literature, though none of those are reproduced here -- see
`models/legendrekan/model.py` for the full honesty note, including that
the specific Legendre-basis-ConvKAN combination follows a community
implementation pattern rather than one dedicated paper.

OBF-Conv is **also not a single published paper**, and constrains a
*different* dimension of the convolution than Legendre-KAN-Conv does.
Kautz and Laguerre orthonormal basis functions (Wahlberg 1991; Oliveira
et al. 2011) are real, established tools -- but exclusively in linear
system identification, for representing an impulse response compactly as
a short combination of fixed basis sequences given a decay (Laguerre) or
resonance (Kautz) prior. This repo transplants that onto a conv kernel's
*spatial shape*: the kernel is constrained to the span of a small number
of fixed, orthonormal Kautz- or Laguerre-generated 2D filters, and only
the combination coefficients are learned. Where Legendre-KAN-Conv bases
its expansion on the *pixel value* at a tap, OBF-Conv bases it on the
*tap/spatial-index* itself -- the kernel's shape, not the function applied
under it. As far as could be found by searching the literature while
building this repo, no CNN-kernel paper does this -- see
`models/obfconv/model.py` for the full honesty note and the real DSP
recursions used to generate the basis.

NCA is worth contrasting against everything above it: every other model in
this repo is a single forward pass (or, for ODE-Net/Liquid-ODE, a single
continuous integration) -- image in, one prediction out. NCA's trained
object is a *local update rule*, the same few conv layers applied
identically and independently at every cell of a grid, for dozens of
stochastic asynchronous steps, with a living-cell mask keeping an
undefined background at exactly zero. A single seed cell, iterated under
this rule, self-organizes into a target pattern -- trained by unrolling
the rule through time and comparing the final state to the target, not by
comparing one prediction to one label. It uses the paper's real mechanism
(fixed Sobel/identity perception "eyes," a tiny learned update net,
stochastic firing, alive-masking) but trains against a procedurally
generated target pattern instead of the paper's Google Noto emoji, and
skips the paper's "sample pool" trick that additionally teaches
persistence and damage-regeneration -- both simplifications are stated
explicitly in `models/nca/model.py`.

The last two (YOLO-style detector, ViT) are deliberate departures from
that ladder, added because the user asked for "CNN or related to CNN":

- **YOLO-style detector** -- the same convolutional feature extractor, but
  predicting *multiple* bounding boxes + classes per image in one forward
  pass, the task-level generalization from classification to detection.
- **Vision Transformer (ViT)** -- a deliberate **non-convolutional**
  contrast baseline: no convolution anywhere, an image is cut into patches
  and fed through a plain Transformer encoder. Included specifically to
  answer "what does the inductive bias of convolution actually buy you,"
  the same way `liquid-nn-playground`'s RNN/CT-RNN baselines isolate what
  "liquid" buys over a plain RNN.

See [`docs/source/model_comparison.md`](../docs/source/model_comparison.md) for the full comparison.

## Real datasets

Four real, standard vision datasets, all with maintained `torchvision`
loaders or one small direct download -- see
[`datasets/README.md`](../datasets/README.md).
