# Papers

For each model in `models/`, the paper picked here is the one that
introduces the architectural idea in its original form. BibTeX for all
thirteen is in [`references.bib`](references.bib).

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
| [U-Net](../models/unet) | U-Net: Convolutional Networks for Biomedical Image Segmentation | MICCAI 2015 | [arXiv:1505.04597](https://arxiv.org/abs/1505.04597) |
| [YOLO-style detector](../models/yolo) | You Only Look Once: Unified, Real-Time Object Detection | CVPR 2016 | [arXiv:1506.02640](https://arxiv.org/abs/1506.02640) |
| [ViT](../models/vit) | An Image is Worth 16x16 Words (Vision Transformer) | ICLR 2021 | [arXiv:2010.11929](https://arxiv.org/abs/2010.11929) |

## Why these thirteen, and the ladder they form

Twelve of these are genuine CNNs, each isolating one specific architectural
idea, roughly in the order the field introduced them:

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
see below) -> **U-Net** (the same convolution/pooling vocabulary,
repurposed as an encoder-decoder with skip connections for *dense*
per-pixel prediction instead of one label per image).

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

The last two are deliberate departures from that ladder, added because the
user asked for "CNN or related to CNN":

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
