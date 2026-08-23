# How the thirteen models differ

Twelve of these are genuine CNNs, each isolating one architectural idea;
the thirteenth (ViT) removes convolution entirely, as a deliberate contrast.

| Model | Core idea | What breaks if you remove it |
|---|---|---|
| [LeNet-5](models/lenet) | conv + pool, stacked a few times | the original recipe -- nothing to remove, it's the baseline |
| [AlexNet](models/alexnet) | much deeper, ReLU instead of tanh/sigmoid, dropout | reverts to LeNet-era depth and vanishing-gradient-prone activations |
| [VGG](models/vgg) | uniform stacks of 3x3 convs, depth as the only knob | loses the "small kernels, more layers" insight -- back to ad hoc kernel sizes |
| [GoogLeNet/Inception](models/googlenet) | multiple kernel sizes in parallel per layer (a "module") | the network must commit to one receptive field per layer instead of blending several |
| [ResNet](models/resnet) | identity skip connections around each block | very deep networks stop training well -- this is *why* ResNet exists |
| [DenseNet](models/densenet) | every layer's output feeds every later layer in its block | far more parameters needed to get the same feature reuse |
| [MobileNet](models/mobilenet) | depthwise-separable convolution (spatial and channel mixing split apart) | back to full/dense convolutions -- far more FLOPs for the same receptive field |
| [SE-Net](models/senet) | a learned per-channel reweighting ("squeeze-and-excite") block | every channel is treated as equally important, regardless of input |
| [ODE-Net](models/odenet) | ResNet's residual step taken to its continuous limit -- depth as ODE integration, not a layer count | back to a fixed discrete number of residual steps (ResNet itself) |
| [Liquid-ODE](models/liquidode) | ODE-Net's dynamics, gated like a Liquid Time-Constant network -- learned, input-dependent relaxation rate | back to ODE-Net's plain (non-gated, fixed-rate) dynamics |
| [U-Net](models/unet) | encoder-decoder with skip connections, one label per *pixel* | without the decoder+skips, you're back to one label per *image* |
| [YOLO-style detector](models/yolo) | predict several boxes + classes in one forward pass | back to a sliding-window/region-proposal pipeline, far slower |
| [ViT](models/vit) | **no convolution at all** -- patches + positional embeddings + self-attention | this *is* the removal -- see below |

## ResNet -> ODE-Net -> Liquid-ODE: making the residual step continuous, then liquid

A three-way progression, all sharing the exact same underlying idea: a
ResNet {doc}`models/resnet` block computes `y = x + F(x)`, one discrete
Euler step of a residual update, applied once per block. ODE-Net
{doc}`models/odenet` observes that this *is* forward-Euler integration of
`dh/dt = f(h(t), t)`, and replaces the entire stack of discrete blocks
with one small conv net `f` integrated continuously from `t=0` to `t=1`
(a hand-rolled fixed-step RK4 solver here, not the paper's adaptive-step
solver + adjoint method -- see `models/odenet/model.py`). Same residual
idea; ResNet takes fixed steps, ODE-Net makes the step size (and
implicitly the "depth") a solver parameter instead.

{doc}`models/liquidode` takes ODE-Net one step further: instead of a plain
conv net for `f(h,t)`, it uses the Liquid Time-Constant governing equation
`dh/dt = -h/tau(h,x) + S(h,x)*(A-h)`, where `tau` (the relaxation rate) and
`S` (a gate) are themselves learned functions of the current state and
input, not fixed. This is **this repo's own combination** of two separate
papers (ODE-Net's continuous depth + Hasani et al. 2021's LTC equation),
not a reproduction of a single published architecture -- see
`models/liquidode/model.py` for the full honesty note. The progression:
ResNet (fixed discrete steps) -> ODE-Net (continuous depth, fixed
dynamics) -> Liquid-ODE (continuous depth, input-dependent/"liquid"
dynamics).

## The one deliberate outlier: ViT

Every other model in this repo is built from `nn.Conv2d`. ViT is built
entirely from `nn.Linear` and attention -- an image is cut into fixed-size
patches, each patch is linearly embedded, a learned position embedding is
added, and the whole sequence goes through a plain Transformer encoder
(the same block used for text). There is no spatial inductive bias
anywhere: the model has to *learn* that nearby pixels are related, instead
of that being baked into a convolution's receptive field the way every
other model here gets for free. Comparing ViT's data efficiency against
any of the CNNs above on the same CIFAR-10 task (see {doc}`benchmarks`) is
a direct, empirical answer to "what does convolution's inductive bias
actually buy you" -- exactly the same framing
`liquid-nn-playground` uses for its RNN/CT-RNN baselines against LTC.

## The real datasets, and who uses them

- **CIFAR-10**: AlexNet, VGG, GoogLeNet/Inception, ResNet, DenseNet,
  MobileNet, SE-Net, ODE-Net, Liquid-ODE, and ViT all train on the
  identical classification task -- ten models, directly comparable.
- **MNIST**: LeNet-5, on the dataset it was originally designed for.
- **Oxford-IIIT Pet** (segmentation masks): U-Net, the one model doing
  dense per-pixel prediction instead of one label per image.
- **Penn-Fudan** (real pedestrian bounding boxes): the YOLO-style
  detector, the one model predicting multiple boxes per image.

See {doc}`benchmarks` for how these clusters are grouped for comparison.
