# Datasets

Four real, standard vision datasets. All auto-download on first use and
cache in `data_cache/` (gitignored) -- no accounts, no manual steps.

## MNIST

LeCun et al.'s original handwritten-digit dataset -- 70,000 `28x28`
grayscale images, 10 classes. Fetched via `torchvision.datasets.MNIST`.
Powers **LeNet-5**, the model it was originally built for.

## CIFAR-10

Krizhevsky's 60,000 `32x32` RGB natural-image dataset, 10 classes. Fetched
via `torchvision.datasets.CIFAR10`. Powers **AlexNet, VGG, GoogLeNet/
Inception, ResNet, DenseNet, MobileNet, SE-Net, and ViT** -- eight
architectures on the identical classification task, directly comparable.

## Oxford-IIIT Pet (segmentation)

37 cat/dog breeds with real pixel-level trimap segmentation masks
(foreground/background/boundary per pixel, collapsed to a binary
foreground mask here). Fetched via
`torchvision.datasets.OxfordIIITPet(target_types="segmentation")`. Powers
**U-Net**.

## Penn-Fudan pedestrians (detection)

170 real photographs (96 near the University of Pennsylvania, 74 near
Fudan University), 345 labeled pedestrians, with real per-instance
segmentation masks that this repo's loader turns into real bounding boxes
(min/max pixel coordinates of each mask instance) -- genuine object-
detection labels, not synthetic ones. Downloaded directly from the
official `cis.upenn.edu` host (the same dataset torchvision's own object-
detection tutorial uses). Powers the **YOLO-style detector**.
