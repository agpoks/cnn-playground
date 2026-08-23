"""Real dataset loaders shared across every model in this repo.

Unlike sciml-playground, every dataset here is a standard, real, publicly
released vision benchmark with a maintained `torchvision` loader (MNIST,
CIFAR-10, Oxford-IIIT Pet) or one small, stable, directly-downloadable real
dataset (Penn-Fudan) -- no custom scraping/API work needed, torchvision
handles the auto-download and caching itself.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import requests
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

CACHE_DIR = Path(__file__).resolve().parents[2] / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)

PENN_FUDAN_URL = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"


def load_mnist(train: bool = True):
    """Real MNIST digits (LeCun et al.) -- (1, 28, 28) grayscale, 10 classes.
    Auto-downloads via torchvision on first call."""
    tfm = transforms.Compose([transforms.ToTensor()])
    return torchvision.datasets.MNIST(root=str(CACHE_DIR), train=train, download=True, transform=tfm)


def load_cifar10(train: bool = True):
    """Real CIFAR-10 (Krizhevsky) -- (3, 32, 32) RGB, 10 classes. Auto-
    downloads via torchvision on first call."""
    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    return torchvision.datasets.CIFAR10(root=str(CACHE_DIR), train=train, download=True, transform=tfm)


def load_oxford_pet_segmentation(train: bool = True, image_size: int = 128):
    """Real Oxford-IIIT Pet dataset with pixel-level trimap segmentation
    masks (37 breeds, background/foreground/boundary per pixel -> we
    collapse to a binary foreground mask for a simple binary-segmentation
    task). Auto-downloads via torchvision on first call. Returns a
    torch.utils.data.Dataset yielding (image (3,H,W), mask (1,H,W))."""
    split = "trainval" if train else "test"
    img_tfm = transforms.Compose(
        [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
    )
    mask_tfm = transforms.Compose(
        [transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST)]
    )
    base = torchvision.datasets.OxfordIIITPet(
        root=str(CACHE_DIR),
        split=split,
        target_types="segmentation",
        download=True,
    )

    class _Wrapped(Dataset):
        def __len__(self):
            return len(base)

        def __getitem__(self, idx):
            img, trimap = base[idx]
            img = img_tfm(img)
            trimap = mask_tfm(trimap)
            trimap = torch.from_numpy(np.array(trimap)).long()
            # trimap labels: 1=foreground(pet), 2=background, 3=boundary -- collapse to binary
            mask = (trimap == 1).float().unsqueeze(0)
            return img, mask

    return _Wrapped()


def _ensure_penn_fudan() -> Path:
    extract_dir = CACHE_DIR / "PennFudanPed"
    if extract_dir.exists() and (extract_dir / "PNGImages").exists():
        return extract_dir
    zip_path = CACHE_DIR / "PennFudanPed.zip"
    if not zip_path.exists():
        resp = requests.get(PENN_FUDAN_URL, timeout=120)
        resp.raise_for_status()
        zip_path.write_bytes(resp.content)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(CACHE_DIR)
    return extract_dir


class PennFudanDetection(Dataset):
    """Real Penn-Fudan pedestrian dataset (170 images, 345 labeled
    pedestrians, University of Pennsylvania + Fudan University). Ships
    per-instance segmentation masks (PNGs where each pedestrian is a
    distinct integer label); bounding boxes are derived from those masks
    (min/max coordinates of each label's pixels) -- real box-level object
    detection labels, not synthetic ones. One image can have multiple
    pedestrians. All boxes are a single class ("pedestrian")."""

    def __init__(self, image_size: int = 224):
        self.root = _ensure_penn_fudan()
        self.image_size = image_size
        self.images = sorted((self.root / "PNGImages").iterdir())
        self.masks = sorted((self.root / "PedMasks").iterdir())
        assert len(self.images) == len(self.masks)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        mask = np.array(Image.open(self.masks[idx]))
        orig_w, orig_h = img.size

        obj_ids = np.unique(mask)
        obj_ids = obj_ids[obj_ids != 0]  # 0 = background
        boxes = []
        for obj_id in obj_ids:
            ys, xs = np.where(mask == obj_id)
            if len(xs) == 0:
                continue
            boxes.append([xs.min(), ys.min(), xs.max(), ys.max()])
        boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)

        # resize image and boxes together to a fixed square size
        sx, sy = self.image_size / orig_w, self.image_size / orig_h
        img = img.resize((self.image_size, self.image_size))
        if len(boxes):
            boxes[:, [0, 2]] *= sx
            boxes[:, [1, 3]] *= sy

        img_t = transforms.ToTensor()(img)
        return img_t, torch.from_numpy(boxes)


def penn_fudan_collate(batch):
    """Detection batches have a variable number of boxes per image, so they
    can't be stacked into one tensor -- return (images: (B,3,H,W) stacked,
    boxes: list of (n_i, 4) tensors, one per image)."""
    images = torch.stack([b[0] for b in batch], dim=0)
    boxes = [b[1] for b in batch]
    return images, boxes


def load_penn_fudan_detection(image_size: int = 224):
    return PennFudanDetection(image_size=image_size)
