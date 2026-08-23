"""Train U-Net on real Oxford-IIIT Pet segmentation masks.

    python models/unet/example.py --device auto --epochs 30

Binary foreground/background segmentation (pet vs. not-pet), collapsed
from the dataset's 3-way trimap. See model.py for the encoder-decoder
architecture and papers/README.md for the reference.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cnn_playground.data import load_oxford_pet_segmentation  # noqa: E402
from cnn_playground.device import add_device_arg, resolve_device  # noqa: E402
from cnn_playground.utils.seed import set_seed  # noqa: E402
from model import UNet  # noqa: E402


def iou(pred_mask: torch.Tensor, true_mask: torch.Tensor) -> float:
    """Mean IoU over a batch. pred_mask, true_mask: (B, 1, H, W) in {0, 1}."""
    intersection = (pred_mask * true_mask).sum(dim=(1, 2, 3))
    union = ((pred_mask + true_mask) > 0).float().sum(dim=(1, 2, 3))
    return (intersection / union.clamp_min(1e-8)).mean().item()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    add_device_arg(parser)
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    train_ds = load_oxford_pet_segmentation(train=True)
    test_ds = load_oxford_pet_segmentation(train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    model = UNet(base_channels=args.base_channels).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        last_loss = None
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            opt.zero_grad()
            logits = model(imgs)
            loss = loss_fn(logits, masks)
            loss.backward()
            opt.step()
            last_loss = loss.item()

        if epoch % 5 == 0 or epoch == args.epochs:
            model.eval()
            ious = []
            with torch.no_grad():
                for imgs, masks in test_loader:
                    imgs, masks = imgs.to(device), masks.to(device)
                    pred = (torch.sigmoid(model(imgs)) > 0.5).float()
                    ious.append(iou(pred, masks))
            test_iou = sum(ious) / len(ious)
            print(f"epoch {epoch:3d} | train_loss {last_loss:.4f} | test_iou {test_iou:.4f}")
    train_time = time.perf_counter() - t0

    n_params = sum(p.numel() for p in model.parameters())
    print(f"RESULT: model=unet metric_name=test_iou metric={test_iou:.4f} params={n_params} train_time_s={train_time:.2f}")


if __name__ == "__main__":
    main()
