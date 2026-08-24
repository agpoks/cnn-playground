"""Train OBF-Conv (a conv net whose kernel shape is constrained to a
Kautz/Laguerre orthonormal-basis-function subspace) on real CIFAR-10.

    python models/obfconv/example.py --device auto --epochs 20
    python models/obfconv/example.py --device auto --basis kautz

See model.py for OBFConv2d/generate_laguerre_basis/generate_kautz_basis
and papers/README.md for the (system-identification, not CNN) references
this repo's own combination draws on.
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

from cnn_playground.data import load_cifar10  # noqa: E402
from cnn_playground.device import add_device_arg, resolve_device  # noqa: E402
from cnn_playground.utils.seed import set_seed  # noqa: E402
from model import OBFConvModel  # noqa: E402


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.numel()
    return correct / total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--basis", choices=["laguerre", "kautz"], default="laguerre")
    parser.add_argument("--n-basis", type=int, default=4, help="number of 1D OBFs (kernel spans n_basis**2 fixed 2D filters)")
    parser.add_argument("--seed", type=int, default=0)
    add_device_arg(parser)
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}, basis: {args.basis}")

    train_ds = load_cifar10(train=True)
    test_ds = load_cifar10(train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    model = OBFConvModel(basis=args.basis, n_basis=args.n_basis).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        last_loss = None
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            loss = loss_fn(model(imgs), labels)
            loss.backward()
            opt.step()
            last_loss = loss.item()

        if epoch % 5 == 0 or epoch == args.epochs:
            test_acc = evaluate(model, test_loader, device)
            print(f"epoch {epoch:3d} | train_loss {last_loss:.4f} | test_acc {test_acc:.4f}")
    train_time = time.perf_counter() - t0

    n_params = sum(p.numel() for p in model.parameters())
    final_acc = evaluate(model, test_loader, device)
    print(f"RESULT: model=obfconv metric_name=test_acc metric={final_acc:.4f} params={n_params} train_time_s={train_time:.2f} basis={args.basis}")


if __name__ == "__main__":
    main()
