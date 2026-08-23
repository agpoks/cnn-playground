"""Train VGG11 (CIFAR-sized) on real CIFAR-10.

    python models/vgg/example.py --device auto --epochs 20

See model.py for the exact VGG11 configuration this reproduces (same
conv-count/channel progression as the paper's "config A"). See
papers/README.md.
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
from model import VGG  # noqa: E402


def evaluate(model, loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.shape[0]
    return correct / total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    add_device_arg(parser)
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    train_ds = load_cifar10(train=True)
    test_ds = load_cifar10(train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    model = VGG(num_classes=10).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            running_loss += loss.item() * x.shape[0]
        train_loss = running_loss / len(train_ds)

        if epoch % 2 == 0 or epoch == args.epochs:
            test_acc = evaluate(model, test_loader, device)
            print(f"epoch {epoch:3d} | train_loss {train_loss:.4f} | test_acc {test_acc:.3f}")
    train_time = time.perf_counter() - t0

    test_acc = evaluate(model, test_loader, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"RESULT: model=vgg metric_name=test_acc metric={test_acc:.4f} params={n_params} train_time_s={train_time:.2f}")


if __name__ == "__main__":
    main()
