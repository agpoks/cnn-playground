"""Train an SE-ResNet on real CIFAR-10, alongside the identical backbone
with no SE blocks, as a direct ablation.

    python models/senet/example.py --device auto --epochs 20

See model.py for the Squeeze-and-Excitation block and papers/README.md for
the reference.
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
from model import SEResNetModel  # noqa: E402


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


def train_one(model, train_loader, test_loader, device, epochs, lr):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        last_loss = None
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            loss = loss_fn(model(imgs), labels)
            loss.backward()
            opt.step()
            last_loss = loss.item()
        if epoch % 5 == 0 or epoch == epochs:
            test_acc = evaluate(model, test_loader, device)
            print(f"  epoch {epoch:3d} | train_loss {last_loss:.4f} | test_acc {test_acc:.4f}")
    train_time = time.perf_counter() - t0
    final_acc = evaluate(model, test_loader, device)
    return final_acc, train_time


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

    print("--- plain backbone (no SE blocks) ---")
    set_seed(args.seed)
    plain_model = SEResNetModel(use_se=False).to(device)
    plain_acc, _ = train_one(plain_model, train_loader, test_loader, device, args.epochs, args.lr)
    plain_params = sum(p.numel() for p in plain_model.parameters())

    print("--- SE-augmented backbone ---")
    set_seed(args.seed)
    se_model = SEResNetModel(use_se=True).to(device)
    se_acc, train_time = train_one(se_model, train_loader, test_loader, device, args.epochs, args.lr)
    se_params = sum(p.numel() for p in se_model.parameters())

    print(
        f"ablation: plain test_acc={plain_acc:.4f} (params={plain_params})  "
        f"vs.  SE test_acc={se_acc:.4f} (params={se_params})"
    )
    print(f"RESULT: model=senet metric_name=test_acc metric={se_acc:.4f} params={se_params} train_time_s={train_time:.2f}")


if __name__ == "__main__":
    main()
