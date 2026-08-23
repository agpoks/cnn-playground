"""Train a simplified YOLO-style detector on real Penn-Fudan pedestrians.

    python models/yolo/example.py --device auto --epochs 40

Single-class (pedestrian), single-anchor, single-scale detection. See
model.py for the grid/target-assignment/loss details and papers/README.md
for the reference.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cnn_playground.data import load_penn_fudan_detection, penn_fudan_collate  # noqa: E402
from cnn_playground.device import add_device_arg, resolve_device  # noqa: E402
from cnn_playground.utils.seed import set_seed  # noqa: E402
from model import GRID_SIZE, YOLOModel, build_targets, decode_box, yolo_loss  # noqa: E402


def box_iou(a, b) -> float:
    """a, b: [xmin, ymin, xmax, ymax]."""
    xa1, ya1, xa2, ya2 = a
    xb1, yb1, xb2, yb2 = b
    ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
    ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    add_device_arg(parser)
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    dataset = load_penn_fudan_detection()
    n_train = int(len(dataset) * 0.8)
    train_idx = list(range(n_train))
    test_idx = list(range(n_train, len(dataset)))
    train_ds = torch.utils.data.Subset(dataset, train_idx)
    test_ds = torch.utils.data.Subset(dataset, test_idx)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=penn_fudan_collate
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=penn_fudan_collate
    )

    model = YOLOModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        last_loss = None
        for imgs, boxes_list in train_loader:
            imgs = imgs.to(device)
            targets = torch.stack([build_targets(b) for b in boxes_list]).to(device)
            opt.zero_grad()
            pred = model(imgs)
            loss = yolo_loss(pred, targets)
            loss.backward()
            opt.step()
            last_loss = loss.item()

        if epoch % 5 == 0 or epoch == args.epochs:
            model.eval()
            obj_correct, obj_total = 0, 0
            ious = []
            with torch.no_grad():
                for imgs, boxes_list in test_loader:
                    imgs = imgs.to(device)
                    targets = torch.stack([build_targets(b) for b in boxes_list]).to(device)
                    pred = model(imgs).cpu()
                    targets_cpu = targets.cpu()
                    obj_pred = (pred[:, 0] > 0.5).float()
                    obj_correct += (obj_pred == targets_cpu[:, 0]).sum().item()
                    obj_total += obj_pred.numel()

                    for b in range(pred.shape[0]):
                        flat = pred[b, 0].flatten()
                        best = flat.argmax().item()
                        row, col = best // GRID_SIZE, best % GRID_SIZE
                        pred_box = decode_box(pred[b, :, row, col], row, col)
                        real_boxes = boxes_list[b]
                        if len(real_boxes) == 0:
                            continue
                        best_iou = max(box_iou(pred_box, rb.tolist()) for rb in real_boxes)
                        ious.append(best_iou)
            obj_acc = obj_correct / obj_total
            mean_iou = sum(ious) / len(ious) if ious else 0.0
            print(
                f"epoch {epoch:3d} | train_loss {last_loss:.4f} | "
                f"objectness_acc {obj_acc:.4f} | best-cell IoU {mean_iou:.4f}"
            )
    train_time = time.perf_counter() - t0

    n_params = sum(p.numel() for p in model.parameters())
    print(f"RESULT: model=yolo metric_name=test_iou metric={mean_iou:.4f} params={n_params} train_time_s={train_time:.2f}")


if __name__ == "__main__":
    main()
