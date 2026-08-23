"""A simplified, single-scale, single-anchor, single-class YOLO-style
detector.

Reference: Redmon, Divvala, Girshick, Farhadi, "You Only Look Once:
Unified, Real-Time Object Detection", CVPR 2016. arXiv:1506.02640. See
papers/README.md (bibtex key `redmon2016yolo`).

Every other model in this repo predicts one label for the whole image.
YOLO's idea: divide the image into an SxS grid and have every grid cell
directly regress "is there an object centered here, and if so, where
exactly and how big" -- one single forward pass over the whole image,
no region proposals, no sliding window. This is the generalization from
classification to *detection*: multiple objects, multiple locations, one
network evaluation.

Simplifications vs. the paper (stated explicitly, since the full YOLO v1
predicts multiple anchor boxes and per-cell class probabilities, and is
trained end-to-end on multi-class detection):
  - single anchor box per cell, not 2 (no anchor-selection logic needed)
  - single class ("pedestrian", from Penn-Fudan) -- no class-probability
    head, just objectness + box geometry
  - box width/height are parameterized as sigmoid(raw) in [0, 1] (fraction
    of the whole image), not the paper's sqrt(w), sqrt(h) target -- this
    repo's loss uses plain MSE on w, h directly rather than their square
    roots, which is a minor simplification of how the paper weights small
    vs. large boxes equally in the loss (their Eq. 3 uses sqrt terms
    specifically to fix this; we skip that refinement)

The loss structure that IS kept from the paper (its Eq. 3): binary
cross-entropy on objectness across *all* SxS cells, but coordinate
regression loss ONLY on cells that actually have a real box assigned to
them, with the paper's characteristic lambda_coord=5 (up-weight box
regression) and lambda_noobj=0.5 (down-weight the vastly more numerous
empty cells) -- without that reweighting, the many empty cells dominate
the loss and objectness collapses to always-zero.
"""

from __future__ import annotations

import torch
import torch.nn as nn

GRID_SIZE = 7
IMAGE_SIZE = 224
CELL_SIZE = IMAGE_SIZE / GRID_SIZE  # 32.0


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class YOLOModel(nn.Module):
    """(B, 3, 224, 224) -> (B, 5, 7, 7): [objectness, x_off, y_off, w, h] per cell."""

    def __init__(self):
        super().__init__()
        # 224 -> 112 -> 56 -> 28 -> 14 -> 7 (five 2x poolings)
        self.backbone = nn.Sequential(
            ConvBlock(3, 16),
            ConvBlock(16, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )
        self.head = nn.Conv2d(256, 5, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)  # (B, 256, 7, 7)
        raw = self.head(feats)  # (B, 5, 7, 7)
        obj = torch.sigmoid(raw[:, 0:1])
        xy = torch.sigmoid(raw[:, 1:3])  # offset within cell, in [0, 1)
        wh = torch.sigmoid(raw[:, 3:5])  # box size, fraction of whole image, in [0, 1]
        return torch.cat([obj, xy, wh], dim=1)


def build_targets(boxes: torch.Tensor, grid_size: int = GRID_SIZE, image_size: int = IMAGE_SIZE) -> torch.Tensor:
    """boxes: (n, 4) real [xmin, ymin, xmax, ymax] pixel boxes for ONE image.
    Returns (5, grid_size, grid_size): [objectness, x_off, y_off, w, h]
    target grid, matching the model's output layout. If two real boxes land
    in the same cell, the later one in `boxes` overwrites the earlier
    (rare with Penn-Fudan's sparse, well-separated pedestrians; a documented
    simplification rather than handling multi-object-per-cell)."""
    cell = image_size / grid_size
    target = torch.zeros(5, grid_size, grid_size)
    for box in boxes:
        xmin, ymin, xmax, ymax = box.tolist()
        cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
        col = min(int(cx // cell), grid_size - 1)
        row = min(int(cy // cell), grid_size - 1)
        x_off = (cx - col * cell) / cell
        y_off = (cy - row * cell) / cell
        w = (xmax - xmin) / image_size
        h = (ymax - ymin) / image_size
        target[0, row, col] = 1.0
        target[1, row, col] = x_off
        target[2, row, col] = y_off
        target[3, row, col] = w
        target[4, row, col] = h
    return target


def decode_box(pred_cell: torch.Tensor, row: int, col: int, grid_size: int = GRID_SIZE, image_size: int = IMAGE_SIZE):
    """pred_cell: (5,) = [objectness, x_off, y_off, w, h] at grid (row, col).
    Returns [xmin, ymin, xmax, ymax] in pixel coordinates."""
    cell = image_size / grid_size
    _obj, x_off, y_off, w, h = pred_cell.tolist()
    cx = (col + x_off) * cell
    cy = (row + y_off) * cell
    bw, bh = w * image_size, h * image_size
    return [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]


def yolo_loss(pred: torch.Tensor, target: torch.Tensor, lambda_coord: float = 5.0, lambda_noobj: float = 0.5):
    """pred, target: (B, 5, S, S). Returns a scalar loss (paper Eq. 3,
    simplified: no sqrt(w)/sqrt(h), single anchor, single class)."""
    obj_mask = target[:, 0:1]  # (B, 1, S, S), 1 where a real box is assigned
    noobj_mask = 1.0 - obj_mask

    obj_loss = nn.functional.binary_cross_entropy(pred[:, 0:1] * obj_mask, target[:, 0:1] * obj_mask, reduction="sum")
    noobj_loss = nn.functional.binary_cross_entropy(
        pred[:, 0:1] * noobj_mask, target[:, 0:1] * noobj_mask, reduction="sum"
    )
    coord_loss = (obj_mask * (pred[:, 1:5] - target[:, 1:5]) ** 2).sum()

    batch = pred.shape[0]
    return (lambda_coord * coord_loss + obj_loss + lambda_noobj * noobj_loss) / batch
