# YOLO-style detector

**Paper:** Redmon, Divvala, Girshick, Farhadi, *"You Only Look Once:
Unified, Real-Time Object Detection"*, CVPR 2016 —
[arXiv:1506.02640](https://arxiv.org/abs/1506.02640). See
[`papers/README.md`](../../papers/README.md).

## Idea in one paragraph

Every other model in this repo predicts one label for the whole image.
YOLO divides the image into an `SxS` grid (here `7x7`) and has every grid
cell directly regress "is an object centered here, and if so, where
exactly and how big" — one single forward pass, no region proposals, no
sliding window. This is the generalization from classification to
*detection*.

**Simplifications** (stated explicitly — the full YOLO v1 predicts 2
anchor boxes per cell and per-class probabilities across many classes):
single anchor box per cell, single class (Penn-Fudan is "pedestrian" only,
so no class head is needed — just objectness + box geometry), and plain
MSE on width/height instead of the paper's `sqrt(w), sqrt(h)` target
(their trick for weighting small and large boxes equally in the loss).
What's kept from the paper: the core `SxS` grid + single-pass regression
idea, and its Eq. 3 loss structure — BCE on objectness across *every*
cell, but coordinate loss only on cells with a real box assigned, with the
paper's `lambda_coord=5` / `lambda_noobj=0.5` reweighting (without it, the
far more numerous empty cells dominate the loss and objectness collapses
to zero).

## Files

- `model.py` — `YOLOModel` (conv backbone + 1x1 detection head),
  `build_targets`/`decode_box` (grid target assignment and box decoding,
  verified against hand-computed examples), `yolo_loss` (the paper's
  Eq. 3 loss, simplified as above).
- `example.py` — trains on real Penn-Fudan pedestrian boxes
  (`--device {auto,cpu,cuda,mps}`).
- `example.ipynb` — same walkthrough, plus real image/box plots.

## Run it

```bash
pip install -e .
python models/yolo/example.py --device auto
# or open models/yolo/example.ipynb
```
