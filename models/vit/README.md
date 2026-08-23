# Vision Transformer (ViT)

**Paper:** Dosovitskiy et al., *"An Image is Worth 16x16 Words: Transformers
for Image Recognition at Scale"*, ICLR 2021 —
[arXiv:2010.11929](https://arxiv.org/abs/2010.11929). See
[`papers/README.md`](../../papers/README.md).

## Idea in one paragraph

Every other model in this repo is built from `nn.Conv2d`. ViT removes
convolution entirely: an image is cut into fixed-size patches, each patch
is linearly embedded, a learned position embedding is added, and the whole
sequence goes through a plain Transformer encoder (hand-written
multi-head self-attention, not `nn.MultiheadAttention`). There is no
spatial inductive bias anywhere — the model has to *learn* that nearby
pixels are related, instead of getting that for free from a convolution's
receptive field. The paper itself says ViT needs much more data than a CNN
to reach comparable accuracy from scratch; trained on CIFAR-10 alone with
no large-scale pretraining, at the epoch budgets used here it is
*expected* to trail the CNNs elsewhere in this repo — that gap is the
finding this baseline is included to show (see
[`../../docs/source/model_comparison.md`](../../docs/source/model_comparison.md)).

## Files

- `model.py` — `image_to_patches`, hand-written `MultiHeadSelfAttention`,
  `EncoderBlock`, `ViTModel`.
- `example.py` — trains on real CIFAR-10 (`--device {auto,cpu,cuda,mps}`).
- `example.ipynb` — same walkthrough, plus a sanity-check visualization of
  the patch grid.

## Run it

```bash
pip install -e .
python models/vit/example.py --device auto
# or open models/vit/example.ipynb
```
