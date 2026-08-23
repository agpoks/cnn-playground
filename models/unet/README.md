# U-Net

**Paper:** Ronneberger, Fischer, Brox, *"U-Net: Convolutional Networks for
Biomedical Image Segmentation"*, MICCAI 2015 —
[arXiv:1505.04597](https://arxiv.org/abs/1505.04597). See
[`papers/README.md`](../../papers/README.md).

## Idea in one paragraph

Every other model in this repo reduces an image to one label. U-Net keeps
the same conv/pool vocabulary but repurposes it for *dense* prediction —
one label per pixel. A contracting (encoder) path downsamples and deepens
the feature maps like an ordinary CNN classifier; a symmetric expanding
(decoder) path upsamples back to the original resolution. The key idea is
the **skip connection**: at each decoder stage, the same-resolution
encoder feature map is concatenated in before convolving further, so the
decoder gets both the encoder's deep semantic features *and* the fine
spatial detail pooling would otherwise have discarded.

**Simplification:** this implementation uses `padding=1` ("same")
convolutions throughout instead of the paper's unpadded ("valid") ones —
output matches input resolution exactly with no cropping needed at the
skip connections — and a smaller channel count (32→64→128→256, bottleneck
512, vs. the paper's 64→128→256→512/1024) to keep CPU training time
reasonable. The encoder/decoder structure itself is unchanged.

## Files

- `model.py` — `UNet` (4-stage encoder/decoder with skip connections).
- `example.py` — trains on real Oxford-IIIT Pet segmentation masks
  (`--device {auto,cpu,cuda,mps}`).
- `example.ipynb` — same walkthrough, plus real image/mask/prediction plots.

## Run it

```bash
pip install -e .
python models/unet/example.py --device auto
# or open models/unet/example.ipynb
```
