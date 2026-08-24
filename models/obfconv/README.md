# OBF-Conv

**Not a single published paper.** Kautz and Laguerre orthonormal basis
functions (OBFs) are real, established tools, but exclusively in linear
*system identification* — Wahlberg, *"System identification using
Laguerre models"*, IEEE Trans. Automatic Control, 1991; Oliveira et al.,
*"An introduction to models based on Laguerre, Kautz and other related
orthonormal functions — Part I"*, IJMIC, 2011. See
[`papers/README.md`](../../papers/README.md) and `model.py` for the full
honesty note: no CNN-kernel paper applies this idea as a spatial kernel
parameterization, as far as could be found.

## Idea in one paragraph

Kautz/Laguerre OBFs classically represent an LTI system's impulse
response compactly as a short linear combination of B fixed basis
sequences, given a decay (Laguerre, one real pole) or resonance (Kautz, a
complex-conjugate pole pair) prior — instead of many free FIR taps. This
model transplants that onto a conv kernel's *spatial shape*: `OBFConv2d`
builds `n_basis**2` fixed, orthonormal 2D filters (a separable outer
product of a 1D Kautz- or Laguerre-generated basis with itself) and
learns only a small `(out_channels, in_channels, n_basis**2)` coefficient
tensor to combine them — the kernel's receptive-field *shape* is
constrained to that low-dimensional subspace, unlike an ordinary conv's
freely learned taps. Contrast with
[`models/legendrekan`](../legendrekan): that one bases a polynomial
expansion on the *pixel value* at a tap (a KAN-style edge function); this
one bases it on the *tap/spatial index* itself (the kernel's shape) —
different dimension of the convolution constrained, both real, both
honestly documented as not-a-single-paper.

## Files

- `model.py` — `generate_laguerre_basis`/`generate_kautz_basis` (real DSP
  cascade recursions, simulated directly) + `_gram_schmidt` (corrects the
  finite-kernel-size truncation error, guarantees exact orthonormality) +
  `OBFConv2d` + `OBFConvModel` (three OBF-conv blocks with strided
  downsampling between them, then GAP + linear classifier — same shape as
  `LegendreKANModel` for a fair comparison).
- `example.py` — trains on real CIFAR-10, `--basis {laguerre,kautz}`
  (`--device {auto,cpu,cuda,mps}`).
- `example.ipynb` — same walkthrough, plus a basis-sequence sanity plot.

## Run it

```bash
pip install -e .
python models/obfconv/example.py --device auto             # Laguerre basis (default)
python models/obfconv/example.py --device auto --basis kautz
# or open models/obfconv/example.ipynb
```
