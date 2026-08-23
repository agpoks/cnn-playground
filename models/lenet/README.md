# LeNet-5

**Paper:** LeCun, Bottou, Bengio, Haffner, *"Gradient-Based Learning
Applied to Document Recognition"*, Proceedings of the IEEE, 1998 — the
original small CNN. See [`papers/README.md`](../../papers/README.md).

## Idea in one paragraph

Two conv+pool stages, then three fully-connected layers -- the original
recipe every later model in this repo builds on. Deliberately not
modernized: it uses the paper's actual `Tanh` activations and `AvgPool`
(not ReLU/MaxPool), so it reads as the historical starting point rather
than a ReLU-ified reinterpretation. Applied to real 28x28 MNIST with
`padding=2` on the first conv (a standard adaptation of the paper's 32x32
input assumption, not a change to the architecture itself).

## Files

- `model.py` -- `LeNet5`, the two conv+pool stages + three-FC classifier.
- `example.py` -- trains on real MNIST (`--device {auto,cpu,cuda,mps}`).
- `example.ipynb` -- same walkthrough with loss/accuracy plots and sample predictions.

## Run it

```bash
pip install -e .
python models/lenet/example.py --device auto
# or open models/lenet/example.ipynb
```
