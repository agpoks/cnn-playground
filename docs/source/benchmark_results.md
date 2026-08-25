# Benchmark: the five new models, head to head

An apples-to-apples comparison of the five models added this session
({doc}`models/odenet`, {doc}`models/liquidode`, {doc}`models/legendrekan`,
{doc}`models/obfconv`, {doc}`models/nca`), with {doc}`models/resnet` included
as a reference baseline. Produced by
[`benchmarks/compare_new_models.py`](https://github.com/agpoks/cnn-playground/blob/main/benchmarks/compare_new_models.py).

## Methodology

- **Hardware**: CPU only (this machine), `OMP_NUM_THREADS=8`.
- **Group A** (resnet, odenet, liquidode, legendrekan, obfconv): identical
  real CIFAR-10 data, identical batch size (128) and a fixed, identical
  epoch budget (**2 epochs**) across all five models -- each model uses its
  own already-tuned default learning rate (a legitimate per-model choice,
  not an unfairness). obfconv uses its default `basis="laguerre"`.
- **Group B** (nca): a different task entirely (grow a target pattern from
  a seed cell, not classification) -- not comparable to Group A's accuracy
  column. Trained for 300 iterations, matching the scale of its own
  from-scratch verification run.
- **Params**: total trainable parameters.
- **Size**: on-disk `state_dict` size (`torch.save`), MB.
- **MMACs**: millions of multiply-accumulates for one single-image forward
  pass, counted via forward hooks on every `nn.Conv2d`/`nn.Linear` only
  (ignores BatchNorm/GroupNorm/activations/elementwise ops -- standard
  practice for this kind of relative comparison; not a substitute for a
  full profiler).
- **Inference latency**: CPU, eval mode, mean over 50 forward passes after
  10 warmup passes, at batch=1 and batch=64.
- **These are NOT converged/SOTA numbers.** Two epochs is a short, fixed,
  *identical* budget chosen so training time and accuracy are directly
  comparable to each other -- useful for relative comparison (which
  architecture is cheaper, faster, or more sample-efficient under equal
  compute), not as an absolute accuracy claim. Every model here reaches
  meaningfully higher accuracy with the epoch counts in its own
  `benchmarks/configs/cifar_suite.yaml` entry.

## Group A: CIFAR-10 classifiers

| Model | Params | Size (MB) | MMACs/img | Train time (2 ep, s) | Test acc | Inference ms (bs=1) | Inference ms (bs=64) |
|---|---:|---:|---:|---:|---:|---:|---:|
| [ResNet](models/resnet) | 272,474 | 1.085 | 40.81 | 291.9 | 0.6111 | 4.83 | 44.40 |
| [ODE-Net](models/odenet) | 209,098 | 0.807 | 137.76 | 725.2 | 0.5777 | 10.83 | 119.91 |
| [Liquid-ODE](models/liquidode) | 290,058 | 1.118 | 261.82 | 1106.9 | 0.6073 | 17.01 | 215.44 |
| [Legendre-KAN-Conv](models/legendrekan) | 308,042 | 1.190 | 40.24 | 201.8 | 0.5764 | 2.65 | 37.58 |
| [OBF-Conv](models/obfconv) | 182,922 | 0.720 | 8.39 | 176.9 | 0.6517 | 2.56 | 27.99 |

## Group B: NCA (not comparable to Group A)

| Params | Size (MB) | MMACs/step | Train time (300 it, s) | Final MSE | Inference ms (1 step) | Inference ms (48-step unroll) |
|---:|---:|---:|---:|---:|---:|---:|
| 8,320 | 0.036 | 13.11 | 232.7 | 0.1626 | 1.23 | 57.42 |

## Reading the numbers

**Params don't track compute cost here.** legendrekan has the *most*
parameters (308K) of any Group-A model but among the *lowest* MMACs (40.2,
tied with resnet); odenet has fewer params than resnet (209K vs. 272K) but
over 3x the MMACs (137.8 vs. 40.8). The reason is structural, not
incidental: {doc}`models/odenet` and {doc}`models/liquidode` each evaluate
their conv net `n_steps=6` RK4 stages x 4 evaluations = 24 times per forward
pass to integrate the ODE, while legendrekan/obfconv/resnet apply each conv
layer exactly once. Liquid-ODE compounds this further -- its `dh/dt`
requires *two* small conv nets (for `tau` and the gate) instead of one, so
it has both the highest MMACs (261.8) and by far the longest training time
(1106.9s, roughly 6x resnet's) of any model here.

**OBF-Conv is the standout on cost-efficiency at this budget**: fewest
MMACs (8.39, ~5x cheaper than resnet), fastest training (176.9s), fastest
inference at both batch sizes, and the *highest* test accuracy (0.6517) of
any model in the table -- its kernel-shape constraint (a handful of
Kautz/Laguerre combination coefficients instead of a full free kernel)
acts as a strong, cheap prior here. legendrekan is close behind on
cost-efficiency (2nd-fastest training and inference) despite carrying the
most parameters, since its Legendre expansion is folded into one ordinary
conv rather than adding sequential compute.

**The continuous-depth models (odenet, liquidode) are the most expensive
per accuracy point** at this short, equal-epoch budget -- unsurprising,
since RK4 integration multiplies compute per forward/backward pass well
beyond what their param counts suggest, and a 2-epoch budget doesn't give
that extra compute much chance to pay off yet. liquidode's gating did
improve on odenet's accuracy (0.6073 vs. 0.5777) here, consistent with the
Liquid Time-Constant idea helping, but at roughly 1.5x odenet's own
already-high training cost.

None of this ranks "best architecture" in general -- it ranks these five
specific implementations under one short, equal, CPU-only training budget.
See each model's own page and {doc}`model_comparison` for what idea each
one is actually testing.
