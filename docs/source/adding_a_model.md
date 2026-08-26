# Adding your own model

A practical checklist for adding a new model to this repo, following the
same conventions every one of the sixteen existing models already uses.

## 1. Files every model needs

```
models/<name>/
├── model.py         the architecture, hand-written (nn.Linear/nn.Conv only)
├── example.py       argparse CLI, trains + evaluates, prints one RESULT: line
├── example.ipynb    same walkthrough, as a notebook
└── README.md        idea in one paragraph, file list, how to run it
```

`model.py` should be **self-contained**: write your building blocks again
in this file rather than importing them from another model's `model.py`,
even if nearly identical to one that already exists -- `models/liquidode`
duplicates `models/odenet`'s RK4-integrated ODE block rather than
importing it, on purpose, so each model directory stays readable on its
own.

## 2. The `RESULT:` line

Every `example.py` must print exactly one line in this format at the end
of its run:

```
RESULT: model=<name> metric_name=<name> metric=<value> params=<n> train_time_s=<value>
```

`benchmarks/run_cluster.py` regex-parses this line (see its `RESULT_RE`)
to build comparison tables across a cluster of models that share a task.
Pick whatever `metric_name` fits your task; it doesn't need to match other
models' metric names, only the format above.

## 3. Benchmark clusters

Add a `benchmarks/configs/<cluster>_suite.yaml`:

```yaml
# One-line description of what this cluster tests and why these models
# are grouped together.
models:
  - your_model_name
epochs: 20
```

If your model shares a real, fair comparison with existing ones on
identical data (like the eight `cifar_suite.yaml` classifiers, all trained
identically for a direct accuracy/params/train-time comparison), add it
there instead of making a new cluster -- most others here (`lenet`, `unet`,
`yolo`, `nca`) are solo clusters since their tasks/datasets aren't
comparable to anything else. **Also add your cluster name to
`benchmarks/run_cluster.py`'s `--cluster` argparse `choices=[...]` list**
-- this repo hasn't switched to auto-discovering cluster files from
`benchmarks/configs/`, so that list needs a manual edit every time.

## 4. Dataset loaders

Add a `load_<dataset>()` function to `cnn_playground/data/datasets.py`,
export it from `cnn_playground/data/__init__.py`, and import it in your
`example.py` as `from cnn_playground.data import load_<dataset>`. Real
data only -- the one deliberate exception in this repo is `models/nca`'s
procedurally-generated RGBA target (no copyrighted emoji asset used, and
no public dataset exists for "grow this exact pattern" in the first
place), stated explicitly in its docs; a synthetic placeholder standing in
for data that *does* exist publicly is not acceptable.

## 5. Docs page

Add `docs/source/models/<name>.md`: intro citing the real paper (or
stating plainly if this is an assembled combination -- see
{doc}`models/obfconv`'s or {doc}`models/liquidode`'s honesty notes as the
pattern to follow), `## The equation` (the real math, in LaTeX), `## How
it's built` (a verbatim code excerpt, not paraphrased), a pre-rendered PNG
diagram plus an identical live `.. plot::` block using
`cnn_playground.utils.diagrams`, an explicit `## Simplifications vs. the
paper` section, `## Try it`, `## References`. Render the diagram via a
throwaway script and look at the PNG before committing.

Then wire it in: `docs/source/index.md` toctree, `docs/source/
model_comparison.md` row, `papers/references.bib` + `papers/README.md`,
root `README.md` model table + count, `datasets/README.md` +
`benchmarks/README.md`.

## 6. Verify before committing

- Smoke test: random input through `forward()` and `.backward()`, check
  output shape and that every parameter has a non-`None` `.grad`.
- A real training run (or an honestly-labeled partial run under load --
  state the exact numbers and why, never fabricate one).
- `python3 -m sphinx -b html docs/source docs/_build_test -q` builds
  clean (a few pre-existing cosmetic warnings -- duplicate citations on
  the combined `papers.md` page, `myst.xref_missing` on plain relative
  links -- are expected; a *new* warning or error is not), then delete
  `docs/_build_test`.

## 7. Try it

```bash
python models/<name>/example.py --device auto
python benchmarks/run_cluster.py --cluster <cluster> --device auto
```
