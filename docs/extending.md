# Extending genforge

Every component joins the framework the same way: a class decorated with `@register(category, name)`
that implements the category's contract, plus (optionally) one config leaf. The builder discovers it
through the registry — no other wiring. There are two homes for a new component:

- **in-tree** — it ships inside `src/forge/` and is imported by the builder's built-in list.
- **as a plugin** — it lives in any importable module and is loaded by an experiment's `plugins:`
  field. This is how concrete environments work, and it is **not** limited to envs — a plugin module
  can register *any* category.

## Adding a model

A model implements the `Model` contract: an `output_type` the schedule knows how to interpret, and a
`forward(x, t, cond=None)`.

```python
import torch.nn as nn
from forge.core.interfaces import Model
from forge.core.registry import register

@register("model", "mynet")
class MyNet(Model):
    def __init__(self, dim: int, hidden: int = 128, output_type: str = "eps"):
        super().__init__()
        self.output_type = output_type          # "eps" | "x0" | "score" | "velocity" | "logits"
        self.net = nn.Sequential(...)

    def forward(self, x, t, cond=None):
        return self.net(...)
```

The sampler and method stay agnostic — the schedule converts whatever `output_type` you declare. The
builder injects `schedule`/`space` automatically if your `__init__` names them.

- **in-tree:** drop the file in `src/forge/models/`, add it to `_BUILTIN_MODULES` in
  `src/forge/core/builder.py`, and add a `src/forge/configs/model/mynet.yaml` leaf
  (`name: mynet` + `params:`).
- **as a plugin:** keep the file in your own package, declare `plugins: [your_pkg.models]` in the
  experiment, and select `model: {name: mynet, params: {...}}`. No fork of forge.

## Adding an environment

Concrete data sources are plugins under `envs/<name>/`:

```
envs/<name>/
├── __init__.py      # exports Environment / Dataset / Processor; importing it fires the @register decorators
├── environment.py   # the raw data source: sample(n, generator) -> (n, *shape)  [or rollouts() for trajectories]
├── dataset.py       # a BaseDataset (gather/fit_tensor/num_items/sample_shape); optional if you reuse envs.common
└── processor.py     # a BaseProcessor (env-specific PRE-membrane encoding: tokenize / pack / window)
```

The data-boundary contracts live in `src/forge/core/protocols.py`:

- **`BatchProtocol`** — a batch as it enters the loop: `x0` (float32 or int64), optional `cond`,
  optional `mask`.
- **`BaseDataset`** — `gather(idx)`/`fit_tensor`/`num_items`/`sample_shape` plus a `batch(idx)`
  entry point and `validate_batch`.
- **`BaseProcessor`** — env-specific encoding *before* the normalization layer. This is **distinct**
  from the `Preprocessor` normalization layer / "membrane" (`standardize`/`minmax`): a processor
  tokenizes/packs, a preprocessor normalizes (centers and rescales). Don't merge them.

An experiment names your env/dataset/processor inline — `environment: {name: mything, params: {...}}`,
`dataset: {name: ...}` — and lists the package under `plugins:` so the `@register` decorators fire;
there is no separate config-group overlay. Generic, env-agnostic datasets (e.g. the `distribution`
"sample from any environment" dataset) live in `envs/common/` and are always available.

Wire it into an experiment with a `plugins:` line:

```yaml
plugins:
  - envs.mything
environment: { name: mything, params: {...} }
```

`forge list` auto-discovers the bundled `envs/*` packages so the full catalog always prints.

## How `plugins:` loading works

`build(cfg)` imports the built-in framework modules, then imports each module named in the
experiment's `plugins:` list (so its `@register` decorators fire) before constructing anything. With
no `plugins:` declared, it falls back to importing every bundled env. The loader (`core/plugins.py`)
puts the repo root on `sys.path` first, since the repo-root `envs/` tree is not part of the installed
`genforge` package.