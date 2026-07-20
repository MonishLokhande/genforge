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

## Adding your own algorithm (method / sampler / schedule / control)

The examples above are infrastructure; the generative machinery joins the same way. Integration
stays free — one `@register` class plus an optional leaf, no wiring — so the real cost is the
algorithm itself, which the spec targets at ~30–60 readable lines for a space, schedule, method,
sampler, or controller (the shipped ones bear this out: `ddim.py` 45 lines, `guidance.py` 35). A new
sampler implements one reverse increment and stays output-type-agnostic by going through the
schedule (Invariant 3):

```python
from forge.core.interfaces import Sampler
from forge.core.registry import register

@register("sampler", "my_sampler")
class MySampler(Sampler):                       # __init__(model, schedule, space, control=None) injected
    def step(self, x, t, s, cond=None):
        x0 = self.schedule.x0_from_eps(x, self.model(x, t, cond), t)   # works for any output_type
        ...                                                            # one increment x_t -> x_s
        return x_s
```

- **in-tree:** drop it in `src/forge/samplers/`, add the module to `_BUILTIN_MODULES` in
  `src/forge/core/builder.py`, add a `src/forge/configs/sampler/my_sampler.yaml` leaf.
- **as a plugin:** keep it in your package, declare `plugins: [your_pkg.samplers]`, select
  `sampler: {name: my_sampler, params: {...}}`. No fork of forge.

`method`, `schedule`, and `space` follow the identical shape against their contracts in
`core/interfaces.py`. A **controller** adds two things: it declares `surface = "x0" | "drift"` and
implements the matching `modify_x0` / `modify_drift`, and its `prepare(preprocessor)` maps the cost
into normalized coordinates at sample time (Invariant 8) — see `src/forge/control/` for worked
controllers. What none of them may break is the invariants they run under (§3): normalized
coordinates inside the membrane, no continuous/discrete branching outside `space`/`schedule`, and
output-type conversions only through the schedule.

## Adding a runner (custom loop or optimizer)

The runner owns the lifecycle — fit the membrane, train, checkpoint, sample, evaluate — and it is a
registered component like everything else: `@register("runner", name)`, discovered by the builder,
selected with `runner=<name>`. The bundled runners show the pattern: `PlanningRunner` and
`PolicyTrainingRunner` subclass `TrainingRunner` and override only what differs (goal-conditioned
sampling; rollout evaluation).

Most training-loop tweaks are a small subclass, not a rewrite, because `TrainingRunner` exposes
seams. A different **optimizer** — SGD, Lion, an 8-bit AdamW — is the canonical case: override
`_build_optimizer`, nothing else.

```python
import torch
from forge.core.registry import register
from forge.runners.training import TrainingRunner

@register("runner", "sgd_training")
class SGDTrainingRunner(TrainingRunner):
    def _build_optimizer(self, params):        # params = the trainable params the base already filtered
        return torch.optim.SGD(params, lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
```

Select it with `runner=sgd_training`; checkpointing, EMA, warmup/schedule, and resume are inherited
unchanged. This is why the built-in optimizer knob stays deliberately two-choice (`adam`/`adamw`): a
third optimizer is a five-line runner, not another branch in the core loop.

- **in-tree:** drop it in `src/forge/runners/`, add the module to `_BUILTIN_MODULES` in
  `src/forge/core/builder.py`, add a `src/forge/configs/runner/sgd_training.yaml` leaf.
- **as a plugin:** keep it in your package, declare `plugins: [your_pkg.runners]`, select
  `runner: {name: sgd_training, params: {...}}`. No fork of forge.

## Adding a metric

A metric implements the `Metric` contract — `__call__(samples, held_out) -> {name: float}` — and
declares the components it needs by name in `__init__`; the builder injects them. Sample-driven
metrics score the generated `samples` (raw units) against a reference; data-driven metrics score the
`held_out` batch (normalized) through the model (needs `runner.params.val_frac>0`).

```python
import torch
from forge.core.interfaces import Metric
from forge.core.registry import register

@register("metric", "mean_gap")
class MeanGap(Metric):
    def __init__(self, environment=None, model=None, method=None, dataset=None, schedule=None, scale: float = 1.0):
        super().__init__(environment, model, method, dataset, schedule)
        self.scale = scale

    def __call__(self, samples=None, held_out=None) -> dict:
        ref = self.environment.sample(len(samples))       # a reference draw of true samples
        return {"mean_gap": self.scale * float((samples.mean(0) - ref.mean(0)).norm())}
```

Injection is not limited to the ABC's defaults — name **any** already-built component in `__init__`
(`environment`, `model`, `method`, `schedule`, `dataset`, even `preprocessor`) and the builder wires
it. A metric that cannot run on the given config should **raise**, not return `{}` — a metric a user
asked for must not silently vanish from `metrics.json`.

- **in-tree:** drop it in `src/forge/metrics/`, add it to `_BUILTIN_MODULES`, add a
  `src/forge/configs/metric/mean_gap.yaml` leaf; bundle several with `metric_set`.
- **as a plugin:** keep it in your own module, declare `plugins: [your_pkg.metrics]`, and select
  `metric: {name: mean_gap, params: {...}}`. Readings land in `metrics.json` — no fork of forge.

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

### Image observations

Camera frames are **conditioning**, so they never cross the normalization membrane (Invariant 9):
the membrane touches only the generated quantity `x`, while a vision model owns its own image and
proprio normalization. Frames stay `uint8` all the way to the model, which does the `/255` — 4x less
host→device traffic than float32.

An adapter opts in by accepting `image_keys` and yielding an `"images"` entry per episode, shaped
`(T, n_cam, C, H, W)` uint8; it also exposes `stack_env_images(obs)` for the rollout side. The
dataset then emits a **dict** `cond` instead of a flat vector:

```python
cond = {"obs_images":  ...,   # (B, To, n_cam, C, H, W) uint8
        "obs_history": ...}   # (B, To, proprio_dim)  — rank-3, NOT flattened
```

`MultiStepWrapper` keeps a parallel frame deque and `PolicyWrapper` builds the same dict at rollout,
so train-time and rollout-time conditioning are identical by construction. See
`envs/robotics/robomimic/adapter.py` and the `experiment/robotics/vision/can_image_ddpm` leaf.

!!! warning "A vision policy's `obs_keys` is not the low-dim default"
    Vision policies drop privileged state (the camera is meant to supply it), so set `obs_keys`
    explicitly. Inheriting a low-dim default silently changes `proprio_dim` and therefore the
    model's `cond_dim`.

### Datasets larger than RAM

A dataset that cannot be preloaded publishes `supports_fast_path = False`; the runner then feeds it
through a `DataLoader` with `runner.params.workers` instead of the default in-RAM index path. The
dataset never builds a loader itself — it declares a capability and the runner decides.

Two rules make that safe:

- **`batch(idx)` must be a pure function of `idx`.** Batch indices are derived from `(seed, step)`,
  so a prefetching loader reading ahead cannot desync from the training step and resume stays
  bit-identical (Invariant 5). Randomness inside `batch()` would come from worker RNG — not a
  function of `step`, and not checkpointed — and would silently break resume.
- **Augment in the main process**, via an optional `augment(cond, generator)` on the dataset. The
  runner calls it with a dedicated, checkpointed generator, so it is the only consumer and cannot
  drift.

## How `plugins:` loading works

`build(cfg)` imports the built-in framework modules, then imports each module named in the
experiment's `plugins:` list (so its `@register` decorators fire) before constructing anything. With
no `plugins:` declared, it falls back to importing every bundled env. The loader (`core/plugins.py`)
puts the repo root on `sys.path` first, since the repo-root `envs/` tree is not part of the installed
`genforge` package.

## Installable plugins (out-of-tree packages)

To ship components in a **separate pip-installable package** — not a folder in this repo — advertise
them with a `forge.plugins` entry point. Any such package installed in the environment is discovered
automatically: it appears in `forge list` and is usable at train/sample time **without** a `plugins:`
line and **without editing genforge**.

```toml
# your package's pyproject.toml
[project.entry-points."forge.plugins"]
my_forge_ext = "my_forge_ext"   # value = a module to import; importing it fires your @register calls
```

```python
# my_forge_ext/__init__.py
from forge.core.registry import register

@register("schedule", "my_schedule")
class MySchedule(...):
    ...
```

`pip install my-forge-ext` → `forge list` now shows `my_schedule`. Discovery uses the standard
Python entry-point mechanism (as pytest/flake8 do); genforge never imports arbitrary installed
packages — a package only participates by declaring the entry point. A plugin that fails to import
is warned about and skipped, so one broken package can't break discovery for the rest.