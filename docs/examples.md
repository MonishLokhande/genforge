# Examples

Worked, copy-pasteable walk-throughs of the two things you will do most: running a bundled recipe
with **different values** ([Experiments §2](experiments.md)) and adding **your own component**
([Extending](extending.md)). Everything here runs on the fast 2-D `distributions/ddpm/base`
experiment — CPU, seconds per run.

The runs below add `runner.params.steps=2` to stay quick; the `eval:` line proves the loop trains,
samples, and scores end to end. The numbers are from a 2-step smoke run — real quality needs the
recipe's full `steps: 4000`.

## 1. Run a baseline

```bash
uv run forge train experiment=distributions/ddpm/base
# [train] done. eval: {'n': 2000.0, 'mmd': 0.71, 'energy': 134.6, ...}
```

DDPM on a bimodal Gaussian mixture: `train` fits the model, then `evaluate()` draws a sample batch
and scores it against a reference draw.

## 2. Change a value

Append `key.path=value` after the selection — a key the recipe already sets is replaced in place,
no file edited and no code:

```bash
uv run forge train experiment=distributions/ddpm/base seed=1 runner.params.batch_size=128
```

Anything in the resolved config is fair game: `seed`, `runner.params.*`, `model.params.*`, ….

## 3. Swap a component

Each selectable category is a config group; name a different leaf and the builder constructs that
class instead. Here the reverse sampler goes DDPM → DDIM:

```bash
uv run forge train experiment=distributions/ddpm/base sampler=ddim
```

The training objective is unchanged — only the reverse sampler differs, so `evaluate()` draws its
samples by integrating the probability-flow path (DDIM) instead of the stochastic one (DDPM).

## 4. Add a knob the recipe omits — the `+` rule

`distributions/ddpm/base` never selects a `criterion` (DDPM defaults to MSE internally), so a plain
group override has nothing to replace and **fails loudly**:

```bash
uv run forge train experiment=distributions/ddpm/base criterion=huber
# hydra.errors.ConfigCompositionException: Could not override 'criterion'.
# No match in the defaults list. To append to your default list use +criterion=huber
```

Hydra tells you the fix — use `+` to *add* a group rather than *replace* one:

```bash
uv run forge train experiment=distributions/ddpm/base +criterion=huber   # MSE -> Huber
```

!!! tip "`=` replaces, `+` adds, `++` forces"
    Configs run in struct mode, so a bare `=` can't invent a key. Override an existing key with
    `key=value`; add a missing one with `+key=value`. Quote values with commas/brackets/spaces:
    `'model.params.dims=[64,64]'`.

## 5. Save a variant as a leaf

For a variant worth keeping, write a small delta file that inherits a family base and states only
the change. This is the shipped `distributions/ddpm/huber` leaf verbatim:

```yaml
# experiment/distributions/ddpm/huber.yaml
# @package _global_
defaults:
  - /experiment/distributions/ddpm/base   # inherit the whole recipe (slash path, no .yaml)
  - /criterion: huber                      # the one change
  - _self_                                 # last, so this file's overrides win
runner:
  params:
    ckpt_path: checkpoints/distributions/ddpm/huber.pt
```

```bash
uv run forge train experiment=distributions/ddpm/huber
```

## 6. Add your own algorithm, end to end

The bundled `huber` swap replaced MSE with another *built-in* loss. Now add a loss that isn't in the
framework at all — a log-cosh criterion — as a plugin, with no fork of forge.

**The component** — one file, implement the `Criterion` contract and register it:

```python
# example_logcosh.py  (anywhere on your Python path: an installed package, or the repo root)
import torch.nn.functional as F
from forge.core.interfaces import Criterion
from forge.core.registry import register

@register("criterion", "logcosh")
class LogCoshCriterion(Criterion):
    def __call__(self, pred, target, weight=None):
        d = (pred - target).abs()
        per = d + F.softplus(-2.0 * d) - 0.6931471805599453   # log cosh, numerically stable
        loss = per.reshape(per.shape[0], -1).mean(dim=1)       # one scalar per sample
        return (weight * loss).mean() if weight is not None else loss.mean()
```

**Wire it in** with a leaf that declares the plugin and selects the loss inline — an out-of-tree
component needs no `configs/criterion/` group leaf:

```yaml
# experiment/distributions/ddpm/logcosh.yaml
# @package _global_
defaults:
  - /experiment/distributions/ddpm/base
  - _self_
plugins: [envs.distributions, example_logcosh]   # your module alongside the env plugin
criterion: {name: logcosh}
```

```bash
uv run forge train experiment=distributions/ddpm/logcosh
# [train] done. eval: {'n': 2000.0, 'mmd': 0.71, 'energy': 135.9, ...}
```

That is the whole integration: `@register` + a `plugins:` line + an inline selection. The sampler,
method, schedule, and runner never learned a new loss existed — DDPM reduces through whichever
`criterion` is injected.

!!! note "`forge list` shows built-ins, not experiment plugins"
    `forge list` prints `criterion  huber, mse` — it never composes an experiment, so a criterion
    registered only through a leaf's `plugins:` field appears when that experiment is *built*, not
    in the catalog. To make a component list-visible everywhere, ship it as a `forge.plugins`
    entry-point package (see [Extending](extending.md#installable-plugins-out-of-tree-packages)).

---

Every other category — `sampler`, `method`, `schedule`, `space`, `cost`, `control` — extends the
identical way; see [Extending](extending.md) for the per-category contracts and the in-tree vs.
plugin vs. installable-package homes.
