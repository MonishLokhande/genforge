# examples

This directory plays **two roles**:

- **The bundled paradigm catalog** — `methods/`, `samplers/`, `schedules/`, `models/`, `costs/`,
  `control/`, `metrics/`, `runners/` hold every concrete implementation moved out of the installed
  wheel (flow matching / OT-CFM, discrete D3PM / MDLM / SEDD, the DDIM / interpolant / τ-leaping
  samplers, temporal-UNet / transformer models, guidance / projection / CBF controllers, …). The
  framework keeps only **one reference path** in-tree; an experiment loads the rest with
  `plugins: [examples]`. Importing the `examples` umbrella registers the whole catalog — that is
  what `plugins: [examples]` and `forge list` rely on.
- **A worked "add your own" script** (`custom_model_and_method.py`, below) — the template for
  registering a component from *outside* the tree entirely: one decorated class, zero wiring changes.

## `custom_model_and_method.py`

A self-contained script that registers a new **model** (`siren`, a sine-activated field) and a new
**method** (`logcosh`, a log-cosh denoising objective), then drops them into the built-in 2-D stack
and trains:

```bash
uv run python examples/custom_model_and_method.py
# → eval: {'n': 2000.0, 'mode_coverage': 0.98, 'radius': 0.6}
```

`mode_coverage ≈ 0.98` = your model+method learned the bimodal target.

### The whole pattern

```python
from forge.core.interfaces import Model, Method   # the contracts
from forge.core.registry import register          # the discovery hook

@register("model", "siren")
class Siren(Model):
    output_type = "eps"                 # the ONLY thing the rest of the stack reads off a model
    def forward(self, x, t, cond=None): ...

@register("method", "logcosh")
class LogCosh(Method):
    def __init__(self, schedule, space, t_eps=1e-3):   # deps injected at construction (Inv 4)
        super().__init__(schedule, space)
    def loss(self, model, x0, cond=None, generator=None): ...   # uses the 3 primitives (§2)
```

That is the entire contract. A new `model` / `method` / `space` / `schedule` / `sampler` /
`controller` is one `@register`ed class implementing the matching ABC — never an edit to the
builder, the registry, or any sibling component.

Two things the contracts hand you for free (don't reimplement them — Invariant 3):
- **`schedule`** owns all output-type math. `schedule.regression_target(model.output_type, …)` and
  `schedule.loss_weight(model.output_type, …)` mean the *same* `logcosh` trains an `eps`, `x0`,
  `score`, or `velocity` model with no branching.
- **`space.forward_sample(x0, t, schedule)`** is the forward primitive — your method calls it
  instead of hand-rolling `q(x_t | x_0)`.

## Running it the Hydra way (the production route)

The script builds from an inline `dict`. Real experiments compose the identical
`<category>: {name, params}` leaves from YAML and declare your module in `plugins:` so its
`@register` decorators fire at build time:

```yaml
# experiment/distributions/mine/base.yaml      →  forge train experiment=distributions/mine/base
# @package _global_
defaults:
  - /experiment/distributions/ddpm/base   # reuse env / sampler / runner unchanged
  - _self_
plugins:                                   # Hydra REPLACES lists — re-list the base's env plugin too
  - envs.distributions
  - examples.custom_model_and_method       # importable: examples/ has __init__.py, repo root is on sys.path
model:  {name: siren,   params: {dim: 2, output_type: eps}}
method: {name: logcosh, params: {}}
```

The same file works as both the runnable demo and the plugin module — the registrations run on
import, and the training demo is guarded by `if __name__ == "__main__"`.
