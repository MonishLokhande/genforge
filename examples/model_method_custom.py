"""Add your OWN model + method to genforge, with zero wiring changes (Invariant 7).

Run it:

    uv run python examples/custom_model_and_method.py

What it shows: an *external author* importing genforge, decorating two classes with
``@register(category, name)``, and dropping them straight into the built-in 2-D stack
(euclidean space / VP schedule / DDPM sampler / training runner / gaussian-mixture env).
The builder discovers them by name — nothing else in the framework is edited (Invariant 7).

The two contracts you implement against:
  - ``Model``  — a learned field; ``output_type`` tells the *schedule* how to read its output.
  - ``Method`` — a training objective; deps (schedule, space) injected at construction.

This same file doubles as a Hydra plugin: the registrations run at import, and the demo is
guarded by ``__main__``, so ``plugins: [examples.custom_model_and_method]`` in an experiment
works too (see README.md).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import forge  # the package; its public API lives in the submodules below
from forge.core.builder import build
from forge.core.interfaces import Method, Model
from forge.core.registry import register


# ── your new MODEL ──────────────────────────────────────────────────────────────────────────────
@register("model", "siren")
class Siren(Model):
    """A SIREN-style field: sine activations (not ReLU/SiLU), each scaled by ``w0``.

    A genuinely different architecture from the built-in ``mlp`` — but the framework neither knows
    nor cares: ``output_type`` is the ONLY thing the rest of the stack reads off the model, and the
    *schedule* owns every conversion of that output (Invariant 3). Set it to ``"eps"`` and this is a
    noise predictor; set ``"x0"`` and it's a denoiser — same class, no sampler/method change.
    """

    def __init__(self, dim: int = 2, hidden: int = 128, depth: int = 3,
                 w0: float = 5.0, output_type: str = "eps"):
        # w0 sets the activation frequency. The SIREN paper's image-fitting default (30) is far too
        # oscillatory for this smooth (x_t, t) → ε regression and won't optimize; ~5 suits it.
        super().__init__()
        self.output_type = output_type
        self.w0 = float(w0)
        self.t_embed = nn.Linear(1, hidden)
        self.in_proj = nn.Linear(dim + hidden, hidden)
        self.layers = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(depth - 1))
        self.out = nn.Linear(hidden, dim)
        # Canonical SIREN init (Sitzmann et al. 2020): sine nets diverge without it. First sine
        # layers ~ U(-1/fan_in, 1/fan_in); deeper ones ~ U(-√(6/fan_in)/w0, …). Pairs with the
        # standardize membrane below so inputs are unit-scale, where sine activations behave.
        for first, lin in ((True, self.t_embed), (True, self.in_proj),
                           *((False, lin) for lin in self.layers)):
            fan_in = lin.weight.shape[1]
            bound = (1.0 / fan_in) if first else (math.sqrt(6.0 / fan_in) / self.w0)
            with torch.no_grad():
                lin.weight.uniform_(-bound, bound)
                lin.bias.zero_()

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond=None) -> torch.Tensor:
        t = torch.as_tensor(t, dtype=x.dtype, device=x.device).reshape(-1, 1)
        if t.shape[0] == 1:               # a scalar grid time → broadcast over the batch
            t = t.expand(x.shape[0], 1)
        tfeat = torch.sin(self.w0 * self.t_embed(t))
        h = torch.sin(self.w0 * self.in_proj(torch.cat([x, tfeat], dim=-1)))
        for lin in self.layers:
            h = torch.sin(self.w0 * lin(h))
        return self.out(h)


# ── your new METHOD ─────────────────────────────────────────────────────────────────────────────
@register("method", "logcosh")
class LogCosh(Method):
    """Denoising objective with a log-cosh penalty (≈MSE near 0, ≈L1 in the tails).

    Written against the three primitives only: sample t, draw ``x_t`` with the
    forward primitive ``space.forward_sample``, then ask the *schedule* for the regression target
    and SNR weight. No α/σ or output-type math lives here (Invariant 3), so the SAME objective
    trains an eps / x0 / score / velocity model unchanged. (To swap only the penalty on an existing
    objective, subclass it instead — see ``methods/ddpm_huber.py``.)
    """

    def __init__(self, schedule, space, t_eps: float = 1e-3):
        super().__init__(schedule, space)          # deps injected at construction (Invariant 4)
        self.t_eps = float(t_eps)

    def loss(self, model: Model, x0: torch.Tensor, cond=None,
             generator: Optional[torch.Generator] = None) -> torch.Tensor:
        b = x0.shape[0]
        t = torch.rand(b, device=x0.device, generator=generator) * (1.0 - self.t_eps) + self.t_eps
        xt = self.space.forward_sample(x0, t, self.schedule, generator=generator)   # primitive 1
        eps = self.schedule.eps_from_x0(xt, x0, t)
        target = self.schedule.regression_target(model.output_type, x0=x0, eps=eps, xt=xt, t=t)
        pred = model(xt, t, cond)
        w = self.schedule.loss_weight(model.output_type, t).reshape(-1, *([1] * (x0.ndim - 1)))
        r = (pred - target).abs()
        per_elem = r + F.softplus(-2.0 * r) - math.log(2.0)    # log(cosh r), numerically stable
        return torch.mean(w * per_elem)


# ── run them: build the built-in 2-D stack, but with model=siren, method=logcosh ─────────────────
# An inline config is just a dict of `<category>: {name, params}` leaves — the same shape the Hydra
# experiment files compose to. The builder wires them in dependency order and returns a ready Runner.
CONFIG = {
    "plugins": ["envs.distributions"],   # load the concrete 2-D env package (registers the env)
    "space":    {"name": "euclidean", "params": {"dim": 2}},
    "schedule": {"name": "vp_linear", "params": {"beta_min": 0.1, "beta_max": 20.0}},
    "model":    {"name": "siren",     "params": {"dim": 2, "hidden": 128, "depth": 3, "output_type": "eps"}},
    "method":   {"name": "logcosh",   "params": {}},
    "sampler":  {"name": "ddpm",      "params": {}},
    "environment": {"name": "gaussian_mixture", "params": {"means": [[-2.0, 0.0], [2.0, 0.0]], "std": 0.2}},
    "dataset":  {"name": "distribution", "params": {"n_samples": 20000, "seed": 0}},
    "preprocessor": {"name": "standardize", "params": {}},   # the membrane: train in unit-scale coords
    "runner":   {"name": "training", "params": {
        "steps": 2000, "batch_size": 512, "lr": 2.0e-3, "n_sample_steps": 50,
        "n_eval_samples": 2000, "eval_radius": 0.6, "device": "cpu", "seed": 0,
    }},
}


def main() -> None:
    print(f"genforge {forge.__version__} — custom model 'siren' + method 'logcosh'\n")
    runner = build(CONFIG)                 # discovers your classes by name; no wiring changes
    runner.train()                         # trains in normalized space, samples with EMA
    metrics = runner.evaluate()            # rollout: fraction of samples landing on a real mode
    print(f"\neval: {metrics}")
    print("mode_coverage near 1.0 → your custom model+method learned the bimodal target.")


if __name__ == "__main__":
    main()
