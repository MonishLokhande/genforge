"""Held-out likelihood metrics — score a NORMALIZED held-out batch through model + method.

Data-driven: they IGNORE the generated ``samples`` and require ``held_out`` (the normalized val
batch the runner provides). ``ValLoss`` is a universal generalization signal (any method).
``Perplexity`` is meaningful only where ``method.loss`` is a per-token NELBO/CE (discrete LMs) — it
HARD-RAISES on a continuous denoising loss rather than reporting a meaningless number (a denoising
MSE is not a likelihood). Discreteness is detected via the method's ``nelbo`` capability flag, not a
cont/disc branch (set on D3PM/MDLM).
"""

from __future__ import annotations

import math

import torch

from forge.core.interfaces import Metric
from forge.core.registry import register
from forge.utils.seeding import make_generator


def _mc_nll(model, method, held_out, n_mc: int, seed: int) -> float:
    """MC-average `method.loss(model, held_out)` over `n_mc` t-draws, in eval mode (no grad)."""
    if held_out is None:
        raise ValueError(
            "held-out (data-driven) metric received held_out=None — configure val_frac>0 (a "
            "validation split), or use `forge eval checkpoint=` rather than `samples=`."
        )
    if model is None or method is None:
        raise ValueError("held-out metric needs both `model` and `method` injected.")
    gen = make_generator(seed, held_out.device)
    was_training = model.training
    model.eval()
    total = 0.0
    with torch.no_grad():
        for _ in range(n_mc):
            total += float(method.loss(model, held_out, generator=gen))
    if was_training:
        model.train()
    return total / n_mc


@register("metric", "val_loss")
class ValLoss(Metric):
    """Held-out loss (generalization signal) — MC-averaged `method.loss` on the val batch."""

    def __init__(self, environment=None, model=None, method=None, dataset=None, schedule=None,
                 n_mc: int = 4, seed: int = 0):
        super().__init__(environment, model, method, dataset, schedule)
        self.n_mc = int(n_mc)
        self.seed = int(seed)

    def __call__(self, samples=None, held_out=None) -> dict:
        return {"val_loss": _mc_nll(self.model, self.method, held_out, self.n_mc, self.seed)}


@register("metric", "perplexity")
class Perplexity(Metric):
    """Held-out bits-per-token + perplexity for discrete LMs (method.loss = per-token NELBO/CE)."""

    def __init__(self, environment=None, model=None, method=None, dataset=None, schedule=None,
                 n_mc: int = 4, seed: int = 0):
        super().__init__(environment, model, method, dataset, schedule)
        self.n_mc = int(n_mc)
        self.seed = int(seed)

    def __call__(self, samples=None, held_out=None) -> dict:
        if not getattr(self.method, "nelbo", False):
            raise ValueError(
                f"perplexity needs a per-token-NELBO method (a discrete LM: d3pm/mdlm); "
                f"{type(self.method).__name__} is not one — its loss is not a likelihood. "
                "Use val_loss for a generalization signal on continuous methods."
            )
        nll = _mc_nll(self.model, self.method, held_out, self.n_mc, self.seed)   # nats/token
        return {"bits_per_token": nll / math.log(2.0), "perplexity": math.exp(nll)}
