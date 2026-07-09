"""Concrete `Criterion`s — the per-sample regression penalty a `Method` reduces with.

Each criterion reduces the per-element discrepancy to ONE scalar per sample (mean over all non-batch
dims), then to a scalar loss — optionally reweighting each sample by ``weight`` first. Reducing over
all non-batch dims (not just the feature axis) is what lets a criterion stand in for a method's
per-element weighted mean at any data rank (2-D points, T×d trajectories, ...): with a per-sample
``weight`` it equals ``mean(weight_broadcast * per_element)`` exactly.

Two penalties ship: plain ``mse`` and a ``huber`` (smooth-L1) that grows linearly once a residual
exceeds ``delta`` so a few large errors don't dominate. `DDPM` reduces through whichever is injected
(defaulting to ``mse``), so ``method=ddpm criterion=huber`` replaces the old bespoke ``ddpm_huber``.
"""

from __future__ import annotations

import torch.nn.functional as F


from ..core.interfaces import Criterion
from ..core.registry import register


def _per_sample(per_element):
    """Reduce a per-element loss to one scalar per sample (mean over every non-batch dim)."""
    return per_element.reshape(per_element.shape[0], -1).mean(dim=1)


@register("criterion", "mse")
class MSECriterion(Criterion):
    def __call__(self, pred, target, weight=None):
        loss = _per_sample((pred - target).pow(2))
        return (weight * loss).mean() if weight is not None else loss.mean()


@register("criterion", "huber")
class HuberCriterion(Criterion):
    def __init__(self, delta: float = 1.0):
        if delta <= 0.0:
            raise ValueError(f"Huber delta must be > 0, got {delta}.")
        self.delta = float(delta)

    def __call__(self, pred, target, weight=None):
        loss = _per_sample(F.huber_loss(pred, target, delta=self.delta, reduction="none"))
        return (weight * loss).mean() if weight is not None else loss.mean()
