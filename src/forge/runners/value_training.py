"""Runner for amortized-control artifacts (value / FBSDE): train + checkpoint, no sampling.

The value model has no reverse process to sample — its lifecycle is "fit a landscape, write a
checkpoint" that a controller later consumes (Invariant 6). Reuses TrainingRunner's loop and
self-contained checkpoint; ``evaluate`` reports the training loss instead of sampling.
"""

from __future__ import annotations

from ..core.registry import register
from .training import TrainingRunner


@register("runner", "value_training")
class ValueRunner(TrainingRunner):
    def evaluate(self, *args, **kwargs) -> dict:
        recent = self._last_losses[-50:]
        return {"final_loss": float(sum(recent) / len(recent))} if recent else {}
