"""Projection controller — a hard constraint expressed as control (surface="x0").

`Projection` projects the clean estimate x̂₀ onto a cost's feasible set each reverse step. The
degenerate equality case (`Pin` — inpainting / goal-conditioning) lives in ``forge.core.conditioning``
because the framework's own sample loop uses it; this example only carries the general projection.
The base model is never touched (Invariant 6).
"""

from __future__ import annotations

import torch

from forge.core.interfaces import Controller
from forge.core.registry import register


@register("control", "projection")
class Projection(Controller):
    surface = "x0"

    def modify_x0(self, x0_hat: torch.Tensor, x: torch.Tensor, t, schedule, cond=None, context=None) -> torch.Tensor:
        return self.cost.project(x0_hat)
