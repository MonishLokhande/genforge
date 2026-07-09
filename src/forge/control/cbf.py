"""CBF — a control-barrier-function safety filter on the reverse DRIFT (the rate ẋ).

The first real `drift`-surface controller (Invariant 6), and the proof the surface works: a CBF
*reads the rate*, so it has no faithful x̂₀ reduction. Given a barrier ``h(x)≥0``, it minimally
edits the drift so the barrier's forward-invariance condition ``ḣ = ∇h·ẋ ≥ −α·h(x)`` holds. For one
linear barrier this is the closed-form CBF-QP solution (project the drift onto the safe halfspace):

    slack = ∇h·drift + α·h ;  if slack < 0:  drift += (−slack/‖∇h‖²)·∇h   (else unchanged).

With the data-ward heading drift and α=1 this keeps the clean estimate feasible every step
(``h(x̂₀') ≥ (1−α)·h(xₜ) = 0``), so the formed sample lands in the feasible set.
"""

from __future__ import annotations

import torch

from ..core.interfaces import Controller
from ..core.registry import register


@register("control", "cbf")
class CBF(Controller):
    surface = "drift"

    def __init__(self, cost, alpha: float = 1.0):
        super().__init__(cost)
        self.alpha = float(alpha)

    def modify_drift(self, drift: torch.Tensor, x: torch.Tensor, t, schedule, cond=None, context=None) -> torch.Tensor:
        grad = self.cost.grad_h(x)                              # ∇h(x), (..., dim)
        h = self.cost.value(x)                                 # h(x),   (...,)
        slack = (grad * drift).sum(dim=-1) + self.alpha * h    # ḣ + α·h ≥ 0 required
        coef = torch.clamp(-slack, min=0.0) / (grad * grad).sum(dim=-1).clamp_min(1e-12)
        return drift + coef.unsqueeze(-1) * grad               # minimal safe correction
