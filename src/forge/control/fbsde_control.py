"""FBSDEControl — consume an FBSDE-learned value (cost-to-go) to steer sampling.

Same amortized machinery as `ValueGuidance` (loads the value from a checkpoint, never imports the
method), but descends the cost-to-go: the optimal control follows ``−∇V`` (``sign = −1``).

FBSDE control belongs on the ``drift`` surface (``u* = Gᵀ∇V`` added to b_θ). This implementation
keeps it on the inherited ``x0`` surface (a −∇V shift of x̂₀) with identical behavior; drift-level
FBSDE is future work (it depends on the dual-surface refactor landing first).
"""

from __future__ import annotations

from ..core.registry import register
from .value_guidance import AmortizedValueController


@register("control", "fbsde_control")
class FBSDEControl(AmortizedValueController):
    surface = "x0"  # TODO: reclassify to the "drift" surface.

    def __init__(self, value_checkpoint: str, cost=None, scale: float = 1.0, sigma_weight: bool = True):
        # Descend the cost-to-go: optimal control is −∇V.
        super().__init__(value_checkpoint, cost=cost, scale=scale, sigma_weight=sigma_weight, sign=-1.0)
