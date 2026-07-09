"""Guidance controller — first-order ∇log h on the clean-sample estimate (DPS/RePaint style).

Each reverse step, nudge x̂₀ along ∇_{x̂₀} log h (computed by autograd), never the noisy iterate.
The step is scaled by ``scale`` and, optionally, by σ(t)² so the correction is gentler near the data
(t→0) — a soft tilt rather than a hard projection. The base model is never touched (Invariant 6).
"""

from __future__ import annotations

import torch

from ..core.interfaces import Controller
from ..core.registry import register


@register("control", "guidance")
class Guidance(Controller):
    surface = "x0"

    def __init__(self, cost, scale: float = 1.0, sigma_weight: bool = True):
        super().__init__(cost)
        self.scale = float(scale)
        self.sigma_weight = bool(sigma_weight)

    def modify_x0(self, x0_hat: torch.Tensor, x: torch.Tensor, t, schedule, cond=None, context=None) -> torch.Tensor:
        with torch.enable_grad():
            z = x0_hat.detach().requires_grad_(True)
            lh = self.cost.log_h(z, t).sum()
            (grad,) = torch.autograd.grad(lh, z)
        step = self.scale
        if self.sigma_weight:
            # σ(t)² fades the correction as the estimate sharpens toward the data manifold.
            sigma = torch.as_tensor(schedule.sigma(t), device=x0_hat.device)
            step = step * (sigma**2)
        return (x0_hat + step * grad).detach()
