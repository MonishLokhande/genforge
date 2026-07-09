"""DDPM denoising objective (the time-discretization of the score SDE).

Uses the three primitives only: it samples ``x_t`` with the **forward primitive**
``space.forward_sample`` and recovers the realized noise via a **schedule** conversion, then
regresses the model onto the schedule-provided target for the model's ``output_type``. The method
never does α/σ math and never branches on output type (Invariant 3) — so the same DDPM trainer
learns an ε-model, an x₀-model, or a score-model unchanged.
"""

from __future__ import annotations

from typing import Optional

import torch

from ..core.interfaces import Method, Model
from ..core.registry import register
from .criterion import MSECriterion


@register("method", "ddpm")
class DDPM(Method):
    """DDPM regression. The per-element penalty is a swappable `criterion` (default MSE); inject
    ``criterion=huber`` for the smooth-L1 objective that used to be the bespoke ``ddpm_huber``."""

    def __init__(self, schedule, space, t_eps: float = 1e-3, criterion=None):
        super().__init__(schedule, space, criterion=criterion or MSECriterion())
        self.t_eps = float(t_eps)

    def loss(
        self,
        model: Model,
        x0: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        b = x0.shape[0]
        # t ~ U(t_eps, 1)
        t = torch.rand(b, device=x0.device, generator=generator) * (1.0 - self.t_eps) + self.t_eps
        # Forward primitive: x_t ~ q(x_t | x_0). Recover the exact ε that produced it (schedule).
        xt = self.space.forward_sample(x0, t, self.schedule, generator=generator)
        eps = self.schedule.eps_from_x0(xt, x0, t)
        target = self.schedule.regression_target(model.output_type, x0=x0, eps=eps, xt=xt, t=t)
        pred = model(xt, t, cond)
        # SNR weighting (from the schedule) makes the objective equivalent for any output_type. It's a
        # per-sample weight; the criterion reduces over the feature dims and applies it, so this equals
        # the old `mean(w_broadcast * (pred - target)**2)` for MSE and its Huber analogue for `huber`.
        w = self.schedule.loss_weight(model.output_type, t)
        return self.criterion(pred, target, weight=w)
