"""Flow-matching (rectified-flow) velocity objective.

Same three-primitive shape as DDPM: draw ``x_t`` with the forward primitive, then regress the model
onto the schedule's velocity target for the model's ``output_type`` (the schedule owns the path
math — Invariant 3). With a velocity-output model on a `LinearFlow` schedule the target is the
clean linear-interpolant velocity ``x_data − ε``.
"""

from __future__ import annotations

from typing import Optional

import torch

from forge.core.interfaces import Method, Model
from forge.core.registry import register


@register("method", "flow_matching")
class FlowMatching(Method):
    def __init__(self, schedule, space, t_eps: float = 1e-3):
        super().__init__(schedule, space)
        self.t_eps = float(t_eps)

    def loss(
        self,
        model: Model,
        x0: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        b = x0.shape[0]
        # t ~ U(0, 1 − t_eps): keep σ(t) bounded away from 0 so the velocity target is finite.
        t = torch.rand(b, device=x0.device, generator=generator) * (1.0 - self.t_eps)
        xt = self.space.forward_sample(x0, t, self.schedule, generator=generator)
        eps = self.schedule.eps_from_x0(xt, x0, t)
        target = self.schedule.regression_target(model.output_type, x0=x0, eps=eps, xt=xt, t=t)
        pred = model(xt, t, cond)
        w = self.schedule.loss_weight(model.output_type, t).reshape(-1, *([1] * (x0.ndim - 1)))
        return torch.mean(w * (pred - target) ** 2)
