"""DDPM objective with a Huber (smooth-L1) loss instead of MSE.

A worked example of **swapping just the loss**: `DDPM` regresses the model onto the schedule's target
with a weighted *MSE*; this subclass changes exactly one thing — the per-element penalty — to *Huber*,
which grows linearly (not quadratically) once a residual exceeds ``delta``, so a few large errors
don't dominate training.

Everything else is inherited: the forward primitive, and the output-type-agnostic target + SNR
weighting that the **schedule** provides (Invariant 3). So this loss works with ANY continuous
schedule and ANY ``output_type``, exactly like DDPM — no schedule, model, sampler, or runner change.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from ..core.interfaces import Model
from ..core.registry import register
from .ddpm import DDPM


@register("method", "ddpm_huber")
class DDPMHuber(DDPM):
    def __init__(self, schedule, space, t_eps: float = 1e-3, delta: float = 1.0):
        super().__init__(schedule, space, t_eps=t_eps)
        if delta <= 0.0:
            raise ValueError(f"Huber delta must be > 0, got {delta}.")
        self.delta = float(delta)

    def loss(
        self,
        model: Model,
        x0: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        b = x0.shape[0]
        t = torch.rand(b, device=x0.device, generator=generator) * (1.0 - self.t_eps) + self.t_eps
        xt = self.space.forward_sample(x0, t, self.schedule, generator=generator)
        eps = self.schedule.eps_from_x0(xt, x0, t)
        target = self.schedule.regression_target(model.output_type, x0=x0, eps=eps, xt=xt, t=t)
        pred = model(xt, t, cond)
        w = self.schedule.loss_weight(model.output_type, t).reshape(-1, *([1] * (x0.ndim - 1)))
        per_elem = F.huber_loss(pred, target, reduction="none", delta=self.delta)   # ← only change
        return torch.mean(w * per_elem)
