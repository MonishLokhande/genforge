"""Conditional (boundary-conditioned) training — the TRAIN-time half of goal-conditioning.

During the loss the pinned boundary timesteps (start/goal) are reset to their clean values and
EXCLUDED from the objective, so the denoiser learns to fill the interior *given fixed endpoints*. This
is a genuine training delta, NOT inpainting's sampling pin: the phase-5 comparison shows it is
load-bearing — a normally-trained model with sampling-only pinning gives a discontinuous interior
(max_step ≈ 1.3) while this conditional training gives a smooth one (≈ 0.085) at identical byte-exact
endpoints. The SAMPLING-time pin is no longer here; it lives in the control layer (the `Pin`
controller), since pinning is a hard equality constraint = control.

Output-type-agnostic via the schedule (Invariant 3). (Was `ddpm_inpainting`; renamed to what it is.)
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from ..core.interfaces import Method, Model
from ..core.registry import register


@register("method", "conditional")
class ConditionalTraining(Method):
    def __init__(self, schedule, space, pin_positions: Sequence[int] = (0, -1), t_eps: float = 1e-3):
        super().__init__(schedule, space)
        self.pin_positions = tuple(pin_positions)
        self.t_eps = float(t_eps)

    def loss(
        self,
        model: Model,
        x0: torch.Tensor,                                    # (B, H, dim)
        cond: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        b, h = x0.shape[0], x0.shape[1]
        pins = [p % h for p in self.pin_positions]
        interior = torch.ones(h, dtype=torch.bool, device=x0.device)
        interior[pins] = False                               # loss only on generated (non-pinned) steps

        t = torch.rand(b, device=x0.device, generator=generator) * (1.0 - self.t_eps) + self.t_eps
        xt = self.space.forward_sample(x0, t, self.schedule, generator=generator)
        eps = self.schedule.eps_from_x0(xt, x0, t)           # read before pinning (uses un-pinned xt)
        # Pin endpoints to clean (the model always sees correct start/goal). In-place is safe: xt is
        # the fresh forward_sample output and carries no grad in training.
        xt[:, pins, :] = x0[:, pins, :]

        target = self.schedule.regression_target(model.output_type, x0=x0, eps=eps, xt=xt, t=t)
        pred = model(xt, t, cond)
        w = self.schedule.loss_weight(model.output_type, t).reshape(-1, 1, 1)
        se = w * (pred - target) ** 2                        # (B, H, dim)
        return se[:, interior, :].mean()
