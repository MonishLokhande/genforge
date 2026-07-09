"""Stochastic-interpolant SDE sampler (Albergo & Vanden-Eijnden, 2023).

Integrates the family of SDEs that all share the interpolant's marginals ρ_t, forward in time from
the Gaussian base (t=0) to data (t=1):

    dx = [ b(x,t) + ε(t)·s(x,t) ] dt + sqrt(2 ε(t)) dW

where b is the velocity and s the score — both obtained output-type-agnostically from the schedule
(Invariant 3), so the same sampler drives a velocity / x₀ / score model with no branching of its own.
``ε(t) ≥ 0`` is the **free diffusion coefficient**: the knob that interpolates from the
probability-flow ODE (``epsilon=0``, identical to the `flow` sampler's Euler integrator) to an
increasingly stochastic sampler, all sharing the SAME marginals — the defining feature of the
stochastic-interpolants framework.

Default profile ``ε(t) = epsilon · σ(t)²``: vanishes at the data endpoint (σ(1)=0), so both the
score correction (s ~ O(1/σ)) and the injected noise stay bounded — no endpoint blow-up.
# constant-scaled σ² profile; drop in a t-dependent ε(t) here if a schedule wants one.

Cont/disc- and output-type-agnostic (Invariants 1 & 3). Prefer a velocity-output model: at the base
endpoint α(0)=0 makes the ε/x₀ conversions ill-posed (same edge the `flow` sampler documents).
"""

from __future__ import annotations

import torch

from ..core.interfaces import Sampler
from ..core.registry import register
from ..utils.torch_utils import expand_like as _expand


@register("sampler", "interpolant")
class InterpolantSampler(Sampler):
    def __init__(self, model, schedule, space, control=None, *, epsilon: float = 1.0):
        super().__init__(model, schedule, space, control)
        if epsilon < 0.0:
            raise ValueError(f"epsilon (diffusion coefficient scale) must be >= 0, got {epsilon}.")
        self.epsilon = float(epsilon)

    def step(self, x: torch.Tensor, t, s, cond=None) -> torch.Tensor:
        t = torch.as_tensor(t)
        s = torch.as_tensor(s)
        dt = s - t                                   # > 0: forward integration, base → data
        pred = self.model(x, t, cond)

        if self.control is None:
            b = self.schedule.velocity_from(self.model.output_type, x, pred, t)
            score = self.schedule.score_from(self.model.output_type, x, pred, t)
        else:
            # Bend the clean estimate (Invariant 6), then re-derive BOTH fields consistent with it.
            x0 = self.schedule.x0_from(self.model.output_type, x, pred, t)
            x0 = self._apply_control(x0, x, t, cond)
            b = self.schedule.velocity_from_x0(x, x0, t)
            score = self.schedule.score_from_x0(x, x0, t)

        eps_t = _expand(self.epsilon * self.schedule.sigma(t) ** 2, x)   # ε(t) = ε₀·σ(t)²
        x_s = x + (b + eps_t * score) * dt
        if self.epsilon > 0.0:
            noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=self._generator)
            x_s = x_s + torch.sqrt(torch.clamp(2.0 * eps_t * dt, min=0.0)) * noise
        return self._apply_conditioning(x_s, cond)
