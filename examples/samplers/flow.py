"""Probability-flow ODE sampler with selectable integrators (euler / heun / midpoint).

Integrates ``dx/dt = v(x, t)`` along the schedule's grid. The velocity is obtained output-type-
agnostically: ``v = schedule.velocity_from(model.output_type, x, model(x,t), t)`` — so the same
sampler drives a velocity-model, or (via the schedule) an ε/x₀/score model, with no branching of
its own (Invariant 3). Cont/disc-agnostic: it touches only `space`/`schedule` (Invariant 1).
"""

from __future__ import annotations

import torch

from forge.core.interfaces import Sampler
from forge.core.registry import register


@register("sampler", "flow")
class FlowSampler(Sampler):
    def __init__(self, model, schedule, space, control=None, integrator: str = "heun"):
        super().__init__(model, schedule, space, control)
        if integrator not in ("euler", "heun", "midpoint"):
            raise ValueError(f"Unknown integrator {integrator!r} (euler|heun|midpoint).")
        self.integrator = integrator

    def _velocity(self, x: torch.Tensor, t, cond=None) -> torch.Tensor:
        # NOTE: with a velocity-output model (the flow default) this is exact everywhere. A non-
        # velocity model routes through velocity_from_x0, which divides by σ(t); on a flow schedule
        # σ(1)=0, so prefer a velocity model when integrating to the data endpoint t=1.
        pred = self.model(x, t, cond)
        v = self.schedule.velocity_from(self.model.output_type, x, pred, t)
        if self.control is not None:
            # Bend the clean estimate, then re-derive the velocity consistent with it (Invariant 6).
            x0 = self.schedule.x0_from(self.model.output_type, x, pred, t)
            x0 = self._apply_control(x0, x, t, cond)
            v = self.schedule.velocity_from_x0(x, x0, t)
        return v

    def step(self, x: torch.Tensor, t, s, cond=None) -> torch.Tensor:
        t = torch.as_tensor(t)
        s = torch.as_tensor(s)
        dt = (s - t)
        if self.integrator == "euler":
            x_s = x + dt * self._velocity(x, t, cond)
        elif self.integrator == "midpoint":
            v1 = self._velocity(x, t, cond)
            xm = x + 0.5 * dt * v1
            vm = self._velocity(xm, t + 0.5 * dt, cond)
            x_s = x + dt * vm
        else:
            # heun (explicit trapezoid): predictor at t, corrector at s
            v1 = self._velocity(x, t, cond)
            xe = x + dt * v1
            v2 = self._velocity(xe, s, cond)
            x_s = x + 0.5 * dt * (v1 + v2)
        return self._apply_conditioning(x_s, cond)
