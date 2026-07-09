"""DDIM sampler — deterministic (η=0) few-step sampling of a score/diffusion-trained model.

One step writes ``x_s = α_s·x̂₀ + √(σ_s² − σ̃²)·ε̂ + σ̃·z`` with ``σ̃ = η·(σ_s/σ_t)·√(1 − (α_t/α_s)²)``.
Both ``x̂₀`` and ``ε̂`` come from the schedule's output-type dispatch, so the **same** sampler works
for an ε-model and an x₀-model with no branching of its own (Invariant 3). η>0 reintroduces
stochasticity; η=0 is the deterministic ODE-like sampler that needs few steps.
"""

from __future__ import annotations

import torch

from ..core.interfaces import Sampler
from ..core.registry import register
from ..utils.torch_utils import expand_like as _expand


@register("sampler", "ddim")
class DDIMSampler(Sampler):
    def __init__(self, model, schedule, space, control=None, eta: float = 0.0):
        super().__init__(model, schedule, space, control)
        self.eta = float(eta)

    def step(self, x: torch.Tensor, t, s, cond=None) -> torch.Tensor:
        pred = self.model(x, t, cond)
        x0_hat = self.schedule.x0_from(self.model.output_type, x, pred, t)
        x0_hat = self._apply_control(x0_hat, x, t, cond)   # control bends the clean estimate (Invariant 6)
        # Re-derive eps from the (possibly controlled) x̂0 so the (x0, eps) pair stays self-consistent
        # — without control this equals the model's eps (round-trip); with control it tracks it.
        eps_hat = self.schedule.eps_from_x0(x, x0_hat, t)

        a_t = _expand(self.schedule.alpha(t), x)
        s_t = _expand(self.schedule.sigma(t), x)
        a_s = _expand(self.schedule.alpha(s), x)
        s_s = _expand(self.schedule.sigma(s), x)

        sigma_ddim = (
            self.eta * (s_s / s_t) * torch.sqrt(torch.clamp(1.0 - (a_t / a_s) ** 2, min=0.0))
        )
        coef_eps = torch.sqrt(torch.clamp(s_s**2 - sigma_ddim**2, min=0.0))
        x_s = a_s * x0_hat + coef_eps * eps_hat
        if self.eta > 0.0 and float(s) > 0.0:
            noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=self._generator)
            x_s = x_s + sigma_ddim * noise
        return self._apply_conditioning(x_s, cond)
