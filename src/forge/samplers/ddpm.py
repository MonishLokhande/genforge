"""Ancestral DDPM sampler (VP posterior).

One reverse increment uses the closed-form Gaussian posterior ``q(x_s | x_t, x̂_0)`` of the VP
process, where ``x̂_0`` is the schedule's output-type-agnostic clean estimate of the model output.
The sampler does no α/σ or output-type math of its own — it asks the schedule (Invariant 3). It is
also cont/disc-agnostic: it calls `space`/`schedule` only (Invariant 1). The framework owns the
sample loop (in the base `Sampler`); this class implements only `step`.
"""

from __future__ import annotations

import torch

from ..core.interfaces import Sampler
from ..core.registry import register
from ..utils.torch_utils import expand_like as _expand


@register("sampler", "ddpm")
class DDPMSampler(Sampler):
    """Ancestral DDPM sampler. ``clip_denoised`` clamps the clean estimate x̂₀ to [-1, 1] each
    reverse step (diffusion_policy's ``clip_sample``). Off by default (Tier A standardize/x0-pred
    is unaffected); REQUIRED for ε-parameterized + cosine-schedule samplers over minmax→[-1,1] data
    (e.g. the Tier B closed-loop policies), where ``x0_from_eps`` divides by α→0 at high t and the
    chain otherwise diverges far out of range."""

    def __init__(self, model, schedule, space, control=None, *, clip_denoised: bool = False):
        super().__init__(model, schedule, space, control)
        self.clip_denoised = bool(clip_denoised)

    def step(self, x: torch.Tensor, t, s, cond=None) -> torch.Tensor:
        pred = self.model(x, t, cond)
        x0_hat = self.schedule.x0_from(self.model.output_type, x, pred, t)
        x0_hat = self._apply_control(x0_hat, x, t, cond)   # control bends the clean estimate (Invariant 6)
        if self.clip_denoised:
            x0_hat = x0_hat.clamp(-1.0, 1.0)         # bounded data range; tames the α→0 ε-blowup

        a_t = _expand(self.schedule.alpha(t), x)
        s_t = _expand(self.schedule.sigma(t), x)
        a_s = _expand(self.schedule.alpha(s), x)
        s_s = _expand(self.schedule.sigma(s), x)

        ratio = a_t / a_s                       # = sqrt(ᾱ_t / ᾱ_s)
        beta_step = 1.0 - ratio**2              # = 1 − ᾱ_t/ᾱ_s
        inv_var_t = 1.0 / (s_t**2)              # = 1/(1 − ᾱ_t)

        # VP posterior mean and variance (Ho et al., written in α/σ form).
        mean = (a_s * beta_step * inv_var_t) * x0_hat + (ratio * s_s**2 * inv_var_t) * x
        var = beta_step * (s_s**2) * inv_var_t

        if float(s) > 0.0:
            std = torch.sqrt(torch.clamp(var, min=0.0))
            if self.control is not None:
                # Optional temperature knob (Invariant 6): a controller may scale/shift the noise std.
                # Default modify_variance is identity, so this is a no-op unless a controller overrides it.
                std = self.control.modify_variance(std, x, t, self.schedule, cond, self._context)
            noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=self._generator)
            x_s = mean + std * noise
        else:
            x_s = mean  # final step (s = 0): the posterior collapses to x̂_0
        return self._apply_conditioning(x_s, cond)
