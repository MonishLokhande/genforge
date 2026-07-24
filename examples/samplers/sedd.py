"""SEDD reverse sampler — the score-based ANALYTIC reverse step (Lou et al.), NOT the x₀-posterior.

The reverse is driven by the same generator as the loss (the schedule's ᾱ-derived `rate_matrix`),
but expressed as its stable time-integral on the absorbing graph: the naive Euler form
``R̄_t(x,y)=Q_t(y,x)·s_θ`` is singular at the prior (``β=1/ᾱ→∞`` at ᾱ=0, un-masking everything in one
step with no context). The analytic absorbing reverse un-masks a bounded, gradual fraction
``p = (ᾱ_s − ᾱ_t)/(1 − ᾱ_t)`` of the masked tokens per step (=ᾱ_s at the all-mask prior → gradual),
each drawn from the concrete score over data tokens; unmasked tokens stay fixed.
"""

from __future__ import annotations

import torch

from forge.core.interfaces import Sampler
from forge.core.registry import register


@register("sampler", "sedd")
class SEDDSampler(Sampler):
    def reverse_probs(self, x: torch.Tensor, t, s, cond=None) -> torch.Tensor:
        """Per-position reverse q(x_s | x_t): masked tokens un-mask to the bounded clean-token
        distribution (softmax over data tokens), a gradual fraction per step; revealed tokens frozen."""
        mask = self.schedule.mask_index
        ab_t = float(torch.as_tensor(self.schedule.alpha_bar(t)))
        ab_s = float(torch.as_tensor(self.schedule.alpha_bar(s)))
        p_unmask = max(0.0, (ab_s - ab_t) / max(1.0 - ab_t, 1e-6))     # bounded fraction this step

        logits = self.model(x, t, cond)                              # cond threaded (else a conditional
        logits[..., mask] = float("-inf")                            # un-mask only to data tokens (SUBS)
        token = torch.softmax(logits, dim=-1)                        # bounded clean-token distribution

        probs = p_unmask * token                                     # un-mask mass, spread by score
        probs[..., mask] = probs[..., mask] + (1.0 - p_unmask)        # stay-masked mass
        # Unmasked positions are frozen (a delta at their current token).
        is_masked = (x == mask).unsqueeze(-1)
        delta = torch.zeros_like(probs).scatter(-1, x.unsqueeze(-1), 1.0)
        return torch.where(is_masked, probs, delta)

    def step(self, x: torch.Tensor, t, s, cond=None) -> torch.Tensor:
        probs = self.reverse_probs(x, t, s, cond)
        idx = torch.multinomial(
            probs.reshape(-1, probs.shape[-1]), 1, generator=self._generator
        ).reshape(x.shape)
        return self._apply_conditioning(idx.to(torch.long), cond)
