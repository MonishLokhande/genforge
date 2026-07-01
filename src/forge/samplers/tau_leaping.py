"""Discrete reverse sampler: walk the time grid, sampling x_s ~ q(x_s | x_t) each step.

The reverse categorical is the schedule's D3PM posterior (``reverse_probs``) given the model's x₀
logits. The sampler does no discrete-specific math itself beyond drawing categoricals — the kernel
lives in the schedule (Invariant 1). Reuses the framework's sample loop (base `Sampler`).
"""

from __future__ import annotations

import torch

from ..core.interfaces import Sampler
from ..core.registry import register


@register("sampler", "tau_leaping")
class TauLeaping(Sampler):
    def step(self, x: torch.Tensor, t, s, cond=None) -> torch.Tensor:
        logits = self.model(x, t, cond)
        probs = self.schedule.reverse_probs(x, t, s, logits)
        idx = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1, generator=self._generator)
        return self._apply_conditioning(idx.reshape(x.shape).to(torch.long), cond)
