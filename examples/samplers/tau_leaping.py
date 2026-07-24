"""Discrete reverse sampler: walk the time grid, sampling x_s ~ q(x_s | x_t) each step.

The reverse categorical is the schedule's D3PM posterior (``reverse_probs``) given the model's x₀
logits. The sampler does no discrete-specific math itself beyond drawing categoricals — the kernel
lives in the schedule (Invariant 1). Reuses the framework's sample loop (base `Sampler`).

NAMING — this is **not** τ-leaping. It is EXACT ancestral sampling from the x₀-parameterized reverse
posterior: one categorical draw per position per step, no approximation. Poisson τ-leaping (Campbell
et al., "A Continuous Time Framework for Discrete Denoising Models") is a different algorithm — it
holds the CTMC jump rates constant over a finite window τ and draws a Poisson number of jumps, which
can fire multiple conflicting jumps at a site. Nothing here does that. The accurate registered name
is ``x0_ancestral``; ``tau_leaping`` stays registered as a back-compat alias because existing configs
select it by name AND checkpoints record the sampler name, which the checkpoint/config fidelity
guard compares on load.
"""

from __future__ import annotations

import torch

from forge.core.interfaces import Sampler
from forge.core.registry import register


@register("sampler", "x0_ancestral")
@register("sampler", "tau_leaping")  # back-compat alias — see the naming note above
class X0Ancestral(Sampler):
    def step(self, x: torch.Tensor, t, s, cond=None) -> torch.Tensor:
        logits = self.model(x, t, cond)
        probs = self.schedule.reverse_probs(x, t, s, logits)
        idx = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1, generator=self._generator)
        return self._apply_conditioning(idx.reshape(x.shape).to(torch.long), cond)


# Old class name kept importable: `from forge.samplers.tau_leaping import TauLeaping` appears in
# ported tests and downstream code.
TauLeaping = X0Ancestral
